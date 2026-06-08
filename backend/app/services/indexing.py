"""Per-session index state + incremental document ingestion.

Each chat session owns an isolated IndexState (its own FAISS store / hybrid
retriever / RAG chain) so one chat's uploaded files are never retrievable by
another. All mutation goes through index_files / clear_session, keyed by
session_id.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from pathlib import Path
from typing import Any

from langchain_community.vectorstores import FAISS

from app.config import TOP_K
from app.langsmith_tracing import langsmith_traceable as traceable
from app.pipeline import load_and_chunk, load_urls_and_chunk
from app.rag.rag_chain import build_rag_chain
from app.retriever.faiss_retriever import build_merged_faiss, group_chunks_by_source
from app.retriever.hybrid_retriever import build_hybrid_retriever

from .resources import embedding_model, llm

logger = logging.getLogger(__name__)


def _file_hash(path: Path) -> str:
    """MD5 fingerprint of file content for deduplication (not security)."""
    h = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


class IndexState:
    """Mutable view of what a single session currently has indexed."""

    def __init__(self) -> None:
        self.chain: Any = None
        self.faiss_store: FAISS | None = None
        self.by_source: dict[str, list] = {}
        self.indexed_files: list[str] = []
        self.indexed_file_hashes: set[str] = set()
        self.filename_to_hash: dict[str, str] = {}
        self.indexed_url_signatures: dict[str, str] = {}

    def clear(self) -> None:
        self.chain = None
        self.faiss_store = None
        self.by_source = {}
        self.indexed_files = []
        self.indexed_file_hashes = set()
        self.filename_to_hash = {}
        self.indexed_url_signatures = {}


# Registry of per-session index states. Guarded by a lock because uploads run
# in a worker thread (asyncio.to_thread) while requests may touch the registry
# concurrently.
_sessions: dict[str, IndexState] = {}
_session_last_access: dict[str, float] = {}
_lock = threading.Lock()

# Evict index states idle for more than 2 hours.
_INDEX_TTL = 7200


def _evict_stale_sessions() -> None:
    now = time.time()
    stale = [sid for sid, t in _session_last_access.items() if now - t > _INDEX_TTL]
    for sid in stale:
        _sessions.pop(sid, None)
        _session_last_access.pop(sid, None)
    if stale:
        logger.debug("Evicted %d stale index session(s)", len(stale))


def get_state(session_id: str) -> IndexState:
    """Return the session's IndexState, creating an empty one on first use."""
    with _lock:
        st = _sessions.get(session_id)
        if st is None:
            st = IndexState()
            _sessions[session_id] = st
        _session_last_access[session_id] = time.time()
        return st


def clear_session(session_id: str) -> None:
    """Drop a session's index entirely (used on chat delete / file clear)."""
    with _lock:
        _sessions.pop(session_id, None)
        _session_last_access.pop(session_id, None)


@traceable(run_type="chain", name="rebuild_rag_chain")
def _rebuild_chain(session_id: str, state: IndexState, all_chunks: list) -> None:
    hybrid_retriever = build_hybrid_retriever(
        semantic_retriever=state.faiss_store,
        chunks=all_chunks,
        k=TOP_K,
        use_bm25=True,
        cache_key=session_id,
    )
    source_names = sorted({
        d.metadata.get("source", "")
        for d in all_chunks
        if d.metadata.get("source")
    })
    state.chain = build_rag_chain(hybrid_retriever, source_names, llm(), all_chunks)


@traceable(run_type="chain", name="ingest_pipeline")
def index_files(session_id: str, paths: list[Path]) -> int:
    """Incrementally index newly-uploaded files into a session's own index.

    Files already indexed (by content hash, not just name) are skipped so
    re-uploading the same file is a no-op. Returns the number of new chunks added.
    """
    logger.info("index_files: session=%s, %d path(s)", session_id, len(paths))
    state = get_state(session_id)

    # Hash-based dedup: identical content is skipped even if the filename differs.
    path_hashes = {p: _file_hash(p) for p in paths}
    new_paths = [p for p in paths if path_hashes[p] not in state.indexed_file_hashes]
    if not new_paths:
        logger.info("index_files: no new paths to index")
        return 0
    logger.info("index_files: indexing %d new file(s)", len(new_paths))

    new_chunks = load_and_chunk([str(p) for p in new_paths], verbose=False)
    logger.info("index_files: load_and_chunk returned %d chunk(s)", len(new_chunks))
    if not new_chunks:
        return 0

    em = embedding_model()
    if state.faiss_store is None:
        logger.debug("index_files: building merged FAISS store")
        state.faiss_store = build_merged_faiss(new_chunks, em)
    else:
        logger.debug("index_files: adding %d chunk(s) to existing FAISS store", len(new_chunks))
        state.faiss_store.add_documents(new_chunks)

    new_by_source = group_chunks_by_source(new_chunks)
    for source, chunks in new_by_source.items():
        state.by_source.setdefault(source, []).extend(chunks)

    all_chunks: list = []
    for docs in state.by_source.values():
        all_chunks.extend(docs)
    logger.info("index_files: total chunks: %d", len(all_chunks))

    _rebuild_chain(session_id, state, all_chunks)

    state.indexed_files.extend(p.name for p in new_paths)
    state.indexed_file_hashes.update(path_hashes[p] for p in new_paths)
    for p in new_paths:
        state.filename_to_hash[p.name] = path_hashes[p]
    logger.info("index_files: complete, added %d chunk(s)", len(new_chunks))
    return len(new_chunks)


@traceable(run_type="chain", name="index_urls")
def index_urls(
    session_id: str,
    urls: list[str],
    allow_domains: list[str] | None = None,
    max_depth: int | None = 2,
    max_pages: int | None = 20,
    auth_cookies: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
    render_javascript: bool = False,
    render_timeout: int = 30,
) -> int:
    """Index authenticated URLs as a session's own web corpus.

    Re-crawled pages are compared by content signature. Unchanged pages are
    skipped, while changed pages replace their previous chunks before the
    retriever is rebuilt.
    """
    logger.info("index_urls: session=%s, %d url(s)", session_id, len(urls))
    state = get_state(session_id)

    raw_chunks = load_urls_and_chunk(
        urls,
        allow_domains=allow_domains,
        max_depth=max_depth,
        max_pages=max_pages,
        auth_cookies=auth_cookies,
        headers=headers,
        render_javascript=render_javascript,
        render_timeout=render_timeout,
    )
    logger.info("index_urls: load_urls_and_chunk returned %d chunk(s)", len(raw_chunks))
    if not raw_chunks:
        return 0

    grouped_chunks = group_chunks_by_source(raw_chunks)
    changed_by_source = {
        source: chunks
        for source, chunks in grouped_chunks.items()
        if chunks
        and chunks[0].metadata.get("signature")
        != state.indexed_url_signatures.get(source)
    }
    if not changed_by_source:
        logger.info("index_urls: no changed urls to index")
        return 0

    for source in changed_by_source:
        state.by_source.pop(source, None)

    changed_chunks = [
        chunk
        for chunks in changed_by_source.values()
        for chunk in chunks
    ]

    em = embedding_model()
    for source, chunks in changed_by_source.items():
        state.by_source[source] = chunks

    all_chunks: list = []
    for docs in state.by_source.values():
        all_chunks.extend(docs)
    logger.info("index_urls: total chunks: %d", len(all_chunks))

    state.faiss_store = build_merged_faiss(all_chunks, em)
    _rebuild_chain(session_id, state, all_chunks)

    for source, chunks in changed_by_source.items():
        signature = chunks[0].metadata.get("signature")
        if signature:
            state.indexed_url_signatures[source] = signature
        if source not in state.indexed_files:
            state.indexed_files.append(source)

    logger.info("index_urls: complete, added/updated %d chunk(s)", len(changed_chunks))
    return len(changed_chunks)


def remove_file(session_id: str, filename: str) -> None:
    """Remove one file from a session's index and rebuild the retriever.

    Embeddings are disk-cached so rebuilding FAISS after removal is cheap —
    no VLM or embedding API calls are made for already-indexed content.
    """
    state = get_state(session_id)

    sources_to_remove = [s for s in state.by_source if Path(s).name == filename]
    if not sources_to_remove:
        return

    for src in sources_to_remove:
        del state.by_source[src]

    state.indexed_files = [f for f in state.indexed_files if f != filename]
    h = state.filename_to_hash.pop(filename, None)
    if h:
        state.indexed_file_hashes.discard(h)

    if not state.by_source:
        state.chain = None
        state.faiss_store = None
        return

    all_chunks = [c for docs in state.by_source.values() for c in docs]
    state.faiss_store = build_merged_faiss(all_chunks, embedding_model())
    _rebuild_chain(session_id, state, all_chunks)
    logger.info("remove_file: rebuilt index after removing %s (%d chunks remain)", filename, len(all_chunks))
