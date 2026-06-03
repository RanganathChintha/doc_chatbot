"""Per-session index state + incremental document ingestion.

Each chat session owns an isolated `IndexState` (its own FAISS / hybrid
retriever / RAG chain) so one chat's uploaded files are never retrievable by
another. All mutation goes through `index_files` / `clear_session`, keyed by
`session_id`.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Any

from app.config import TOP_K
from app.langsmith_tracing import langsmith_traceable as traceable
from app.pipeline import load_and_chunk
from app.rag.rag_chain import build_rag_chain
from app.retriever.faiss_retriever import MultiRetriever, build_per_source_retrievers
from app.retriever.hybrid_retriever import build_hybrid_retriever

from .resources import embedding_model, llm

logger = logging.getLogger(__name__)


class IndexState:
    """Mutable view of what a single session currently has indexed."""

    def __init__(self) -> None:
        self.chain: Any = None
        self.semantic_retriever: MultiRetriever | None = None
        self.by_source: dict[str, list] = {}
        self.indexed_files: list[str] = []

    def clear(self) -> None:
        self.chain = None
        self.semantic_retriever = None
        self.by_source = {}
        self.indexed_files = []


# Registry of per-session index states. Guarded by a lock because uploads run
# in a worker thread (`asyncio.to_thread`) while requests may touch the registry
# concurrently.
_sessions: dict[str, IndexState] = {}
_session_last_access: dict[str, float] = {}
_lock = threading.Lock()

# Evict index states idle for more than 2 hours — mirrors the chat-history TTL
# and prevents unbounded RAM growth as users create chats and upload files.
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
        _evict_stale_sessions()
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
def _rebuild_chain(
    session_id: str,
    state: IndexState,
    all_chunks: list,
    semantic_multi: MultiRetriever,
) -> None:
    hybrid_retriever = build_hybrid_retriever(
        semantic_retriever=semantic_multi,
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
    state.semantic_retriever = semantic_multi


@traceable(run_type="chain", name="ingest_pipeline")
def index_files(session_id: str, paths: list[Path]) -> int:
    """Incrementally index newly-uploaded files into a session's own index.

    Returns the number of new chunks added.
    """
    logger.info("index_files: session=%s, %d path(s)", session_id, len(paths))
    state = get_state(session_id)
    new_paths = [p for p in paths if p.name not in state.indexed_files]
    if not new_paths:
        logger.info("index_files: no new paths to index")
        return 0
    logger.info("index_files: indexing %d new file(s)", len(new_paths))

    logger.debug("index_files: calling load_and_chunk...")
    new_chunks = load_and_chunk([str(p) for p in new_paths], verbose=False)
    logger.info("index_files: load_and_chunk returned %d chunk(s)", len(new_chunks))
    if not new_chunks:
        logger.info("index_files: no chunks returned")
        return 0

    logger.debug("index_files: building retrievers with embedding_model...")
    new_retrievers, new_by_source = build_per_source_retrievers(
        new_chunks, embedding_model(), k_per_source=TOP_K,
    )
    logger.info("index_files: built %d retriever(s), %d source(s)", len(new_retrievers), len(new_by_source))

    if state.semantic_retriever is None:
        logger.debug("index_files: creating new MultiRetriever")
        semantic_multi = MultiRetriever(new_retrievers)
    else:
        logger.debug("index_files: adding retrievers to existing MultiRetriever")
        semantic_multi = state.semantic_retriever
        semantic_multi.add_retrievers(new_retrievers)
    logger.debug("index_files: MultiRetriever updated")

    state.by_source.update(new_by_source)
    all_chunks: list = []
    for docs in state.by_source.values():
        all_chunks.extend(docs)
    logger.info("index_files: total chunks now: %d", len(all_chunks))

    logger.debug("index_files: building RAG chain...")
    _rebuild_chain(session_id, state, all_chunks, semantic_multi)
    logger.info("index_files: RAG chain built")
    
    state.indexed_files.extend(p.name for p in new_paths)
    logger.info("index_files: complete, added %d chunk(s)", len(new_chunks))
    return len(new_chunks)
