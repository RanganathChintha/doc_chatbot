"""Per-session index state + incremental document ingestion.

Each chat session owns an isolated `IndexState` (its own FAISS / hybrid
retriever / RAG chain) so one chat's uploaded files are never retrievable by
another. All mutation goes through `index_files` / `clear_session`, keyed by
`session_id`.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

from config import TOP_K
from langsmith_tracing import langsmith_traceable as traceable
from pipeline import load_and_chunk
from rag.rag_chain import build_rag_chain
from retriever.faiss_retriever import MultiRetriever, build_per_source_retrievers
from retriever.hybrid_retriever import build_hybrid_retriever

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
_lock = threading.Lock()


def get_state(session_id: str) -> IndexState:
    """Return the session's IndexState, creating an empty one on first use."""
    with _lock:
        st = _sessions.get(session_id)
        if st is None:
            st = IndexState()
            _sessions[session_id] = st
        return st


def clear_session(session_id: str) -> None:
    """Drop a session's index entirely (used on chat delete / file clear)."""
    with _lock:
        _sessions.pop(session_id, None)


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
        k=10,
        use_bm25=True,
        cache_key=session_id,
    )
    source_names = sorted({
        d.metadata.get("source", "")
        for d in all_chunks
        if d.metadata.get("source")
    })
    state.chain = build_rag_chain(hybrid_retriever, source_names, llm())
    state.semantic_retriever = semantic_multi


@traceable(run_type="chain", name="ingest_pipeline")
def index_files(session_id: str, paths: list[Path]) -> int:
    """Incrementally index newly-uploaded files into a session's own index.

    Returns the number of new chunks added.
    """
    state = get_state(session_id)
    new_paths = [p for p in paths if p.name not in state.indexed_files]
    if not new_paths:
        return 0

    new_chunks = load_and_chunk([str(p) for p in new_paths], verbose=False)
    if not new_chunks:
        return 0

    new_retrievers, new_by_source = build_per_source_retrievers(
        new_chunks, embedding_model(), k_per_source=TOP_K,
    )

    if state.semantic_retriever is None:
        semantic_multi = MultiRetriever(new_retrievers)
    else:
        semantic_multi = state.semantic_retriever
        semantic_multi.add_retrievers(new_retrievers)

    state.by_source.update(new_by_source)
    all_chunks: list = []
    for docs in state.by_source.values():
        all_chunks.extend(docs)

    _rebuild_chain(session_id, state, all_chunks, semantic_multi)
    state.indexed_files.extend(p.name for p in new_paths)
    return len(new_chunks)
