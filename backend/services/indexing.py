"""Index state + incremental document ingestion.

Owns the FAISS / hybrid retriever / RAG chain for the running process. All
mutation goes through `index_files` and `clear`.
"""

from __future__ import annotations

import logging
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
    """Mutable, process-global view of what's currently indexed."""

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


state = IndexState()


@traceable(run_type="chain", name="rebuild_rag_chain")
def _rebuild_chain(all_chunks: list, semantic_multi: MultiRetriever) -> None:
    hybrid_retriever = build_hybrid_retriever(
        semantic_retriever=semantic_multi,
        chunks=all_chunks,
        k=10,
        use_bm25=True,
    )
    source_names = sorted({
        d.metadata.get("source", "")
        for d in all_chunks
        if d.metadata.get("source")
    })
    state.chain = build_rag_chain(hybrid_retriever, source_names, llm())
    state.semantic_retriever = semantic_multi


@traceable(run_type="chain", name="ingest_pipeline")
def index_files(paths: list[Path]) -> int:
    """Incrementally index newly-uploaded files. Returns the number of new chunks."""
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

    _rebuild_chain(all_chunks, semantic_multi)
    state.indexed_files.extend(p.name for p in new_paths)
    return len(new_chunks)
