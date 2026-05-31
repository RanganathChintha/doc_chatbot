# retriever/faiss_retriever.py
#
# Per-source FAISS retrievers plus a MultiRetriever facade for the hybrid layer.

import logging
from collections import defaultdict
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.config import TOP_K
from app.langsmith_tracing import langsmith_traceable as traceable

logger = logging.getLogger(__name__)


@traceable(run_type="tool", name="build_per_source_retrievers")
def build_per_source_retrievers(
    chunks: list[Document],
    embedding_model,
    k_per_source: int = 3,
) -> tuple[dict, dict]:
    """Build one FAISS retriever per source file.

    Returns:
        retrievers: {source_path -> Retriever} — per-file similarity search.
        by_source:  {source_path -> list[Document]} — chunks grouped by file,
                    used to rebuild BM25 / answer per-file context queries.
    """
    by_source: dict[str, list[Document]] = defaultdict(list)
    for c in chunks:
        by_source[c.metadata.get("source", "unknown")].append(c)

    retrievers: dict = {}
    for source, source_chunks in by_source.items():
        k = min(k_per_source, len(source_chunks))
        if k == 0:
            continue
        store = FAISS.from_documents(source_chunks, embedding_model)
        retrievers[source] = store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    logger.info("Per-file retrievers ready: %d file(s), top-%d per file.", len(retrievers), k_per_source)
    return retrievers, dict(by_source)


class MultiRetriever:
    """Facade that fans `similarity_search_with_score` out to N per-source retrievers."""

    def __init__(self, retrievers: dict[str, Any] | None = None):
        self.retrievers: dict[str, Any] = retrievers or {}

    def add_retrievers(self, new: dict[str, Any]) -> None:
        self.retrievers.update(new)

    def similarity_search_with_score(self, query: str, k: int | None = None):
        results: list[tuple[Document, float]] = []
        per_k = k or TOP_K
        for r in self.retrievers.values():
            try:
                results.extend(r.similarity_search_with_score(query, k=per_k))
            except Exception:
                try:
                    results.extend((d, 0.5) for d in r.invoke(query))
                except Exception:
                    continue

        seen: set[tuple] = set()
        unique: list[tuple[Document, float]] = []
        for doc, score in sorted(results, key=lambda x: x[1], reverse=True):
            key = (
                doc.metadata.get("source", ""),
                doc.metadata.get("page", ""),
                hash(doc.page_content) % 100000,
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append((doc, score))

        return unique[: per_k * 2]
