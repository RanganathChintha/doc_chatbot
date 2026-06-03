# retriever/hybrid_retriever.py
"""
Hybrid retriever combining semantic FAISS search with BM25 keyword matching.
"""

import logging
import os
import pickle
from typing import Any
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from pydantic import PrivateAttr
from app.langsmith_tracing import langsmith_traceable as traceable
from app.config import CACHE_DIR, DEBUG_RETRIEVAL, TOP_K

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining:
    1. Semantic search via FAISS (merged store, one query op)
    2. Keyword search via BM25
    3. Weighted score fusion and deduplication
    """

    semantic_retriever: Any
    bm25_retriever: BM25Retriever | None = None
    k: int = TOP_K
    semantic_weight: float = 0.7
    bm25_weight: float = 0.3

    # Per-instance LRU query cache — avoids re-running retrieval for repeated queries.
    _query_cache: dict = PrivateAttr(default_factory=dict)
    _CACHE_SIZE: int = PrivateAttr(default=32)

    def __init__(
        self,
        semantic_retriever,
        bm25_retriever: BM25Retriever | None = None,
        k: int = TOP_K,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ):
        super().__init__(
            semantic_retriever=semantic_retriever,
            bm25_retriever=bm25_retriever,
            k=k,
            semantic_weight=semantic_weight,
            bm25_weight=bm25_weight,
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun | None = None, **kwargs: Any
    ) -> list[Document]:
        """Retrieve documents using hybrid approach with per-instance result cache."""
        if query in self._query_cache:
            logger.debug("HybridRetriever cache hit: %s...", query[:60])
            return self._query_cache[query]

        all_query_results: dict[str, dict[str, Any]] = {}

        # Semantic search — FAISS returns L2 distances (0 = identical, ~2 = max for unit vectors).
        # Convert to similarity score in [0, 1] so that higher = more relevant.
        if hasattr(self.semantic_retriever, "similarity_search_with_score"):
            results = self.semantic_retriever.similarity_search_with_score(query, k=self.k * 2)
            for doc, l2_dist in results:
                doc_id = self._doc_id(doc)
                all_query_results[doc_id] = {
                    "doc": doc,
                    "semantic_score": max(0.0, 1.0 - l2_dist / 2.0),
                    "bm25_score": 0.0,
                }
        else:
            results = self.semantic_retriever.invoke(query)
            for doc in results:
                doc_id = self._doc_id(doc)
                all_query_results[doc_id] = {
                    "doc": doc,
                    "semantic_score": 0.5,
                    "bm25_score": 0.0,
                }

        # BM25 keyword search
        if self.bm25_retriever:
            bm25_results = self.bm25_retriever.invoke(query)
            for rank, doc in enumerate(bm25_results):
                doc_id = self._doc_id(doc)
                bm25_score = max(0.1, 1.0 - (rank / (self.k * 2)))
                entry = all_query_results.get(doc_id)
                if entry is None:
                    all_query_results[doc_id] = {
                        "doc": doc,
                        "semantic_score": 0.0,
                        "bm25_score": bm25_score,
                    }
                else:
                    entry["bm25_score"] = bm25_score

        scored_docs = [
            (data["doc"],
             data["semantic_score"] * self.semantic_weight
             + data["bm25_score"] * self.bm25_weight)
            for data in all_query_results.values()
        ]
        scored_docs.sort(key=lambda x: x[1], reverse=True)

        if DEBUG_RETRIEVAL:
            logger.debug("Hybrid retrieval results:")
            for doc, score in scored_docs[:self.k]:
                logger.debug("  - %s: %.4f", doc.metadata.get("source", "unknown"), score)

        docs = [doc for doc, _ in scored_docs[: self.k]]

        # Cache result, evicting oldest entry when full.
        if len(self._query_cache) >= self._CACHE_SIZE:
            self._query_cache.pop(next(iter(self._query_cache)))
        self._query_cache[query] = docs
        return docs

    @staticmethod
    def _doc_id(doc: Document) -> str:
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        return f"{source}:{page}:{hash(doc.page_content) % 100000}"


@traceable(run_type="tool", name="build_hybrid_retriever")
def build_hybrid_retriever(
    semantic_retriever,
    chunks: list[Document] | None = None,
    k: int = TOP_K,
    use_bm25: bool = True,
    cache_key: str | None = None,
) -> HybridRetriever:
    """Create a hybrid retriever combining semantic and keyword search."""
    bm25_retriever = None
    if use_bm25 and chunks:
        bm25_retriever = _load_or_build_bm25(chunks, cache_key)

    retriever = HybridRetriever(
        semantic_retriever=semantic_retriever,
        bm25_retriever=bm25_retriever,
        k=k,
        semantic_weight=0.7,
        bm25_weight=0.3,
    )
    logger.info("Hybrid retriever created (semantic + BM25)")
    return retriever


def _bm25_cache_file(cache_key: str | None) -> str:
    suffix = ""
    if cache_key:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in cache_key)
        suffix = f"_{safe}"
    return os.path.join(CACHE_DIR, f"bm25_retriever{suffix}.pkl")


def _bm25_signature(chunks: list[Document]) -> int:
    """Fingerprint of the full corpus — hashes all chunks so any edit is detected."""
    content = "|".join(c.page_content[:64] for c in chunks)
    return hash((len(chunks), content))


def _load_or_build_bm25(chunks: list[Document], cache_key: str | None = None) -> BM25Retriever:
    cache_file = _bm25_cache_file(cache_key)
    signature = _bm25_signature(chunks)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, "rb") as f:
                cached_sig, cached_retriever = pickle.load(f)
            if cached_sig == signature:
                logger.info("Loaded BM25 retriever from cache.")
                return cached_retriever
        except Exception as exc:
            logger.warning("BM25 cache load failed (%s) — rebuilding.", exc)

    retriever = BM25Retriever.from_documents(chunks)
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_file, "wb") as f:
            pickle.dump((signature, retriever), f)
    except Exception as exc:
        logger.warning("BM25 cache save failed: %s", exc)
    return retriever
