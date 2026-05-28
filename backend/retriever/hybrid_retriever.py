# retriever/hybrid_retriever.py
"""
Enhanced hybrid retriever combining semantic search with keyword (BM25) matching,
query expansion, and re-ranking for better relevance.
"""

import logging
import os
import pickle
from typing import Any
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langsmith_tracing import langsmith_traceable as traceable
from config import CACHE_DIR, DEBUG_RETRIEVAL, TOP_K

logger = logging.getLogger(__name__)


class HybridRetriever(BaseRetriever):
    """
    Hybrid retriever combining:
    1. Semantic search via FAISS vector store
    2. Keyword search via BM25
    3. Query expansion for broader matching
    4. Result re-ranking for final ordering
    """

    semantic_retriever: Any
    bm25_retriever: BM25Retriever | None = None
    k: int = TOP_K
    semantic_weight: float = 0.7
    bm25_weight: float = 0.3

    def __init__(
        self,
        semantic_retriever,
        bm25_retriever: BM25Retriever | None = None,
        k: int = TOP_K,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
    ):
        """
        Args:
            semantic_retriever: FAISS or similar vector store retriever
            bm25_retriever: Optional BM25 retriever for keyword matching
            k: Number of results to return
            semantic_weight: Weight for semantic search results (0-1)
            bm25_weight: Weight for BM25 results (0-1)
        """
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
        """Retrieve documents using hybrid approach (one semantic + one BM25 pass)."""
        all_query_results: dict[str, dict[str, Any]] = {}

        # Semantic search
        if hasattr(self.semantic_retriever, "similarity_search_with_score"):
            results = self.semantic_retriever.similarity_search_with_score(query, k=self.k * 2)
            for doc, score in results:
                doc_id = self._doc_id(doc)
                all_query_results[doc_id] = {
                    "doc": doc,
                    "semantic_score": min(1.0, score),
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
                entry = all_query_results.get(doc_id)
                bm25_score = max(0.1, 1.0 - (rank / (self.k * 2)))
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
                source = doc.metadata.get("source", "unknown")
                logger.debug("  - %s: %.4f", source, score)

        return [doc for doc, _ in scored_docs[: self.k]]

    @staticmethod
    def _doc_id(doc: Document) -> str:
        """Generate unique ID for deduplication."""
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        return f"{source}:{page}:{hash(doc.page_content) % 100000}"


@traceable(run_type="tool", name="build_hybrid_retriever")
def build_hybrid_retriever(
    semantic_retriever,
    chunks: list[Document] | None = None,
    k: int = TOP_K,
    use_bm25: bool = True,
) -> HybridRetriever:
    """
    Create a hybrid retriever combining semantic and keyword search.

    Args:
        semantic_retriever: FAISS or similar vector store
        chunks: Document chunks for BM25 indexing (optional)
        k: Number of results to return
        use_bm25: Whether to use BM25 in hybrid search

    Returns:
        HybridRetriever instance
    """
    bm25_retriever = None
    if use_bm25 and chunks:
        bm25_retriever = _load_or_build_bm25(chunks)

    retriever = HybridRetriever(
        semantic_retriever=semantic_retriever,
        bm25_retriever=bm25_retriever,
        k=k,
        semantic_weight=0.7,
        bm25_weight=0.3,
    )
    logger.info("Hybrid retriever created (semantic + BM25 keyword search)")
    return retriever


_BM25_CACHE_FILE = os.path.join(CACHE_DIR, "bm25_retriever.pkl")


def _bm25_signature(chunks: list[Document]) -> int:
    """Cheap fingerprint of the corpus — if chunks change, the BM25 cache misses."""
    # Hash a sample so we don't walk huge corpora on every startup. Content +
    # length is enough to detect re-chunking or new sources.
    sample = "|".join(c.page_content[:64] for c in chunks[:256])
    return hash((len(chunks), sample))


def _load_or_build_bm25(chunks: list[Document]) -> BM25Retriever:
    signature = _bm25_signature(chunks)
    if os.path.exists(_BM25_CACHE_FILE):
        try:
            with open(_BM25_CACHE_FILE, "rb") as f:
                cached_sig, cached_retriever = pickle.load(f)
            if cached_sig == signature:
                logger.info("Loaded BM25 retriever from cache.")
                return cached_retriever
        except Exception as exc:
            logger.warning("BM25 cache load failed (%s) — rebuilding.", exc)

    retriever = BM25Retriever.from_documents(chunks)
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(_BM25_CACHE_FILE, "wb") as f:
            pickle.dump((signature, retriever), f)
    except Exception as exc:
        logger.warning("BM25 cache save failed: %s", exc)
    return retriever
