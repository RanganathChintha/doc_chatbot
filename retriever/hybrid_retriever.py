# retriever/hybrid_retriever.py
"""
Enhanced hybrid retriever combining semantic search with keyword (BM25) matching,
query expansion, and re-ranking for better relevance.
"""

from typing import Any
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from langsmith_tracing import langsmith_traceable as traceable
from config import DEBUG_RETRIEVAL, TOP_K


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
        """Retrieve documents using hybrid approach."""
        expanded_queries = self._expand_query(query)
        all_query_results = {}

        # Semantic search with all query variations
        for q in expanded_queries:
            if hasattr(self.semantic_retriever, "similarity_search_with_score"):
                results = self.semantic_retriever.similarity_search_with_score(q, k=self.k * 2)
                for doc, score in results:
                    doc_id = self._doc_id(doc)
                    if doc_id not in all_query_results:
                        all_query_results[doc_id] = {
                            "doc": doc,
                            "semantic_score": 0.0,
                            "bm25_score": 0.0,
                            "count": 0,
                        }
                    # Normalize score to 0-1 range (FAISS uses similarity)
                    all_query_results[doc_id]["semantic_score"] = max(
                        all_query_results[doc_id]["semantic_score"],
                        min(1.0, score),
                    )
                    all_query_results[doc_id]["count"] += 1
            else:
                # Fallback for non-scored retrievers
                results = self.semantic_retriever.invoke(q)
                for doc in results:
                    doc_id = self._doc_id(doc)
                    if doc_id not in all_query_results:
                        all_query_results[doc_id] = {
                            "doc": doc,
                            "semantic_score": 0.5,
                            "bm25_score": 0.0,
                            "count": 0,
                        }

        # BM25 keyword search if available
        if self.bm25_retriever:
            for q in expanded_queries:
                bm25_results = self.bm25_retriever.invoke(q)
                for rank, doc in enumerate(bm25_results):
                    doc_id = self._doc_id(doc)
                    if doc_id not in all_query_results:
                        all_query_results[doc_id] = {
                            "doc": doc,
                            "semantic_score": 0.0,
                            "bm25_score": 0.0,
                            "count": 0,
                        }
                    # BM25 scoring: higher rank = higher score
                    bm25_score = max(0.1, 1.0 - (rank / (self.k * 2)))
                    all_query_results[doc_id]["bm25_score"] = max(
                        all_query_results[doc_id]["bm25_score"],
                        bm25_score,
                    )

        # Calculate hybrid scores and sort
        scored_docs = []
        for doc_id, data in all_query_results.items():
            hybrid_score = (
                data["semantic_score"] * self.semantic_weight
                + data["bm25_score"] * self.bm25_weight
            )
            # Boost score if query matched in multiple variations
            hybrid_score *= (1.0 + data["count"] * 0.1)
            scored_docs.append((data["doc"], hybrid_score))

        scored_docs.sort(key=lambda x: x[1], reverse=True)

        if DEBUG_RETRIEVAL:
            print("Hybrid retrieval results:")
            for doc, score in scored_docs[:self.k]:
                source = doc.metadata.get("source", "unknown")
                print(f"  - {source}: {score:.4f}")

        return [doc for doc, _ in scored_docs[: self.k]]

    @staticmethod
    def _doc_id(doc: Document) -> str:
        """Generate unique ID for deduplication."""
        source = doc.metadata.get("source", "")
        page = doc.metadata.get("page", "")
        return f"{source}:{page}:{hash(doc.page_content) % 100000}"

    @staticmethod
    def _expand_query(query: str) -> list[str]:
        """
        Expand the query to capture more relevant documents.
        Uses simple techniques:
        1. Original query
        2. Query with synonyms/related terms (placeholder)
        3. Query parts (if compound query)
        """
        queries = [query]

        # Add variations of the original query
        if len(query) > 20:
            # For longer queries, try key phrase extraction (simplified)
            words = query.split()
            if len(words) > 5:
                # Try first half
                queries.append(" ".join(words[: len(words) // 2]))
                # Try second half
                queries.append(" ".join(words[len(words) // 2 :]))

        return list(set(queries))  # Remove duplicates


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
        bm25_retriever = BM25Retriever.from_documents(chunks)

    retriever = HybridRetriever(
        semantic_retriever=semantic_retriever,
        bm25_retriever=bm25_retriever,
        k=k,
        semantic_weight=0.7,
        bm25_weight=0.3,
    )
    print("Hybrid retriever created (semantic + BM25 keyword search)")
    return retriever
