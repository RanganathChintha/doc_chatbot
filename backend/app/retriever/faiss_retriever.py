# retriever/faiss_retriever.py
#
# Single merged FAISS store for the entire session corpus.
# Embedding all chunks in one batch is faster than building N per-source stores.

import logging
from collections import defaultdict

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.langsmith_tracing import langsmith_traceable as traceable

logger = logging.getLogger(__name__)


def group_chunks_by_source(chunks: list[Document]) -> dict[str, list[Document]]:
    """Group chunks by their 'source' metadata key (no FAISS building)."""
    by_source: dict[str, list[Document]] = defaultdict(list)
    for c in chunks:
        by_source[c.metadata.get("source", "unknown")].append(c)
    return dict(by_source)


@traceable(run_type="tool", name="build_merged_faiss")
def build_merged_faiss(chunks: list[Document], embedding_model) -> FAISS:
    """Embed all chunks in one batch and return a single FAISS store.

    One embedding call over the full corpus is far cheaper than calling
    embed_documents once per source file as the old per-source approach did.
    """
    logger.info("build_merged_faiss: embedding %d chunk(s) in one batch", len(chunks))
    store = FAISS.from_documents(chunks, embedding_model)
    logger.info("build_merged_faiss: complete")
    return store
