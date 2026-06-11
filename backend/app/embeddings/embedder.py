from __future__ import annotations

import logging
import os

# Note: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE must be set in main.py BEFORE
# any HF library is imported, not here — by the time this module runs, those
# libraries have already cached their config from os.environ.

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from app.config import EMBEDDING_MODEL, EMBEDDING_CACHE_DIR
from app.langsmith_tracing import langsmith_traceable as traceable

logger = logging.getLogger(__name__)


@traceable(run_type="tool", name="get_embedding_model")
def get_embedding_model() -> CacheBackedEmbeddings:
    """
    Local sentence-transformers embedding model wrapped in a disk-backed
    cache. Identical chunk texts skip re-embedding on subsequent runs.

    Runs fully offline once the model is downloaded to the HF hub cache.
    To download (or update) the model, temporarily unset HF_HUB_OFFLINE.
    """
    os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
    # normalize_embeddings=True is required: FAISS ranks by L2 distance, and the
    # hybrid retriever's L2→similarity conversion (1 - l2/2) is only valid for
    # unit vectors. Without normalization, ranking is wrong and most semantic
    # scores clamp to 0, collapsing retrieval onto BM25 alone.
    underlying = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
    store = LocalFileStore(EMBEDDING_CACHE_DIR)
    cached = CacheBackedEmbeddings.from_bytes_store(
        underlying,
        store,
        # "_norm" suffix busts any cached un-normalized vectors from before this fix.
        namespace=EMBEDDING_MODEL.replace("/", "_") + "_norm",
    )
    logger.info("Embedding model loaded locally (cached, offline): %s", EMBEDDING_MODEL)
    return cached
