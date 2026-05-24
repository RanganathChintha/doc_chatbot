from __future__ import annotations

import os

# Note: HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE must be set in main.py BEFORE
# any HF library is imported, not here — by the time this module runs, those
# libraries have already cached their config from os.environ.

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.embeddings import CacheBackedEmbeddings
from langchain_classic.storage import LocalFileStore
from config import EMBEDDING_MODEL, EMBEDDING_CACHE_DIR


def get_embedding_model() -> CacheBackedEmbeddings:
    """
    Local sentence-transformers embedding model wrapped in a disk-backed
    cache. Identical chunk texts skip re-embedding on subsequent runs.

    Runs fully offline once the model is downloaded to the HF hub cache.
    To download (or update) the model, temporarily unset HF_HUB_OFFLINE.
    """
    os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
    underlying = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    store = LocalFileStore(EMBEDDING_CACHE_DIR)
    cached = CacheBackedEmbeddings.from_bytes_store(
        underlying,
        store,
        namespace=EMBEDDING_MODEL.replace("/", "_"),
    )
    print(f"✅ Embedding model loaded locally (cached, offline): {EMBEDDING_MODEL}")
    return cached
