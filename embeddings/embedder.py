from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings
from config import EMBEDDING_MODEL

def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Initialize and return the embedding model locally
    via sentence-transformers (no API key required).
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    print(f"✅ Embedding model loaded locally: {EMBEDDING_MODEL}")
    return embeddings