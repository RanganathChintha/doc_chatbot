"""Lazy singletons for expensive resources (embedding model, LLM).

Resolved on first use so importing this module doesn't trigger model loads or
network calls.
"""

from typing import Any

from embeddings.embedder import get_embedding_model
from rag.rag_chain import get_llm

_embedding_model: Any = None
_llm: Any = None


def embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = get_embedding_model()
    return _embedding_model


def llm():
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm
