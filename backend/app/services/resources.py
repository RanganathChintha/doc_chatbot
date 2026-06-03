"""Lazy singletons for expensive resources (embedding model, LLM).

Resolved on first use so importing this module doesn't trigger model loads or
network calls.
"""

import logging
from typing import Any

from app.embeddings.embedder import get_embedding_model
from app.rag.rag_chain import get_llm

logger = logging.getLogger(__name__)

_embedding_model: Any = None
_llm: Any = None


def embedding_model():
    global _embedding_model
    if _embedding_model is None:
        logger.info("resources.embedding_model: initializing embedding model...")
        _embedding_model = get_embedding_model()
        logger.info("resources.embedding_model: embedding model initialized")
    return _embedding_model


def llm():
    global _llm
    if _llm is None:
        logger.info("resources.llm: initializing LLM...")
        _llm = get_llm()
        logger.info("resources.llm: LLM initialized")
    return _llm
