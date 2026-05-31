# rag/rag_chain.py

import logging
import os
import re
import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from app.config import (
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_CACHE_FILE,
    CACHE_DIR,
    CHAT_HISTORY_WINDOW,
    TOP_K,
    RETRIEVAL_SCORE_THRESHOLD,
    DEBUG_RETRIEVAL,
)
from app.langsmith_tracing import langsmith_traceable as traceable

logger = logging.getLogger(__name__)

os.makedirs(CACHE_DIR, exist_ok=True)
set_llm_cache(SQLiteCache(database_path=LLM_CACHE_FILE))

_SESSION_TTL = 3600  # evict sessions idle for more than 1 hour
_session_histories: dict[str, InMemoryChatMessageHistory] = {}
_session_last_access: dict[str, float] = {}


def _evict_stale_sessions() -> None:
    now = time.time()
    stale = [sid for sid, t in _session_last_access.items() if now - t > _SESSION_TTL]
    for sid in stale:
        _session_histories.pop(sid, None)
        _session_last_access.pop(sid, None)
    if stale:
        logger.debug("Evicted %d stale session(s)", len(stale))


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Per-session window-buffer chat history with TTL eviction."""
    _evict_stale_sessions()
    history = _session_histories.get(session_id)
    if history is None:
        history = InMemoryChatMessageHistory()
        _session_histories[session_id] = history
    _session_last_access[session_id] = time.time()
    max_msgs = CHAT_HISTORY_WINDOW * 2
    if len(history.messages) > max_msgs:
        history.messages = history.messages[-max_msgs:]
    return history


@traceable(run_type="tool", name="get_llm")
def get_llm() -> ChatGroq:
    if not GROQ_API_KEY:
        raise EnvironmentError(
            "GROQ_API_KEY is not set. Add it to your .env file or environment."
        )
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.2,
    )
    logger.info("LLM loaded: %s via ChatGroq (SQLite response cache active)", LLM_MODEL)
    return llm


def _format_tagged_context(docs: list[Document]) -> str:
    """Tag each chunk with [FILE: filename | TYPE] so the LLM can answer per file."""
    lines = []
    for d in docs:
        source = d.metadata.get("source", "?")
        source_name = os.path.basename(source) if source else "?"
        kind = "TEXT" if d.metadata.get("source_type") == "text" else "IMAGE"
        page = d.metadata.get("page")
        page_str = f" p.{page}" if page is not None else ""
        lines.append(f"[FILE: {source_name}{page_str} | {kind}] {d.page_content}")
    return "\n\n".join(lines)


@traceable(run_type="chain", name="build_rag_chain")
def build_rag_chain(
    retriever,
    source_names: list[str],
    llm: ChatGroq,
):
    """
    Conversational RAG chain with global top-K retrieval.

    Args:
        retriever: global FAISS store over all chunks
        source_names: all source file paths in the indexed corpus
        llm: ChatGroq instance

    Behavior:
      - For normal queries: pulls the top-K most related chunks overall.
      - History-aware: rewrites follow-ups into standalone questions before retrieval.
    """
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, rewrite the "
         "question as a standalone question that can be understood without "
         "the chat history. Do NOT answer it; only reformulate it. If the "
         "question is already standalone, return it unchanged."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    contextualize_question = contextualize_prompt | llm | StrOutputParser()

    # Cheap heuristic: only pay for the extra LLM rewrite call when the query
    # actually depends on prior turns. Saves ~1 round-trip per standalone Q.
    _REFERENTIAL = re.compile(
        r"\b(it|its|that|this|these|those|they|them|their|he|she|his|her|"
        r"above|previous|earlier|same|former|latter)\b",
        re.IGNORECASE,
    )

    def _needs_contextualization(query: str) -> bool:
        q = query.strip()
        if not q:
            return False
        # Very short queries are almost always follow-ups ("why?", "and then?").
        if len(q.split()) <= 3:
            return True
        return bool(_REFERENTIAL.search(q))

    def _find_source_hint(query: str) -> str | None:
        normalized = query.lower()
        for source in source_names:
            source_name = os.path.basename(source).lower()
            if source_name and source_name in normalized:
                return source
        return None

    def _get_docs(inputs: dict) -> list[Document]:
        """Enhanced retrieval with scoring, filtering, and re-ranking."""
        raw_query = inputs.get("input", "")

        # Contextualize only when the query actually references prior turns —
        # avoids an extra LLM round-trip on every standalone question.
        if inputs.get("chat_history") and _needs_contextualization(raw_query):
            query = contextualize_question.invoke(inputs)
        else:
            query = raw_query

        # Retrieve with scoring
        if hasattr(retriever, "similarity_search_with_score"):
            results = retriever.similarity_search_with_score(query, k=TOP_K * 2)
            
            # Apply score threshold for quality filtering
            threshold_filtered = [
                (doc, score) for doc, score in results 
                if score >= RETRIEVAL_SCORE_THRESHOLD
            ]
            
            # If threshold filtering removes too many, fall back to top-k
            if len(threshold_filtered) < max(3, TOP_K // 2):
                filtered = results[:TOP_K]
            else:
                filtered = threshold_filtered
            
            # Apply source hint prioritization
            source_hint = _find_source_hint(query)
            if source_hint:
                def sort_key(pair):
                    doc, score = pair
                    is_source_match = doc.metadata.get("source") == source_hint
                    return (is_source_match, score)
                filtered.sort(key=sort_key, reverse=True)
            
            docs = [doc for doc, _ in filtered[:TOP_K]]
            
            if DEBUG_RETRIEVAL:
                logger.debug("Retrieval debug info:")
                logger.debug("   Query: %s...", query[:100])
                logger.debug("   Total candidates: %d", len(results))
                logger.debug("   After threshold: %d", len(threshold_filtered))
                logger.debug("   Final selection: %d", len(docs))
                for i, (doc, score) in enumerate(results[:TOP_K], 1):
                    source = os.path.basename(doc.metadata.get("source", "?"))
                    page = doc.metadata.get("page", "?")
                    logger.debug("   [%d] %s (p.%s): %.4f", i, source, page, score)
            
            return docs

        # Fallback for non-scored retrievers (HybridRetriever)
        return retriever.invoke(query)

    qa_system_prompt = (
        "Answer using only the context. Do not hallucinate. "
        "If unavailable, say so. Cite sources. Be concise."
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("system", "Use the following retrieved context to answer the question. If the answer is not contained in the context, say that you do not know.") ,
        ("system", "CONTEXT:\n{context}"),
        ("human", "{question}"),
    ])

    rag_chain = (
        RunnablePassthrough.assign(
            docs=RunnableLambda(_get_docs),
        )
        | RunnablePassthrough.assign(
            context=lambda x: _format_tagged_context(x["docs"]),
        )
        | RunnablePassthrough.assign(
            question=lambda x: x["input"],
        )
        | RunnablePassthrough.assign(
            answer=qa_prompt | llm | StrOutputParser(),
        )
    )

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return conversational_rag_chain
