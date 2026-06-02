# rag/rag_chain.py

import logging
import os
import re
import time
import certifi
import httpx
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

# Compiled once at import time, not per chain build.
_PAGE_RE = re.compile(r'\b(?:page|pg|p\.?)\s*(\d+)\b', re.IGNORECASE)

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
    
    # Create a custom httpx client with proper SSL certificate verification
    # This fixes SSL certificate verification errors on Windows
    client = httpx.Client(
        verify=certifi.where(),
        timeout=30.0,
    )
    
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.2,
        max_retries=5,
        http_client=client,
    )
    logger.info("LLM loaded: %s via ChatGroq (SQLite response cache active)", LLM_MODEL)
    return llm


def _format_tagged_context(docs: list[Document]) -> str:
    """Tag each chunk with [FILE: filename | page N | TYPE] so the LLM can cite pages."""
    lines = []
    for d in docs:
        source = d.metadata.get("source", "?")
        source_name = os.path.basename(source) if source else "?"
        kind = "TEXT" if d.metadata.get("source_type") == "text" else "IMAGE"
        page_label = d.metadata.get("page_label")
        page_raw = d.metadata.get("page")
        if page_raw is not None:
            page_str = f" | page {page_raw + 1}"
            if page_label is not None:
                page_str += f" (label {page_label})"
        elif page_label is not None:
            page_str = f" | page {page_label}"
        else:
            page_str = ""
        lines.append(f"[FILE: {source_name}{page_str} | {kind}] {d.page_content}")
    return "\n\n".join(lines)


@traceable(run_type="chain", name="build_rag_chain")
def build_rag_chain(
    retriever,
    source_names: list[str],
    llm: ChatGroq,
    all_chunks: list[Document] | None = None,
):
    """
    Conversational RAG chain with global top-K retrieval.

    Args:
        retriever: global FAISS store over all chunks
        source_names: all source file paths in the indexed corpus
        llm: ChatGroq instance
        all_chunks: flat list of every indexed chunk — used for direct page lookup
            so "what's on page 30?" bypasses vector search and hits the right chunks.

    Behavior:
      - For page-specific queries: returns all chunks from that page directly.
      - For normal queries: pulls the top-K most related chunks overall.
      - History-aware: rewrites follow-ups into standalone questions before retrieval.
    """
    def _page_numbers_for_chunk(chunk: Document) -> list[str]:
        labels: list[str] = []
        page_raw = chunk.metadata.get("page")
        if page_raw is not None and page_raw != "":
            try:
                raw_num = int(page_raw)
                labels.append(str(raw_num + 1))
                labels.append(str(raw_num))
            except (TypeError, ValueError):
                labels.append(str(page_raw).strip())
        return [label for label in dict.fromkeys(labels) if label]

    # Build page index: (source_basename, raw_page_number) → [chunks]
    _page_index: dict[tuple[str, str], list[Document]] = {}
    _page_label_index: dict[str, list[Document]] = {}
    for chunk in (all_chunks or []):
        labels = _page_numbers_for_chunk(chunk)
        source = os.path.basename(chunk.metadata.get("source", ""))
        for label in labels:
            if source:
                _page_index.setdefault((source, label), []).append(chunk)
            _page_label_index.setdefault(label, []).append(chunk)
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

    def _find_page_hint(query: str) -> str | None:
        """Return the page label string if the query mentions a specific page number."""
        m = _PAGE_RE.search(query)
        return m.group(1) if m else None

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

        # --- Direct page lookup (bypasses vector search entirely) ---
        page_label = _find_page_hint(query)
        if page_label and _page_index:
            source_hint = _find_source_hint(query)
            page_chunks: list[Document] = []
            if source_hint:
                source_base = os.path.basename(source_hint)
                page_chunks = _page_index.get((source_base, page_label), [])
            if not page_chunks:
                # No source match — fall back to all chunks with that label.
                page_chunks = _page_label_index.get(page_label, [])
            if page_chunks:
                logger.debug("Page lookup: label=%s → %d chunk(s)", page_label, len(page_chunks))
                return page_chunks

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
        "Answer using only the context provided. Do not hallucinate. "
        "Each context chunk is tagged with [FILE: filename | page N | TYPE]. "
        "When the user asks about a specific page, use the chunks from that page. "
        "If the requested page has no content in the context, say so explicitly. "
        "Cite the file and page number in your answer. Be concise."
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
