# rag/rag_chain.py

import logging
import os
import re
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from app.config import (
    SIEMENS_API_KEY,
    SIEMENS_API_BASE,
    LLM_MODEL,
    LLM_CACHE_FILE,
    CACHE_DIR,
    CHAT_HISTORY_WINDOW,
)
from app.langsmith_tracing import langsmith_traceable as traceable

logger = logging.getLogger(__name__)

# Compiled once at import time, not per chain build.
_PAGE_RE = re.compile(r'\b(?:page|pg|p\.?)\s*(\d+)\b', re.IGNORECASE)
_REFERENTIAL = re.compile(
    r"\b(it|its|that|this|these|those|they|them|their|he|she|his|her|"
    r"above|previous|earlier|same|former|latter)\b",
    re.IGNORECASE,
)

os.makedirs(CACHE_DIR, exist_ok=True)
set_llm_cache(SQLiteCache(database_path=LLM_CACHE_FILE))

_SESSION_TTL = 3600  # evict sessions idle for more than 1 hour
_session_histories: dict[str, "InMemoryChatMessageHistory"] = {}
_session_last_access: dict[str, float] = {}

_MAX_HISTORY_MSGS = CHAT_HISTORY_WINDOW * 2  # user+assistant pairs


class _BoundedHistory(InMemoryChatMessageHistory):
    """Chat history that trims on add rather than on every read access."""
    max_messages: int = _MAX_HISTORY_MSGS

    def add_message(self, message) -> None:  # type: ignore[override]
        super().add_message(message)
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]


def _evict_stale_sessions() -> None:
    now = time.time()
    stale = [sid for sid, t in _session_last_access.items() if now - t > _SESSION_TTL]
    for sid in stale:
        _session_histories.pop(sid, None)
        _session_last_access.pop(sid, None)
    if stale:
        logger.debug("Evicted %d stale session(s)", len(stale))


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Per-session bounded chat history with TTL eviction (eviction runs in background)."""
    history = _session_histories.get(session_id)
    if history is None:
        history = _BoundedHistory()
        _session_histories[session_id] = history
    _session_last_access[session_id] = time.time()
    return history


@traceable(run_type="tool", name="get_llm")
def get_llm() -> ChatOpenAI:
    if not SIEMENS_API_KEY:
        raise EnvironmentError(
            "Siemens API key not set. Ensure OPENAI_API_KEY is in your .env or environment."
        )
    llm = ChatOpenAI(
        model_name=LLM_MODEL,
        temperature=0.2,
        max_retries=5,
        openai_api_key=SIEMENS_API_KEY,
        openai_api_base=SIEMENS_API_BASE,
    )
    logger.info("LLM loaded: %s via ChatOpenAI (SQLite response cache active)", LLM_MODEL)
    return llm


def _format_tagged_context(docs: list[Document]) -> str:
    """Tag each chunk so the LLM can cite its origin.

    Wiki/web pages are tagged [WIKI: title | url] so the model can cite a real
    page name and link; files are tagged [FILE: filename | page N | TYPE].
    """
    lines = []
    for d in docs:
        meta = d.metadata
        # Only true images are IMAGE; text, wiki, web, and tabular chunks are TEXT.
        kind = "IMAGE" if meta.get("source_type") == "image" else "TEXT"
        title = (meta.get("title") or "").strip()
        if title:
            url = meta.get("remote_url") or meta.get("source") or ""
            tag = f"[WIKI: {title}{' | ' + url if url else ''} | {kind}]"
        else:
            source = meta.get("source", "?")
            source_name = os.path.basename(source) if source else "?"
            page_label = meta.get("page_label")
            page_raw = meta.get("page")
            if page_raw is not None:
                page_str = f" | page {page_raw + 1}"
                if page_label is not None:
                    page_str += f" (label {page_label})"
            elif page_label is not None:
                page_str = f" | page {page_label}"
            else:
                page_str = ""
            tag = f"[FILE: {source_name}{page_str} | {kind}]"
        lines.append(f"{tag} {d.page_content}")
    return "\n\n".join(lines)


# Max number of distinct queries cached per chain instance (one per session).
_RETRIEVAL_CACHE_SIZE = 32


@traceable(run_type="chain", name="build_rag_chain")
def build_rag_chain(
    retriever,
    source_names: list[str],
    llm: ChatOpenAI,
    all_chunks: list[Document] | None = None,
):
    """
    Conversational RAG chain with global top-K retrieval.

    Args:
        retriever: HybridRetriever wrapping the merged FAISS store
        source_names: all source file paths in the indexed corpus
        llm: ChatOpenAI instance (Siemens-configured)
        all_chunks: flat list of every indexed chunk — used for direct page lookup
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

    # Build page index: (source_basename, page_label) → [chunks]
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

    def _needs_contextualization(query: str) -> bool:
        q = query.strip()
        if not q:
            return False
        if len(q.split()) <= 3:
            return True
        return bool(_REFERENTIAL.search(q))

    def _find_page_hint(query: str) -> str | None:
        m = _PAGE_RE.search(query)
        return m.group(1) if m else None

    def _find_source_hint(query: str) -> str | None:
        normalized = query.lower()
        for source in source_names:
            source_name = os.path.basename(source).lower()
            if source_name and source_name in normalized:
                return source
        return None

    # Per-chain LRU cache for retrieval results (keyed by effective query string).
    # Each chain instance is per-session and is rebuilt when new files are indexed,
    # so the cache is naturally invalidated on corpus changes.
    _retrieval_cache: dict[str, list[Document]] = {}

    def _get_docs(inputs: dict) -> list[Document]:
        """Retrieval with page-hint shortcut, hybrid search, and result caching."""
        raw_query = inputs.get("input", "")

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
                page_chunks = _page_label_index.get(page_label, [])
            if page_chunks:
                logger.debug("Page lookup: label=%s → %d chunk(s)", page_label, len(page_chunks))
                return page_chunks

        # --- Query cache (skips vector + BM25 search for repeated queries) ---
        if query in _retrieval_cache:
            logger.debug("Retrieval cache hit for query: %s...", query[:60])
            return _retrieval_cache[query]

        # HybridRetriever owns semantic + BM25 fusion, scoring, and the top-K cut
        # (and logs its own ranking when DEBUG_RETRIEVAL is on).
        docs = retriever.invoke(query)

        # Store in cache, evicting the oldest entry when full.
        if len(_retrieval_cache) >= _RETRIEVAL_CACHE_SIZE:
            _retrieval_cache.pop(next(iter(_retrieval_cache)))
        _retrieval_cache[query] = docs
        return docs

    qa_system_prompt = (
        "Answer using only the context provided. Do not hallucinate. "
        "Each context chunk is tagged with either [FILE: filename | page N | TYPE] "
        "or [WIKI: page title | url | TYPE]. "
        "When the user asks about a specific page, use the chunks from that page. "
        "If the requested page has no content in the context, say so explicitly. "
        "Cite the file and page number, or the wiki page title and its url, in your answer. "
        "When the user asks to list, enumerate, or extract items (such as tickets, "
        "action items, issues, IDs, or rows), reproduce EVERY matching item found in "
        "the context verbatim and in full — do NOT summarize, merge, or omit any of "
        "them. Preserve identifiers (e.g. ticket IDs) exactly as written. "
        "Otherwise, be concise."
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("system", "Use the following retrieved context to answer the question. If the answer is not contained in the context, say that you do not know."),
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
