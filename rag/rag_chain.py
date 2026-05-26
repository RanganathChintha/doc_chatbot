# rag/rag_chain.py

import os
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory, InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from config import (
    GROQ_API_KEY,
    LLM_MODEL,
    LLM_CACHE_FILE,
    CACHE_DIR,
    CHAT_HISTORY_WINDOW,
    TOP_K,
    RETRIEVAL_SCORE_THRESHOLD,
    DEBUG_RETRIEVAL,
)
from langsmith_tracing import langsmith_traceable as traceable

os.makedirs(CACHE_DIR, exist_ok=True)
set_llm_cache(SQLiteCache(database_path=LLM_CACHE_FILE))

_session_histories: dict[str, InMemoryChatMessageHistory] = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """Per-session window-buffer chat history."""
    history = _session_histories.get(session_id)
    if history is None:
        history = InMemoryChatMessageHistory()
        _session_histories[session_id] = history
    max_msgs = CHAT_HISTORY_WINDOW * 2
    if len(history.messages) > max_msgs:
        history.messages = history.messages[-max_msgs:]
    return history


@traceable(run_type="tool", name="get_llm")
def get_llm() -> ChatGroq:
    llm = ChatGroq(
        api_key=GROQ_API_KEY,
        model_name=LLM_MODEL,
        temperature=0.2,
    )
    print(f"✅ LLM loaded: {LLM_MODEL} via ChatGroq (SQLite response cache active)")
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

    def _find_source_hint(query: str) -> str | None:
        normalized = query.lower()
        for source in source_names:
            source_name = os.path.basename(source).lower()
            if source_name and source_name in normalized:
                return source
        return None

    def _get_docs(inputs: dict) -> list[Document]:
        """Global top-K retrieval for all queries."""
        raw_query = inputs.get("input", "")

        # Contextualize the question if there is a chat history.
        if inputs.get("chat_history"):
            query = contextualize_question.invoke(inputs)
        else:
            query = raw_query

        if hasattr(retriever, "similarity_search_with_score"):
            results = retriever.similarity_search_with_score(query, k=TOP_K)
            source_hint = _find_source_hint(query)
            if source_hint:
                results.sort(
                    key=lambda pair: pair[0].metadata.get("source") == source_hint,
                    reverse=True,
                )
            filtered = [doc for doc, score in results if score >= RETRIEVAL_SCORE_THRESHOLD]
            if not filtered:
                filtered = [doc for doc, _ in results]
            docs = filtered[:TOP_K]
            if DEBUG_RETRIEVAL:
                print("🧠 Retrieval debug:")
                for doc, score in results[:TOP_K]:
                    print(f"  - source={doc.metadata.get('source')} score={score:.4f}")
            return docs

        return retriever.invoke(query)

    qa_system_prompt = (
        "Answer using only the context. Do not hallucinate. "
        "If unavailable, say so. Cite sources. Be concise."
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
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
