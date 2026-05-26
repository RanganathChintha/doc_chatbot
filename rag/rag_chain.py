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
from config import GROQ_API_KEY, LLM_MODEL, LLM_CACHE_FILE, CACHE_DIR, CHAT_HISTORY_WINDOW
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
    per_source_retrievers: dict,
    per_source_chunks: dict,
    llm: ChatGroq,
):
    """
    Conversational RAG chain with per-file retrieval.

    Args:
        per_source_retrievers: {source_path -> Retriever} produced by
            retriever.faiss_retriever.build_per_source_retrievers
        per_source_chunks: {source_path -> list[Document]} (all chunks per file)
        llm: ChatGroq instance

    Behavior:
      - For normal queries: pulls top-K from EACH file independently, so every
        uploaded file gets representation in the answer.
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

    def _get_docs(inputs: dict) -> list[Document]:
        """Per-file retrieval for all queries."""
        raw_query = inputs.get("input", "")

        # Contextualize the question if there is a chat history.
        if inputs.get("chat_history"):
            query = contextualize_question.invoke(inputs)
        else:
            query = raw_query

        docs = []
        for source, retriever in per_source_retrievers.items():
            docs.extend(retriever.invoke(query))
        return docs

    qa_system_prompt = (
        "You are a careful, concise assistant answering questions about the "
        "user's uploaded documents.\n\n"
        "RULES:\n"
        "1. Use the context below. If the answer is not present, say "
        "   \"I don't have enough information to answer that.\"\n"
        "2. Synthesize and rephrase in your own words. Do NOT copy long "
        "   passages verbatim from the context.\n"
        "3. Respect any length / format the user asks for (e.g. \"in 5 lines\", "
        "   \"as a table\", \"per file\"). If the user does not specify, be brief.\n"
        "4. When the question is about each / every / per file, structure the "
        "   answer with one section per file, using the file name as a heading.\n"
        "5. Context chunks are tagged [FILE: <name> | TEXT] (authoritative) or "
        "   [FILE: <name> | IMAGE] (vision-model description — supporting only).\n\n"
        "Context:\n{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    rag_chain = (
        RunnablePassthrough.assign(
            docs=RunnableLambda(_get_docs),
        )
        | RunnablePassthrough.assign(
            context=lambda x: _format_tagged_context(x["docs"]),
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
