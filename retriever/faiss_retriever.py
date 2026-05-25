# retriever/faiss_retriever.py

from collections import defaultdict
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config import TOP_K


def build_faiss_retriever(chunks: list[Document], embedding_model):
    """
    Single global FAISS index over all chunks (legacy / fallback).
    """
    faiss_store = FAISS.from_documents(documents=chunks, embedding=embedding_model)
    retriever = faiss_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    print(f"✅ FAISS retriever ready — will fetch top {TOP_K} chunks.")
    return retriever


def build_per_source_retrievers(
    chunks: list[Document],
    embedding_model,
    k_per_source: int = 3,
) -> tuple[dict, dict]:
    """
    Build one FAISS retriever per source file so retrieval is guaranteed to
    cover every uploaded file (and not just the globally most-similar chunks).

    Returns:
        retrievers: {source_path -> Retriever} — for query-time per-file similarity
        by_source: {source_path -> list[Document]} — all chunks per file, used
                   when a query asks for full per-file context (e.g. summaries)
    """
    by_source: dict[str, list[Document]] = defaultdict(list)
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        by_source[src].append(c)

    retrievers: dict = {}
    for source, source_chunks in by_source.items():
        k = min(k_per_source, len(source_chunks))
        if k == 0:
            continue
        store = FAISS.from_documents(source_chunks, embedding_model)
        retrievers[source] = store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    print(
        f"✅ Per-file retrievers ready: {len(retrievers)} file(s), "
        f"top-{k_per_source} per file."
    )
    return retrievers, dict(by_source)
