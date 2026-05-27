# vectorstore/chroma_store.py

import hashlib
import os

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import CHROMA_COLLECTION_NAME, CHROMA_PERSIST_DIR
from langsmith_tracing import langsmith_traceable as traceable

MetadataValue = str | int | float | bool


def _document_id(doc: Document) -> str:
    source = doc.metadata.get("source", "")
    page = doc.metadata.get("page", "")
    row = doc.metadata.get("row", "")
    chunk = doc.metadata.get("chunk", "")
    chunk_id = doc.metadata.get("chunk_id", "")
    raw = f"{source}|{page}|{row}|{chunk}|{chunk_id}|{doc.page_content}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _chroma_safe_document(doc: Document) -> Document:
    metadata: dict[str, MetadataValue] = {}
    for key, value in doc.metadata.items():
        if value is None:
            continue
        if isinstance(value, str | int | float | bool):
            metadata[key] = value
        else:
            metadata[key] = str(value)
    return Document(page_content=doc.page_content, metadata=metadata)


def _chroma_vectorstore(embedding_model) -> Chroma:
    os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
    return Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_PERSIST_DIR,
    )


def _document_ids(chunks: list[Document]) -> list[str]:
    seen: dict[str, int] = {}
    ids: list[str] = []
    for doc in chunks:
        base_id = _document_id(doc)
        occurrence = seen.get(base_id, 0)
        seen[base_id] = occurrence + 1
        if occurrence == 0:
            ids.append(base_id)
        else:
            ids.append(f"{base_id}-{occurrence}")
    return ids


@traceable(run_type="tool", name="persist_chunks_to_chroma")
def persist_chunks_to_chroma(
    chunks: list[Document],
    embedding_model,
    source_files: list[str] | None = None,
) -> Chroma:
    """
    Persist chunks to ChromaDB for durable storage/inspection.

    FAISS remains the active retriever. Chroma is kept in sync when documents
    are indexed so deployments can retain a database-style copy of chunks and
    embeddings if the host provides persistent storage.
    """
    vectorstore = _chroma_vectorstore(embedding_model)

    sources = source_files or sorted({
        doc.metadata.get("source", "")
        for doc in chunks
        if doc.metadata.get("source")
    })
    for source in sources:
        try:
            vectorstore.delete(where={"source": source})
        except Exception:
            pass

    if chunks:
        vectorstore.add_documents(
            [_chroma_safe_document(doc) for doc in chunks],
            ids=_document_ids(chunks),
        )

    print(f"ChromaDB persisted {len(chunks)} chunk(s) to {CHROMA_PERSIST_DIR}.")
    return vectorstore


@traceable(run_type="tool", name="clear_chroma_vectorstore")
def clear_chroma_vectorstore(embedding_model) -> None:
    """Clear the configured Chroma collection without affecting FAISS code."""
    vectorstore = _chroma_vectorstore(embedding_model)
    try:
        vectorstore.delete_collection()
    except Exception:
        pass
