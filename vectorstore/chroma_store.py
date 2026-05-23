# vectorstore/chroma_store.py

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME

def build_chroma_vectorstore(
    chunks: list[Document],
    embedding_model: HuggingFaceEndpointEmbeddings
) -> Chroma:
    """
    Build and persist a Chroma vector store from document chunks.
    """
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR
    )
    print(f"✅ Chroma vectorstore built with {len(chunks)} chunks.")
    return vectorstore

def load_chroma_vectorstore(
    embedding_model: HuggingFaceEndpointEmbeddings
) -> Chroma:
    """
    Load an existing persisted Chroma vector store.
    """
    vectorstore = Chroma(
        collection_name=CHROMA_COLLECTION_NAME,
        embedding_function=embedding_model,
        persist_directory=CHROMA_PERSIST_DIR
    )
    print("✅ Chroma vectorstore loaded from disk.")
    return vectorstore