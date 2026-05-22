# retriever/faiss_retriever.py

from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_chroma import Chroma
from config import TOP_K

def build_faiss_retriever(
    chunks: list[Document],
    embedding_model: HuggingFaceEndpointEmbeddings
):
    """
    Build a FAISS vector store from chunks and return a retriever.
    FAISS is used here as the retrieval layer (top-K similarity search).
    """
    faiss_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model
    )
    retriever = faiss_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K}
    )
    print(f"✅ FAISS retriever ready — will fetch top {TOP_K} chunks.")
    return retriever