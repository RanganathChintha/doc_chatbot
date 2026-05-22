# embeddings/embedder.py

from langchain_huggingface import HuggingFaceEndpointEmbeddings
from config import HUGGINGFACE_API_KEY, EMBEDDING_MODEL

def get_embedding_model() -> HuggingFaceEndpointEmbeddings:
    """
    Initialize and return the all-MiniLM embedding model
    via the HuggingFace Inference API.
    """
    embeddings = HuggingFaceEndpointEmbeddings(
        model=EMBEDDING_MODEL,
        huggingfacehub_api_token=HUGGINGFACE_API_KEY
    )
    print(f"✅ Embedding model loaded: {EMBEDDING_MODEL}")
    return embeddings