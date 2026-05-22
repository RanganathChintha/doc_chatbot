# config.py

import os

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "your_groq_api_key_here")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "your_hf_api_key_here")

# --- Model Names ---
IMAGE_EXTRACTOR_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # Scout via Groq
LLM_MODEL = "gpt-oss-120b"                                           # LLM via Groq
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"           # HuggingFace

# --- Splitter Config ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Retriever Config ---
TOP_K = 4

# --- Chroma Config ---
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "rag_collection"