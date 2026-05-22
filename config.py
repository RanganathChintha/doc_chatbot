# config.py

import os

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")

# --- Model Names ---
IMAGE_EXTRACTOR_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # Scout via Groq
LLM_MODEL = "openai/gpt-oss-120b"                                           # LLM via Groq
EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-8B" #"sentence-transformers/all-minilm-l6-v2"           # HuggingFace

# --- Splitter Config ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150

# --- Retriever Config ---
TOP_K = 4

# --- Chroma Config ---
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "rag_collection"