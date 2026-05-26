# config.py

import os
from dotenv import load_dotenv

load_dotenv()

# --- API Keys ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
LANGSMITH_PROJECT_NAME = os.getenv("LANGSMITH_PROJECT_NAME", "doc_chatbot")

if LANGSMITH_API_KEY:
    os.environ.setdefault("LANGCHAIN_API_KEY", LANGSMITH_API_KEY)

# --- Model Names ---
IMAGE_EXTRACTOR_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"  # Scout via Groq
LLM_MODEL = "openai/gpt-oss-120b"               # LLM via Groq
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"           # HuggingFace

# --- Splitter Config ---
# Smaller chunks with moderate overlap improve retrieval precision and speed.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# --- Retriever Config ---
TOP_K = 5
RETRIEVAL_SCORE_THRESHOLD = 0.1
DEBUG_RETRIEVAL = False

# --- Chroma Config ---
CHROMA_PERSIST_DIR = "./chroma_db"
CHROMA_COLLECTION_NAME = "rag_collection"

# --- Cache Config ---
CACHE_DIR = "./cache"
IMAGE_CACHE_FILE = "./cache/image_extractions.json"
EMBEDDING_CACHE_DIR = "./cache/embeddings"
LLM_CACHE_FILE = "./cache/llm_responses.sqlite"
MANIFEST_FILE = "./cache/source_manifest.json"

# --- Memory Config ---
CHAT_HISTORY_WINDOW = 5  # number of recent turns to keep in context