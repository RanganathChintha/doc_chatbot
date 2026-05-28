# config.py

import logging
import os
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# --- API Keys ---
# Note: LangSmith config lives entirely in langsmith_tracing.py.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# --- Model Names ---
IMAGE_EXTRACTOR_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
LLM_MODEL = "openai/gpt-oss-120b"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Splitter ---
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120

# --- Retriever ---
TOP_K = 10
RETRIEVAL_SCORE_THRESHOLD = 0.3
DEBUG_RETRIEVAL = False

# --- Cache locations ---
CACHE_DIR = "./cache"
IMAGE_CACHE_FILE = "./cache/image_extractions.json"
EMBEDDING_CACHE_DIR = "./cache/embeddings"
LLM_CACHE_FILE = "./cache/llm_responses.sqlite"

# --- Memory ---
CHAT_HISTORY_WINDOW = 5
