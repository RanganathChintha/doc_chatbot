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
# Siemens API key: stored in the environment variable named OPENAI_API_KEY
SIEMENS_API_KEY = os.getenv("OPENAI_API_KEY")
SIEMENS_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.siemens.com/llm/v1")

# --- Model Names ---
IMAGE_EXTRACTOR_MODEL = "pixtral-12b-2409"
LLM_MODEL = "gpt-oss-120b-onprem"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Splitter ---
CHUNK_SIZE = 10000
CHUNK_OVERLAP = 1000

# --- Retriever ---
TOP_K = 6
RETRIEVAL_SCORE_THRESHOLD = 0.3
DEBUG_RETRIEVAL = False

# --- Cache locations ---
CACHE_DIR = "./cache"
IMAGE_CACHE_FILE = "./cache/image_extractions.json"
EMBEDDING_CACHE_DIR = "./cache/embeddings"
LLM_CACHE_FILE = "./cache/llm_responses.sqlite"

# --- Memory ---
CHAT_HISTORY_WINDOW = 5
