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
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# --- Retriever ---
# Higher TOP_K gives the LLM more of the document at once, which matters for
# "list all X" / aggregation queries (tickets, action items) where the items are
# spread across many chunks. Costs more tokens per request — tune down if needed.
TOP_K = 4
DEBUG_RETRIEVAL = True

# --- Cache locations ---
CACHE_DIR = "./cache"
IMAGE_CACHE_FILE = "./cache/image_extractions.json"
EMBEDDING_CACHE_DIR = "./cache/embeddings"
LLM_CACHE_FILE = "./cache/llm_responses.sqlite"

# --- Memory ---
CHAT_HISTORY_WINDOW = 5
