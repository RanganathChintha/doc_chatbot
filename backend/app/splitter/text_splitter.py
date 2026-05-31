# splitter/text_splitter.py

import logging

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.langsmith_tracing import langsmith_traceable as traceable
from app.config import CHUNK_SIZE, CHUNK_OVERLAP

logger = logging.getLogger(__name__)


@traceable(run_type="tool", name="Splitting_the_text_into_chunks")
def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    logger.info("Split into %d chunks.", len(chunks))
    return chunks