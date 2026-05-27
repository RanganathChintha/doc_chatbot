# splitter/text_splitter.py

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langsmith import traceable
from config import CHUNK_SIZE, CHUNK_OVERLAP

@traceable(run_type="tool", name="Splitting_the_text_into_chunks")
def split_documents(documents: list[Document]) -> list[Document]:
    """
    Split a list of LangChain Documents into smaller chunks
    using RecursiveCharacterTextSplitter.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", " ", ""]
    )
    chunks = splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")
    return chunks