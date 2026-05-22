# splitter/text_splitter.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from config import CHUNK_SIZE, CHUNK_OVERLAP

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
    print(f"✅ Split into {len(chunks)} chunks.")
    return chunks