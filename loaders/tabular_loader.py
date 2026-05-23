# loaders/tabular_loader.py

import pandas as pd
from langchain_core.documents import Document

def load_csv(file_path: str) -> list[Document]:
    """
    Load a CSV file and convert each row to a LangChain Document.
    """
    df = pd.read_csv(file_path)
    return _dataframe_to_documents(df, source=file_path)

def load_excel(file_path: str) -> list[Document]:
    """
    Load an Excel file and convert each row to a LangChain Document.
    """
    df = pd.read_excel(file_path)
    return _dataframe_to_documents(df, source=file_path)

def _dataframe_to_documents(df: pd.DataFrame, source: str) -> list[Document]:
    """
    Convert a DataFrame to a list of LangChain Document objects.
    Each row becomes one document with all column values as text.
    """
    documents = []
    for idx, row in df.iterrows():
        content = "\n".join([f"{col}: {val}" for col, val in row.items()])
        doc = Document(
            page_content=content,
            metadata={"source": source, "row": idx}
        )
        documents.append(doc)
    return documents