# loaders/tabular_loader.py

import logging

import pandas as pd
from langchain_core.documents import Document
from pandas.errors import EmptyDataError, ParserError
from app.langsmith_tracing import langsmith_traceable as traceable

logger = logging.getLogger(__name__)

@traceable(run_type="tool", name="load_csv")
def load_csv(file_path: str) -> list[Document]:
    """
    Load a CSV file and convert each row to a LangChain Document.
    """
    try:
        df = pd.read_csv(
            file_path,
            dtype=str,
            engine="python",
            on_bad_lines="skip",
            encoding_errors="replace",
        )
    except (ParserError, UnicodeDecodeError, EmptyDataError) as exc:
        logger.warning(
            "CSV load failed for %s with %s; retrying with fallback parser",
            file_path,
            exc,
        )
        try:
            df = pd.read_csv(
                file_path,
                dtype=str,
                engine="python",
                on_bad_lines="skip",
                encoding="utf-8",
                encoding_errors="replace",
            )
        except Exception as exc2:
            logger.exception("Failed to load CSV %s", file_path)
            raise ValueError(
                f"CSV parse failed for {file_path}: {exc2}"
            ) from exc2

    if df.empty:
        logger.warning("CSV %s parsed with no rows", file_path)

    return _dataframe_to_documents(df, source=file_path)

@traceable(run_type="tool", name="load_excel")
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