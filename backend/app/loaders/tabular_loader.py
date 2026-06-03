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

_ROWS_PER_DOC = 25  # batch size: keeps embedding count low while preserving context


def _dataframe_to_documents(df: pd.DataFrame, source: str) -> list[Document]:
    """Convert a DataFrame to Documents, batching rows to reduce embedding count.

    25 rows per document means a 1,000-row CSV produces 40 embeddings instead of 1,000.
    Rows within a batch stay together in retrieval, giving the LLM useful context.
    """
    documents = []
    for start in range(0, len(df), _ROWS_PER_DOC):
        batch = df.iloc[start : start + _ROWS_PER_DOC]
        lines = []
        for idx, row in batch.iterrows():
            lines.append(f"[Row {idx}] " + " | ".join(f"{col}: {val}" for col, val in row.items()))
        documents.append(Document(
            page_content="\n".join(lines),
            metadata={"source": source, "row_start": int(batch.index[0]), "row_end": int(batch.index[-1])},
        ))
    return documents