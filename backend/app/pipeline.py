# pipeline.py
"""Shared document-loading + chunking pipeline used by both main.py (CLI)
and app.py (Streamlit)."""

import logging
import os
from app.ingestion.azure_wiki_fetcher import fetch_wiki_pages, fetch_wiki_pages_from_url
from app.loaders.pdf_loader import load_pdf_text, extract_images_from_pdf
from app.loaders.image_loader import load_image
from app.loaders.tabular_loader import load_csv, load_excel
from app.extractors.image_extractor import extract_text_from_images
from app.splitter.text_splitter import split_documents
from app.langsmith_tracing import langsmith_traceable as traceable
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


@traceable(run_type="tool", name="load_and_chunk")
def load_and_chunk(input_files: list[str], verbose: bool = True):
    """Parse all input files into Documents, then split into chunks."""
    logger.info("load_and_chunk: starting with %d file(s)", len(input_files))
    all_documents = []
    for file_path in input_files:
        ext = os.path.splitext(file_path)[-1].lower()
        before = len(all_documents)
        try:
            if ext == ".pdf":
                all_documents.extend(load_pdf_text(file_path))
                images = extract_images_from_pdf(file_path)
                if images:
                    all_documents.extend(extract_text_from_images(images, source=file_path))

            elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
                all_documents.extend(
                    extract_text_from_images(load_image(file_path), source=file_path)
                )

            elif ext == ".csv":
                all_documents.extend(load_csv(file_path))

            elif ext in [".xlsx", ".xls"]:
                all_documents.extend(load_excel(file_path))

            else:
                logger.warning("Unsupported file type: %s", ext)
                continue
        except Exception:
            logger.exception("Failed to load %s — skipping", file_path)
            continue

        logger.info("load_and_chunk: %s → %d doc(s)", os.path.basename(file_path), len(all_documents) - before)

    logger.info("load_and_chunk: %d doc(s) total, splitting...", len(all_documents))
    chunked = split_documents(all_documents)
    logger.info("load_and_chunk: %d chunk(s) produced", len(chunked))
    return chunked


def load_and_chunk_wiki(
    pat: str,
    wiki_id: str,
    target_page_id: int,
    organization: str | None = None,
    project: str | None = None,
) -> list[Document]:
    """Fetch Azure Wiki pages via the standalone fetcher, then split into chunks.

    Wraps the clean dict output from ``fetch_wiki_pages`` into LangChain
    ``Document`` objects so downstream consumers (splitters, retrievers) can
    work with them as usual.
    """
    raw = fetch_wiki_pages(
        pat=pat,
        wiki_id=wiki_id,
        target_page_id=target_page_id,
        organization=organization,
        project=project,
    )
    logger.info("load_and_chunk_wiki: %d page(s) fetched", len(raw))
    documents = [
        Document(
            page_content=doc["content"],
            metadata={k: v for k, v in doc.items() if k != "content"},
        )
        for doc in raw
    ]
    chunked = split_documents(documents)
    logger.info("load_and_chunk_wiki: %d chunk(s) produced", len(chunked))
    return chunked


def load_and_chunk_wiki_url(pat: str, wiki_url: str) -> list[Document]:
    """Fetch Azure Wiki pages from a browser URL, then split into chunks."""
    raw = fetch_wiki_pages_from_url(pat=pat, wiki_url=wiki_url)
    logger.info("load_and_chunk_wiki_url: %d page(s) fetched", len(raw))
    documents = [
        Document(
            page_content=doc["content"],
            metadata={k: v for k, v in doc.items() if k != "content"},
        )
        for doc in raw
    ]
    chunked = split_documents(documents)
    logger.info("load_and_chunk_wiki_url: %d chunk(s) produced", len(chunked))
    return chunked
