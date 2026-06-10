# pipeline.py
"""Shared document-loading + chunking pipeline: parses uploaded files and Azure
DevOps wiki pages into chunked Documents for indexing."""

import logging
import os
from app.ingestion.azure_wiki_fetcher import fetch_wiki_pages_from_url
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


@traceable(run_type="tool", name="load_and_chunk_wiki_url")
def load_and_chunk_wiki_url(pat: str, wiki_url: str):
    """Fetch an Azure DevOps wiki page and all of its child pages, then chunk.

    `fetch_wiki_pages_from_url` calls the Wiki REST API with recursionLevel=full,
    so a single root URL yields the target page plus every descendant page with
    its markdown content. Each page becomes one Document (keyed by a unique
    per-page `source`) before splitting, so retrieval and per-source dedup work
    the same way they do for uploaded files.
    """
    logger.info("load_and_chunk_wiki_url: fetching wiki pages...")
    pages = fetch_wiki_pages_from_url(pat=pat, wiki_url=wiki_url)
    logger.info("load_and_chunk_wiki_url: fetched %d page(s)", len(pages))

    documents = []
    for page in pages:
        content = (page.get("content") or "").strip()
        if not content:
            # Container/parent pages often hold only a list of child links and no
            # body text — skip them so they don't add empty, noisy chunks.
            logger.debug("load_and_chunk_wiki_url: skipping empty page %s", page.get("path"))
            continue
        documents.append(Document(
            page_content=content,
            metadata={
                "source": page.get("source", "Azure Wiki"),
                "source_type": "wiki",
                "title": page.get("title", ""),
                "remote_url": page.get("remote_url", ""),
                "path": page.get("path", ""),
                "page_id": page.get("id"),
            },
        ))

    logger.info("load_and_chunk_wiki_url: %d page(s) with content, splitting...", len(documents))
    chunked = split_documents(documents)
    logger.info("load_and_chunk_wiki_url: %d chunk(s) produced", len(chunked))
    return chunked
