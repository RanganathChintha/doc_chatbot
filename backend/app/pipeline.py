# pipeline.py
"""Shared document-loading + chunking pipeline used by both main.py (CLI)
and app.py (Streamlit)."""

import logging
import os
from app.loaders.pdf_loader import load_pdf_text, extract_images_from_pdf
from app.loaders.image_loader import load_image
from app.loaders.tabular_loader import load_csv, load_excel
from app.extractors.image_extractor import extract_text_from_images
from app.splitter.text_splitter import split_documents
from app.langsmith_tracing import langsmith_traceable as traceable

logger = logging.getLogger(__name__)


@traceable(run_type="tool", name="load_and_chunk")
def load_and_chunk(input_files: list[str], verbose: bool = True):
    """Parse all input files into Documents, then split into chunks.

    Args:
        input_files: list of file paths to process
        verbose: if True, print progress to stdout (good for CLI; turn off for UI)
    """
    logger.info("load_and_chunk: starting with %d file(s)", len(input_files))
    all_documents = []
    for file_path in input_files:
        ext = os.path.splitext(file_path)[-1].lower()
        if verbose:
            print(f"\n📂 Processing: {file_path}")
        logger.debug("Processing file: %s (type: %s)", file_path, ext)

        if ext == ".pdf":
            logger.debug("Loading PDF text...")
            text_docs = load_pdf_text(file_path)
            logger.debug("PDF text: %d document(s)", len(text_docs))
            all_documents.extend(text_docs)
            
            logger.debug("Extracting images from PDF...")
            images = extract_images_from_pdf(file_path)
            logger.debug("Found %d image(s) in PDF", len(images))
            if images:
                logger.debug("Extracting text from %d image(s)...", len(images))
                image_docs = extract_text_from_images(images, source=file_path)
                logger.debug("Image extraction complete: %d document(s)", len(image_docs))
                all_documents.extend(image_docs)
            logger.info("PDF processing complete: %d total document(s) so far", len(all_documents))

        elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            logger.debug("Loading image file...")
            images = load_image(file_path)
            logger.debug("Extracting text from %d image(s)...", len(images))
            image_docs = extract_text_from_images(images, source=file_path)
            logger.debug("Image text extraction complete: %d document(s)", len(image_docs))
            all_documents.extend(image_docs)

        elif ext == ".csv":
            logger.debug("Loading CSV...")
            csv_docs = load_csv(file_path)
            logger.debug("CSV loaded: %d document(s)", len(csv_docs))
            all_documents.extend(csv_docs)

        elif ext in [".xlsx", ".xls"]:
            logger.debug("Loading Excel...")
            excel_docs = load_excel(file_path)
            logger.debug("Excel loaded: %d document(s)", len(excel_docs))
            all_documents.extend(excel_docs)

        else:
            if verbose:
                print(f"Unsupported file type: {ext} - skipping.")
            logger.warning("Unsupported file type: %s", ext)

    if verbose:
        print(f"\nTotal documents collected: {len(all_documents)}")
    logger.info("load_and_chunk: collected %d total document(s)", len(all_documents))
    
    logger.debug("Starting text split...")
    chunked = split_documents(all_documents)
    logger.info("load_and_chunk: split into %d chunk(s), returning", len(chunked))
    return chunked
