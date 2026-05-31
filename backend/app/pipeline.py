# pipeline.py
"""Shared document-loading + chunking pipeline used by both main.py (CLI)
and app.py (Streamlit)."""

import os
from app.loaders.pdf_loader import load_pdf_text, extract_images_from_pdf
from app.loaders.image_loader import load_image
from app.loaders.tabular_loader import load_csv, load_excel
from app.extractors.image_extractor import extract_text_from_images
from app.splitter.text_splitter import split_documents
from app.langsmith_tracing import langsmith_traceable as traceable


@traceable(run_type="tool", name="load_and_chunk")
def load_and_chunk(input_files: list[str], verbose: bool = True):
    """Parse all input files into Documents, then split into chunks.

    Args:
        input_files: list of file paths to process
        verbose: if True, print progress to stdout (good for CLI; turn off for UI)
    """
    all_documents = []
    for file_path in input_files:
        ext = os.path.splitext(file_path)[-1].lower()
        if verbose:
            print(f"\n📂 Processing: {file_path}")

        if ext == ".pdf":
            text_docs = load_pdf_text(file_path)
            all_documents.extend(text_docs)
            images = extract_images_from_pdf(file_path)
            if images:
                image_docs = extract_text_from_images(images, source=file_path)
                all_documents.extend(image_docs)

        elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            images = load_image(file_path)
            image_docs = extract_text_from_images(images, source=file_path)
            all_documents.extend(image_docs)

        elif ext == ".csv":
            all_documents.extend(load_csv(file_path))

        elif ext in [".xlsx", ".xls"]:
            all_documents.extend(load_excel(file_path))

        else:
            if verbose:
                print(f"Unsupported file type: {ext} - skipping.")

    if verbose:
        print(f"\nTotal documents collected: {len(all_documents)}")
    return split_documents(all_documents)
