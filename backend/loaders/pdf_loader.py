# loaders/pdf_loader.py

import fitz  # PyMuPDF - for image extraction from PDF
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langsmith_tracing import langsmith_traceable as traceable
import io

MIN_IMAGE_DIMENSION = 100  # skip logos/icons/decorative images smaller than this

@traceable(run_type="tool", name="load_pdf_text")
def load_pdf_text(file_path: str) -> list[Document]:
    """
    Extract text from PDF using PyPDFLoader.
    Tags each page with source_type='text' so the retriever/LLM can distinguish.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    for p in pages:
        p.metadata["source_type"] = "text"
    return pages

@traceable(run_type="tool", name="extract_images_from_pdf")
def extract_images_from_pdf(file_path: str) -> list[Image.Image]:
    """
    Extract embedded images from a PDF using PyMuPDF + PIL.
    Filters out tiny images (logos, icons) below MIN_IMAGE_DIMENSION.
    """
    images = []
    seen_xrefs: set[int] = set()
    pdf_doc = fitz.open(file_path)

    for page_num in range(len(pdf_doc)):
        page = pdf_doc[page_num]
        image_list = page.get_images(full=True)

        for img in image_list:
            xref = img[0]
            if xref in seen_xrefs:
                continue
            seen_xrefs.add(xref)
            base_image = pdf_doc.extract_image(xref)
            image_bytes = base_image["image"]

            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            if pil_image.width < MIN_IMAGE_DIMENSION or pil_image.height < MIN_IMAGE_DIMENSION:
                continue
            images.append(pil_image)

    pdf_doc.close()
    return images