# loaders/pdf_loader.py

import fitz  # PyMuPDF - for image extraction from PDF
from PIL import Image
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
import io

def load_pdf_text(file_path: str) -> list[Document]:
    """
    Extract text from PDF using PyPDFLoader.
    Returns a list of LangChain Document objects.
    """
    loader = PyPDFLoader(file_path)
    pages = loader.load()
    return pages

def extract_images_from_pdf(file_path: str) -> list[Image.Image]:
    """
    Extract embedded images from a PDF using PyMuPDF + PIL.
    Returns a list of PIL Image objects.
    """
    images = []
    pdf_doc = fitz.open(file_path)

    for page_num in range(len(pdf_doc)):
        page = pdf_doc[page_num]
        image_list = page.get_images(full=True)

        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = pdf_doc.extract_image(xref)
            image_bytes = base_image["image"]

            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            images.append(pil_image)

    pdf_doc.close()
    return images