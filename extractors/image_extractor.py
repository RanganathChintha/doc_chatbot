# extractors/image_extractor.py

import base64
import io
from PIL import Image
from groq import Groq
from langchain.schema import Document
from config import GROQ_API_KEY, IMAGE_EXTRACTOR_MODEL

# Initialize Groq client for Scout model
groq_client = Groq(api_key=GROQ_API_KEY)

def pil_image_to_base64(image: Image.Image) -> str:
    """
    Convert a PIL Image to a base64-encoded JPEG string.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")

def extract_text_from_image(image: Image.Image, source: str = "image") -> Document:
    """
    Use the Scout model (via Groq) to extract descriptive text from a PIL image.
    Returns a LangChain Document with the extracted description.
    """
    base64_image = pil_image_to_base64(image)

    response = groq_client.chat.completions.create(
        model=IMAGE_EXTRACTOR_MODEL,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Please analyze this image thoroughly. "
                            "Describe all visible text, data, charts, diagrams, "
                            "objects, and any other relevant information in detail."
                        )
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        max_tokens=1024
    )

    description = response.choices[0].message.content
    return Document(
        page_content=description,
        metadata={"source": source, "type": "image_extraction"}
    )

def extract_text_from_images(images: list[Image.Image], source: str = "image") -> list[Document]:
    """
    Process a list of PIL images through the Scout model.
    Returns a list of LangChain Documents.
    """
    documents = []
    for idx, image in enumerate(images):
        doc = extract_text_from_image(image, source=f"{source}_img_{idx}")
        documents.append(doc)
    return documents