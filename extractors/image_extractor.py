# extractors/image_extractor.py

import base64
import hashlib
import io
import json
import os
import threading
from PIL import Image
from groq import Groq
from langchain_core.documents import Document
from config import GROQ_API_KEY, IMAGE_EXTRACTOR_MODEL, IMAGE_CACHE_FILE, CACHE_DIR

groq_client = Groq(api_key=GROQ_API_KEY)

_cache_lock = threading.Lock()
_cache: dict[str, str] | None = None


def _load_cache() -> dict[str, str]:
    global _cache
    if _cache is not None:
        return _cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(IMAGE_CACHE_FILE):
        with open(IMAGE_CACHE_FILE, "r", encoding="utf-8") as f:
            _cache = json.load(f)
    else:
        _cache = {}
    return _cache


def _save_cache() -> None:
    with _cache_lock:
        with open(IMAGE_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False, indent=2)


def _image_hash(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def pil_image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def extract_text_from_image(image: Image.Image, source: str = "image") -> Document:
    """
    Use the Scout VLM (via Groq) to summarize an image.
    Results are cached on disk by SHA-256 of the image bytes so repeated
    runs over the same PDF do not re-call the API.
    """
    cache = _load_cache()
    key = _image_hash(image)

    if key in cache:
        description = cache[key]
    else:
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
                                "Summarize this image in 2-3 concise sentences. "
                                "Focus only on the key information (main subject, "
                                "any visible text, key data points). "
                                "Skip decorative details. If the image is a logo, "
                                "icon, or has no substantive content, reply exactly: SKIP."
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
            max_tokens=256
        )
        description = response.choices[0].message.content
        cache[key] = description
        _save_cache()

    return Document(
        page_content=description,
        metadata={"source": source, "source_type": "image"}
    )


def extract_text_from_images(images: list[Image.Image], source: str = "image") -> list[Document]:
    documents = []
    hits = 0
    misses = 0
    cache = _load_cache()
    for idx, image in enumerate(images):
        key = _image_hash(image)
        if key in cache:
            hits += 1
        else:
            misses += 1
        doc = extract_text_from_image(image, source=f"{source}_img_{idx}")
        if doc.page_content.strip().upper().startswith("SKIP"):
            continue
        documents.append(doc)
    print(f"   image cache: {hits} hit / {misses} miss")
    return documents
