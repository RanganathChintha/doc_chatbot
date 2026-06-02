# extractors/image_extractor.py

import base64
import hashlib
import io
import json
import logging
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from PIL import Image
from groq import Groq
from langchain_core.documents import Document
from app.config import GROQ_API_KEY, IMAGE_EXTRACTOR_MODEL, IMAGE_CACHE_FILE, CACHE_DIR
from app.langsmith_tracing import langsmith_traceable as traceable

logger = logging.getLogger(__name__)

if not GROQ_API_KEY:
    logger.warning(
        "GROQ_API_KEY is not set — image extraction will be skipped. "
        "Add it to your .env file to enable image analysis."
    )

# max_retries=0 — we handle failures ourselves and don't want the SDK to
# silently retry and amplify 429s when many images are processed in parallel.
groq_client = Groq(api_key=GROQ_API_KEY, max_retries=0) if GROQ_API_KEY else None

_cache_lock = threading.Lock()
_cache: dict[str, str] | None = None

# Keep parallelism low to avoid blowing through the per-minute token budget.
_MAX_WORKERS = 2


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


def _call_vlm(image: Image.Image, source: str) -> str | None:
    """Single VLM round-trip. Returns the description text, or None on failure."""
    try:
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
                                "Summarize this image. "
                                "Focus only on the key information, main subject, "
                                "any visible text, key data points. "
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
        return response.choices[0].message.content
    except Exception as exc:
        logger.warning("Image extraction failed for %s: %s", source, exc)
        return None


@traceable(run_type="tool", name="extract_text_from_image")
def extract_text_from_image(image: Image.Image, source: str = "image") -> Document | None:
    """
    Summarize a single image via the VLM, with on-disk caching by image hash.
    Kept for callers that handle one image at a time; for bulk work prefer
    extract_text_from_images() which parallelizes.
    """
    if groq_client is None:
        return None

    cache = _load_cache()
    key = _image_hash(image)
    description = cache.get(key)
    if description is None:
        description = _call_vlm(image, source)
        if description is None:
            return None
        cache[key] = description
        _save_cache()

    metadata = {"source": source, "source_type": "image"}
    # If the image was extracted from a PDF, include page info when available.
    try:
        page = image.info.get("page")
    except Exception:
        page = None
    if page is not None:
        metadata["page"] = page
    return Document(
        page_content=description,
        metadata=metadata,
    )


@traceable(run_type="tool", name="extract_text_from_images")
def extract_text_from_images(images: list[Image.Image], source: str = "image") -> list[Document]:
    if groq_client is None:
        logger.info("Skipping image extraction (no GROQ_API_KEY).")
        return []
    if not images:
        return []

    cache = _load_cache()

    # Dedupe by content hash so identical images (e.g. same logo on every page)
    # only get hashed/sent once even within this call.
    unique: dict[str, Image.Image] = {}
    pages_map: dict[str, list[int | None]] = {}
    keys_in_order: list[str] = []
    for image in images:
        key = _image_hash(image)
        keys_in_order.append(key)
        if key not in unique:
            unique[key] = image
            # Record page occurrences for this image (may be None)
            try:
                pages_map[key] = [image.info.get("page")]
            except Exception:
                pages_map[key] = [None]
        else:
            try:
                pages_map[key].append(image.info.get("page"))
            except Exception:
                pages_map[key].append(None)

    hits = sum(1 for k in unique if k in cache)
    to_fetch = [(k, img) for k, img in unique.items() if k not in cache]

    if to_fetch:
        with ThreadPoolExecutor(max_workers=min(_MAX_WORKERS, len(to_fetch))) as pool:
            future_to_key = {
                pool.submit(_call_vlm, img, source): k for k, img in to_fetch
            }
            for fut in as_completed(future_to_key):
                k = future_to_key[fut]
                desc = fut.result()
                if desc is not None:
                    cache[k] = desc
        # One disk write per batch instead of one per image.
        _save_cache()

    documents: list[Document] = []
    seen_emitted: set[str] = set()
    for k in keys_in_order:
        if k in seen_emitted:
            continue
        seen_emitted.add(k)
        description = cache.get(k)
        if description is None:
            continue
        if description.strip().upper().startswith("SKIP"):
            continue
        # Attach page info: prefer list of pages (may include duplicates), and keep
        # compatibility by exposing the first page as `page`.
        pages = pages_map.get(k, [])
        metadata = {"source": source, "source_type": "image"}
        if pages:
            # normalize pages: keep unique, preserve order
            seen = []
            for p in pages:
                if p not in seen:
                    seen.append(p)
            metadata["pages"] = seen
            # also include a single `page` key for backward compatibility
            if seen[0] is not None:
                metadata["page"] = seen[0]

        documents.append(Document(
            page_content=description,
            metadata=metadata,
        ))

    logger.info(
        "Image extraction: %d unique image(s), %d cache hit(s), %d API call(s), %d doc(s) produced.",
        len(unique), hits, len(to_fetch), len(documents),
    )
    return documents
