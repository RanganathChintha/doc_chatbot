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
from openai import OpenAI, APITimeoutError
from langchain_core.documents import Document
from app.config import SIEMENS_API_KEY, SIEMENS_API_BASE, IMAGE_EXTRACTOR_MODEL, IMAGE_CACHE_FILE, CACHE_DIR
from app.langsmith_tracing import langsmith_traceable as traceable

logger = logging.getLogger(__name__)

vlm_client = None
if SIEMENS_API_KEY:
    try:
        # Initialize with explicit timeout to prevent hanging on slow responses
        vlm_client = OpenAI(api_key=SIEMENS_API_KEY, base_url=SIEMENS_API_BASE, timeout=60.0)
        logger.info("VLM client initialized with Siemens API (base_url=%s, timeout=60s)", SIEMENS_API_BASE)
    except Exception as e:
        logger.error("Failed to initialize VLM client: %s", e)
        vlm_client = None
else:
    logger.warning(
        "Siemens API key (OPENAI_API_KEY) is not set — image extraction will be skipped. "
        "Add it to your .env file to enable image analysis."
    )

_cache_lock = threading.Lock()
_cache: dict[str, str] | None = None

# Parallelism for image extraction — process multiple images concurrently
# Higher parallelism speeds up extraction but watch rate limits
_MAX_WORKERS = 8

# Keep at most this many entries in the on-disk image cache.
# Oldest entries (by insertion order, Python 3.7+ dict) are dropped first.
_MAX_CACHE_SIZE = 1000

# Resize and compress images before sending to the VLM.
# This dramatically reduces request payload size and improves latency.
MAX_IMAGE_DIMENSION = 1024
JPEG_QUALITY = 72


def _load_cache() -> dict[str, str]:
    global _cache
    if _cache is not None:
        return _cache
    os.makedirs(CACHE_DIR, exist_ok=True)
    if os.path.exists(IMAGE_CACHE_FILE):
        with open(IMAGE_CACHE_FILE, "r", encoding="utf-8") as f:
            loaded: dict[str, str] = json.load(f)
        # Evict oldest entries when the cache exceeds the size limit.
        if len(loaded) > _MAX_CACHE_SIZE:
            keys = list(loaded.keys())
            loaded = {k: loaded[k] for k in keys[-_MAX_CACHE_SIZE:]}
            _cache = loaded
            _save_cache()
        else:
            _cache = loaded
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


def _prepare_image_bytes(image: Image.Image) -> bytes:
    image = image.convert("RGB")
    width, height = image.size
    if max(width, height) > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(width, height)
        new_size = (int(width * scale), int(height * scale))
        resample = getattr(Image, "Resampling", Image).LANCZOS
        image = image.resize(new_size, resample)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buffer.getvalue()


def _call_vlm(image: Image.Image, source: str) -> str | None:
    """Single VLM round-trip. Returns the description text, or None on failure."""
    if vlm_client is None:
        logger.debug("VLM client is None, skipping image extraction for %s", source)
        return None
    
    try:
        logger.debug("Calling VLM for image from %s with model %s", source, IMAGE_EXTRACTOR_MODEL)
        image_bytes = _prepare_image_bytes(image)
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        response = vlm_client.chat.completions.create(
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
            temperature=0.0,
            max_tokens=120,
            timeout=30.0,
        )
        result = response.choices[0].message.content
        # Normalize the VLM content to a plain string. Different backends
        # may return a string, a list of content blocks, or a nested dict.
        def _ensure_str(content) -> str:
            if content is None:
                return ""
            if isinstance(content, str):
                return content
            # List of content blocks (common in multimodal responses)
            if isinstance(content, list):
                parts: list[str] = []
                for block in content:
                    if isinstance(block, str):
                        parts.append(block)
                    elif isinstance(block, dict):
                        # Common shapes: {"type":"text","text":"..."}
                        text = block.get("text") or block.get("value") or block.get("content")
                        if isinstance(text, str):
                            parts.append(text)
                        else:
                            # Try nested fields
                            if "message" in block and isinstance(block["message"], dict):
                                msg_text = block["message"].get("text")
                                if isinstance(msg_text, str):
                                    parts.append(msg_text)
                if parts:
                    return "\n".join(parts)
                # Fallback to JSON serialization
                try:
                    return json.dumps(content, ensure_ascii=False)
                except Exception:
                    return str(content)
            # Dict fallback
            if isinstance(content, dict):
                # Common: {"type":"output","text":"..."} or nested
                text = content.get("text") or content.get("value") or content.get("content")
                if isinstance(text, str):
                    return text
                # Try common nested keys
                for k in ("message", "output", "choices"):
                    v = content.get(k)
                    if isinstance(v, str):
                        return v
                    if isinstance(v, dict):
                        t = v.get("text")
                        if isinstance(t, str):
                            return t
                try:
                    return json.dumps(content, ensure_ascii=False)
                except Exception:
                    return str(content)
            return str(content)

        result_str = _ensure_str(result)
        logger.debug("VLM response received for %s: %s", source, result_str[:200] if result_str else None)
        return result_str
    except APITimeoutError as e:
        logger.error("VLM API timeout for %s (taking >50s): %s", source, e)
        return None
    except Exception as exc:
        logger.error("Image extraction failed for %s: %s", source, exc, exc_info=True)
        return None


@traceable(run_type="tool", name="extract_text_from_images")
def extract_text_from_images(images: list[Image.Image], source: str = "image") -> list[Document]:
    if vlm_client is None:
        logger.info("Skipping image extraction (no Siemens API key / OPENAI_API_KEY).")
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
