# api.py — FastAPI routes. Business logic lives in services/.
#
# Run from the backend/ directory with:
#   uvicorn app.api:app --reload --port 8000

import os

# HF offline flags must be set before any HF library is imported transitively.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import asyncio
import glob
import io
import logging
import shutil
import time
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse

from app.rag.rag_chain import get_session_history
from app.services.chat import stream_chat
from app.services.indexing import clear_session, get_state, index_files, index_wiki_url, remove_file
from app.retriever.hybrid_retriever import _bm25_cache_file
from app.services.schemas import ChatRequest, ResetRequest, UploadResponse, WikiIngestRequest

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# CORS origins: comma-separated list via env var, defaulting to local dev ports.
_ALLOWED_ORIGINS = os.getenv(
    "ALLOWED_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173"
).split(",")


def _session_dir(session_id: str) -> Path:
    """Per-session upload directory. Sanitize the id so it can't escape UPLOAD_DIR."""
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in session_id)
    if not safe:
        raise HTTPException(status_code=400, detail="Invalid session_id.")
    d = UPLOAD_DIR / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


app = FastAPI(title="Doc Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def _log_requests(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    logger.info("%s %s → %d (%.0f ms)", request.method, request.url.path, response.status_code, ms)
    return response


@app.on_event("startup")
async def _startup() -> None:
    """Pre-warm embedding model, clean up stale artifacts, start background eviction."""
    from app.services.resources import embedding_model
    from app.config import CACHE_DIR

    await asyncio.to_thread(embedding_model)
    logger.info("Startup: embedding model pre-warmed.")

    # Remove BM25 caches whose upload directory no longer exists (orphaned).
    for pkl in glob.glob(f"{CACHE_DIR}/bm25_retriever_*.pkl"):
        session_id = Path(pkl).stem.removeprefix("bm25_retriever_")
        if session_id and not (UPLOAD_DIR / session_id).exists():
            Path(pkl).unlink(missing_ok=True)
            logger.info("Startup cleanup: removed orphaned BM25 cache %s", Path(pkl).name)

    # Remove upload directories (and their BM25 caches) not accessed in 7 days.
    STALE_DAYS = 7
    cutoff = time.time() - STALE_DAYS * 86400
    for session_dir in UPLOAD_DIR.iterdir():
        if not session_dir.is_dir():
            continue
        marker = session_dir / ".last_accessed"
        mtime = marker.stat().st_mtime if marker.exists() else session_dir.stat().st_mtime
        if mtime < cutoff:
            shutil.rmtree(session_dir, ignore_errors=True)
            bm25 = Path(f"{CACHE_DIR}/bm25_retriever_{session_dir.name}.pkl")
            bm25.unlink(missing_ok=True)
            logger.info("Startup cleanup: removed stale session %s (>%dd old)", session_dir.name, STALE_DAYS)

    # Start background eviction so hot-path request handlers don't do O(n) scans.
    asyncio.create_task(_eviction_loop())


async def _eviction_loop() -> None:
    """Evict stale in-memory sessions every 30 minutes instead of on every request."""
    from app.rag.rag_chain import _evict_stale_sessions as _evict_chat
    from app.services.indexing import _evict_stale_sessions as _evict_index
    while True:
        await asyncio.sleep(1800)
        _evict_chat()
        _evict_index()


@app.get("/thumbnail")
async def file_thumbnail(session_id: str, filename: str):
    """Return a cached JPEG thumbnail for a PDF (first page) or an image file."""
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in session_id)
    file_path = UPLOAD_DIR / safe / Path(filename).name
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    ext = file_path.suffix.lower()

    def _render() -> bytes:
        if ext == ".pdf":
            doc = fitz.open(str(file_path))
            pix = doc[0].get_pixmap(matrix=fitz.Matrix(0.25, 0.25))
            doc.close()
            return pix.tobytes("jpeg")
        if ext in {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}:
            img = Image.open(str(file_path)).convert("RGB")
            img.thumbnail((128, 128))
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=75)
            return buf.getvalue()
        return b""

    try:
        data = await asyncio.to_thread(_render)
    except Exception as exc:
        logger.warning("Thumbnail failed for %s: %s", filename, exc)
        raise HTTPException(status_code=500, detail="Thumbnail generation failed")

    if not data:
        raise HTTPException(status_code=404, detail="No thumbnail for this file type")
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@app.delete("/file")
async def delete_single_file(session_id: str, filename: str):
    """Remove one file from a session's index and delete it from disk."""
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in session_id)
    (UPLOAD_DIR / safe / Path(filename).name).unlink(missing_ok=True)
    try:
        await asyncio.to_thread(remove_file, session_id, filename)
    except Exception as exc:
        logger.exception("Failed to remove %s from session %s", filename, session_id)
        raise HTTPException(status_code=500, detail=f"Removal failed: {exc}")
    return {"ok": True, "indexed_files": get_state(session_id).indexed_files}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/files")
def list_files(session_id: str):
    return {"files": get_state(session_id).indexed_files}


@app.post("/upload", response_model=UploadResponse)
async def upload(session_id: str = Form(...), files: list[UploadFile] = File(...)):
    dest_dir = _session_dir(session_id)
    # Touch the last-accessed marker so the startup cleanup knows this session is active.
    (dest_dir / ".last_accessed").touch()
    saved: list[Path] = []
    for uf in files:
        if not uf.filename:
            continue
        # Strip any directory components a misbehaving client might send.
        dest = dest_dir / Path(uf.filename).name
        with open(dest, "wb") as f:
            f.write(await uf.read())
        saved.append(dest)

    try:
        new_chunks = await asyncio.to_thread(index_files, session_id, saved)
    except Exception as exc:
        logger.exception("Indexing failed")
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}")

    return UploadResponse(
        indexed_files=get_state(session_id).indexed_files, new_chunks=new_chunks
    )


@app.post("/wiki", response_model=UploadResponse)
async def ingest_wiki(req: WikiIngestRequest):
    if not req.wiki_url.strip():
        raise HTTPException(status_code=400, detail="Wiki URL is required.")
    if not req.pat.strip():
        raise HTTPException(status_code=400, detail="Azure DevOps PAT is required.")

    dest_dir = _session_dir(req.session_id)
    (dest_dir / ".last_accessed").touch()

    try:
        new_chunks = await asyncio.to_thread(
            index_wiki_url,
            req.session_id,
            req.wiki_url,
            req.pat,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Wiki indexing failed")
        raise HTTPException(status_code=500, detail=f"Wiki indexing failed: {exc}")

    return UploadResponse(
        indexed_files=get_state(req.session_id).indexed_files,
        new_chunks=new_chunks,
    )


@app.delete("/files")
def clear_files(session_id: str):
    """Remove a single session's uploaded files, index, and cache artifacts."""
    dest_dir = UPLOAD_DIR / "".join(
        ch if ch.isalnum() or ch in "-_" else "_" for ch in session_id
    )
    shutil.rmtree(dest_dir, ignore_errors=True)
    bm25_path = Path(_bm25_cache_file(session_id))
    bm25_path.unlink(missing_ok=True)
    clear_session(session_id)
    try:
        get_session_history(session_id).clear()
    except Exception:
        logger.exception("Failed to clear session history for %s", session_id)
    return {"ok": True}


@app.post("/reset")
def reset(req: ResetRequest):
    get_session_history(req.session_id).clear()
    return {"ok": True}


@app.post("/chat")
async def chat(req: ChatRequest):
    if get_state(req.session_id).chain is None:
        raise HTTPException(
            status_code=400, detail="No documents indexed for this chat yet."
        )
    # Refresh last-accessed so active chat sessions aren't swept by startup cleanup.
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in req.session_id)
    marker = UPLOAD_DIR / safe / ".last_accessed"
    if marker.parent.exists():
        marker.touch()
    return StreamingResponse(
        stream_chat(req.session_id, req.message),
        media_type="text/event-stream",
    )
