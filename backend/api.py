# api.py — FastAPI routes. Business logic lives in services/.
#
# Run with:  uvicorn api:app --reload --port 8000

import os

# HF offline flags must be set before any HF library is imported transitively.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import asyncio
import logging
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from rag.rag_chain import get_session_history
from services.chat import stream_chat
from services.indexing import clear_session, get_state, index_files
from services.schemas import ChatRequest, ResetRequest, UploadResponse

logger = logging.getLogger(__name__)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


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
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/files")
def list_files(session_id: str):
    return {"files": get_state(session_id).indexed_files}


@app.post("/upload", response_model=UploadResponse)
async def upload(session_id: str = Form(...), files: list[UploadFile] = File(...)):
    dest_dir = _session_dir(session_id)
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


@app.delete("/files")
def clear_files(session_id: str):
    """Remove a single session's uploaded files and drop its index."""
    dest_dir = UPLOAD_DIR / "".join(
        ch if ch.isalnum() or ch in "-_" else "_" for ch in session_id
    )
    shutil.rmtree(dest_dir, ignore_errors=True)
    clear_session(session_id)
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
    return StreamingResponse(
        stream_chat(req.session_id, req.message),
        media_type="text/event-stream",
    )
