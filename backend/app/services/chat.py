"""SSE streaming for chat turns.

Wraps the conversational RAG chain so the FastAPI route can simply
`async for event in stream_chat(...)` and pipe events as SSE frames.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncIterator

from app.langsmith_tracing import langsmith_traceable as traceable

from .indexing import get_state

logger = logging.getLogger(__name__)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _format_docs(docs: list) -> list[dict]:
    result = []
    for d in docs:
        meta = d.metadata
        url = (meta.get("remote_url") or "").strip()
        source_type = meta.get("source_type", "")
        if not url and source_type == "wiki":
            logger.debug("Wiki chunk missing remote_url: source=%s title=%s", meta.get("source"), meta.get("title"))
        result.append({
            "source": meta.get("source", ""),
            "source_type": source_type,
            "title": meta.get("title", ""),
            "url": url,
            "page": meta.get("page"),
            "snippet": d.page_content[:300],
        })
    return result


async def stream_chat(session_id: str, message: str) -> AsyncIterator[str]:
    """Yield SSE-formatted frames for one chat turn.

    Frames:
        session — echoes session_id once at the start
        token   — incremental answer text
        sources — list of retrieved docs (one frame near the end of the stream)
        done    — terminator
        error   — on failure (terminator too)
    """
    chain = get_state(session_id).chain
    if chain is None:
        yield _sse({"type": "error", "message": "No documents indexed yet."})
        yield _sse({"type": "done"})
        return

    yield _sse({"type": "session", "session_id": session_id})

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue(maxsize=256)

    def emit(item):
        loop.call_soon_threadsafe(queue.put_nowait, item)

    @traceable(run_type="chain", name="chat_turn")
    def _run_chain():
        """Single LangSmith parent run — all downstream traceables nest under it."""
        for chunk in chain.stream(
            {"input": message},
            config={"configurable": {"session_id": session_id}},
        ):
            emit(("chunk", chunk))

    def _user_facing_error(exc: Exception) -> str:
        """Return a clean, safe message for the frontend — no internal stack details."""
        msg = str(exc)
        if "429" in msg or "rate_limit" in msg.lower() or "rate limit" in msg.lower():
            return (
                "Service rate limit reached. The model is receiving too many requests. "
                "Please wait a moment and try again."
            )
        if "401" in msg or "authentication" in msg.lower():
            return "API authentication failed. Check your OPENAI_API_KEY (Siemens key)."
        if "503" in msg or "unavailable" in msg.lower():
            return "The AI service is temporarily unavailable. Please try again shortly."
        return "Something went wrong while generating a response. Please try again."

    def producer():
        try:
            _run_chain()
        except Exception as exc:
            logger.error("Chain stream failed: %s", exc)
            emit(("error", _user_facing_error(exc)))
        finally:
            emit(("done", None))

    loop.run_in_executor(None, producer)

    while True:
        kind, payload = await queue.get()
        if kind == "done":
            yield _sse({"type": "done"})
            return
        if kind == "error":
            yield _sse({"type": "error", "message": payload})
            yield _sse({"type": "done"})
            return
        # kind == "chunk"
        if "answer" in payload:
            yield _sse({"type": "token", "text": payload["answer"]})
        if "docs" in payload:
            yield _sse({"type": "sources", "sources": _format_docs(payload["docs"])})
