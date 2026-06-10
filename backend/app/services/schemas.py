"""Pydantic schemas exchanged with the frontend."""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ResetRequest(BaseModel):
    session_id: str


class WikiIngestRequest(BaseModel):
    session_id: str
    wiki_url: str
    pat: str


class UploadResponse(BaseModel):
    indexed_files: list[str]
    new_chunks: int
