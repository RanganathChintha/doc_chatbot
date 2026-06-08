"""Pydantic schemas exchanged with the frontend."""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ResetRequest(BaseModel):
    session_id: str


class UrlCrawlRequest(BaseModel):
    session_id: str
    urls: list[str]
    allow_domains: list[str] | None = None
    max_depth: int | None = None
    max_pages: int | None = None
    auth_cookies: dict[str, str] | None = None
    headers: dict[str, str] | None = None
    render_javascript: bool = False
    render_timeout: int = 30


class UploadResponse(BaseModel):
    indexed_files: list[str]
    new_chunks: int
