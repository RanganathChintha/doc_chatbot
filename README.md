# Doc Chatbot

Conversational RAG over PDFs, images, CSVs and spreadsheets.

## Overview

Doc Chatbot is a Retrieval-Augmented Generation (RAG) application that allows users to chat with their documents. It supports multiple file formats (PDF, images, CSV, spreadsheets) and provides conversational AI with source citations.

Key characteristics:
- **Per-session isolation**: Each conversation has its own isolated document index
- **Hybrid retrieval**: Combines FAISS vector search with BM25 keyword search
- **Streaming responses**: Server-Sent Events (SSE) for real-time token streaming
- **Multi-modal support**: Processes text, tables, and images using vision language models

## Features

- **Multi-format support**: PDF, PNG/JPG images, CSV, Excel spreadsheets
- **Conversational memory**: Context-aware chat with per-session history
- **Source citations**: Responses include references to source document chunks
- **Image understanding**: Vision language model extracts text and descriptions from images
- **Incremental indexing**: New files are added without reindexing existing ones
- **LangSmith tracing**: Optional observability for debugging and monitoring
- **File management**: Upload, list, and delete files per session
- **Thumbnail generation**: Previews for PDFs andimens and images

## Quick Start

1. **Configure environment** — copy `.env.example` to `.env` and set `OPENAI_API_KEY` and `OPENAI_API_BASE`.
2. **Run** — double-click `start.bat` (Windows) or run manually with the commands below.

## Tech Stack

- **Backend**: Python, FastAPI, LangChain (FAISS + BM25 hybrid retriever)
- **Frontend**: React 18, Vite

## Prerequisites

- Python 3.11+
- Node.js 18+

## Manual Setup

### Backend

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate     # On Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.api:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev
```

- Frontend: http://localhost:5173
- Backend: http://localhost:8000

### Production Build

```bash
cd frontend
npm run build
cd ../backend
uvicorn app.api:app --host 0.0.0.0 --port 8000
```

## Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | LLM API key | Yes |
| `OPENAI_API_BASE` | LLM API base URL | Yes |
| `LANGSMITH_API_KEY` | LangSmith API key | No |
| `LANGSMITH_PROJECT_NAME` | LangSmith project name | No |

> See `.env.example` for the full list of optional variables.

## API Endpoints

All API endpoints are relative to `http://localhost:8000`.

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `GET` | `/files?session_id=<uuid>` | List indexed files |
| `POST` | `/upload` | Upload and index files |
| `DELETE` | `/files?session_id=<uuid>` | Clear all session files |
| `DELETE` | `/file?session_id=<uuid>&filename=<name>` | Remove specific file |
| `POST` | `/chat` | Chat (SSE streaming) |
| `POST` | `/reset` | Clear chat history |
| `POST` | `/wiki` | Ingest Azure DevOps wiki |

### Chat Endpoint

`POST /chat` accepts `{ "session_id": "<uuid>", "message": "..." }` and returns a Server-Sent Events stream with token streaming and source citations.

## Architecture

### Data Flow

1. **File Upload**: Files are uploaded → processed (load → split → extract → embed) → stored in FAISS + BM25
2. **Chat Request**: User message → hybrid retriever (semantic + keyword) → LLM generates response with citations
3. **Caching**: Embedding model, LLM responses, and image extractions are cached for performance

### Core Components

- **Backend (FastAPI)**: REST API with file handling, streaming chat, and RAG pipeline
- **RAG Pipeline**: Conversational retrieval chain with session memory and hybrid retrieval
- **Hybrid Retriever**: Combines FAISS (semantic) and BM25 (keyword) results, reranked before passing to LLM
- **Frontend (React + Vite)**: Chat UI with SSE streaming, file management, and session persistence

## Project Structure

```
doc_chatbot/
├── backend/
│   ├── app/
│   │   ├── api.py              # FastAPI routes
│   │   ├── config.py           # Configuration
│   │   ├── pipeline.py         # File processing pipeline
│   │   ├── rag/                # Conversational RAG chain
│   │   ├── retriever/          # Hybrid FAISS + BM25 retrievers
│   │   ├── services/           # Business logic
│   │   ├── loaders/            # PDF, image, tabular loaders
│   │   ├── extractors/         # Image extraction
│   │   └── splitter/           # Text splitting
│   ├── requirements.txt
│   └── pyproject.toml
├── frontend/
│   ├── src/
│   │   ├── main.jsx            # React entry point
│   │   ├── App.jsx             # Top-level state management
│   │   ├── api.js              # HTTP/SSE communication
│   │   └── components/         # UI components
│   ├── package.json
│   └── vite.config.js
├── .env.example
├── start.bat                   # Windows startup script
└── README.md
```

## Troubleshooting

### Backend Fails to Start
- Ensure `OPENAI_API_KEY` and `OPENAI_API_BASE` are set in `..env`
- Check port 8000 isn't already in use
- Validate `.env` syntax (no extra spaces around `=`)

### File Upload Failures
- Check file is supported (PDF, image, CSV, Excel)
- Verify backend has write permissions for `backend/data/` and `backend/cache/`

### Chat Not Working
- Make sure documents are uploaded first — chat requires indexed content
- Check browser devtools network tab for SSE connection errors

### Performance Issues
- First query may be slower due to model loading
- Tune `TOP_K` in `backend/app/config.py` for faster/cheaper queries
- Enable `DEBUG_RETRIEVAL = True` in `config.py` to debug retrieval quality

## License

MIT
