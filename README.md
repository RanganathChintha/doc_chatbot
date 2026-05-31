# Doc Chatbot

Conversational RAG over PDFs, images, CSVs and spreadsheets.
- **Backend:** FastAPI + LangChain RAG (FAISS per-source + BM25 hybrid retrieval).
- **Frontend:** Vite + React with streaming SSE.
- **Models:** Groq (`openai/gpt-oss-120b` chat, `llama-4-scout` VLM) + local `all-MiniLM-L6-v2` embeddings.
- **Tracing:** LangSmith (optional).

## Layout

```
doc_chatbot-main/
├─ backend/          FastAPI app + RAG pipeline + retrievers
│   ├─ api.py                routes
│   ├─ config.py             knobs
│   ├─ langsmith_tracing.py  single LangSmith entrypoint
│   ├─ pipeline.py           file → Documents → chunks
│   ├─ requirements.txt
│   ├─ services/             business logic
│   │   ├─ schemas.py          Pydantic models
│   │   ├─ resources.py        lazy embedding / LLM singletons
│   │   ├─ indexing.py         IndexState + ingest pipeline
│   │   └─ chat.py             SSE streaming
│   ├─ embeddings/           HF embeddings with disk cache
│   ├─ extractors/           parallel VLM image extraction
│   ├─ loaders/              pdf / image / tabular
│   ├─ splitter/             recursive char splitter
│   ├─ retriever/            FAISS per-source + hybrid
│   ├─ rag/                  conversational chain
│   ├─ cache/                generated: FAISS, BM25, image cache, LLM cache
│   └─ data/uploads/         uploaded source files
├─ frontend/         Vite + React UI
│   └─ src/{App,components,api,styles}
├─ .env              shared by backend (loaded via python-dotenv walk-up)
└─ README.md         (this file)
```

## Setup

```bash
# Python deps
python -m venv myvenv
myvenv\Scripts\activate                 # Windows
pip install -r backend/requirements.txt

# Node deps
cd frontend
npm install
cd ..
```

Create a `.env` in the repo root (see `.env.example`):
```
GROQ_API_KEY=...
LANGSMITH_API_KEY=...              # optional
LANGSMITH_PROJECT_NAME=doc_chatbot # optional
```

## Run (dev — two terminals)

**Terminal 1 — backend** (from `backend/` so relative cache/data paths resolve correctly):
```bash
cd backend
uvicorn app.api:app --reload --port 8000
```

**Terminal 2 — frontend**:
```bash
cd frontend
npm run dev
```

Open http://localhost:5173. Vite proxies `/api/*` → `http://localhost:8000`.

## API

| Method | Path     | Body                              | Notes                                   |
|--------|----------|-----------------------------------|-----------------------------------------|
| GET    | /health  | —                                 | `{ok, indexed}`                         |
| GET    | /files   | —                                 | Currently-indexed filenames             |
| POST   | /upload  | multipart `files`                 | Incrementally indexes new files         |
| DELETE | /files   | —                                 | Wipes uploads + retriever state         |
| POST   | /reset   | `{session_id}`                    | Clears chat history for that session    |
| POST   | /chat    | `{session_id, message}`           | SSE: `session` → `token*` → `sources?` → `done` |

## How frontend & backend connect

- **HTTP:** frontend calls `/api/*` → Vite dev proxy → `http://localhost:8000`.
- **Streaming:** `/chat` returns Server-Sent Events; the React client (`frontend/src/api.js`) reads frames with `fetch` + `ReadableStream` and yields `{type, …}` objects to the React state.
- **Sessions:** each UI conversation has its own UUID (`session_id`), persisted in `localStorage`. The backend keeps per-`session_id` chat history in memory (`rag/rag_chain.py: get_session_history`).
- **Corpus:** documents are shared across all conversations (single global index in `services/indexing.py: state`).

## LangSmith tracing

Set `LANGSMITH_API_KEY` and uploads/chats each produce one parent trace
(`ingest_pipeline`, `chat_turn`) with all sub-steps nested under it.
