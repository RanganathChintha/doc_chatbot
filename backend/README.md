# Backend — Doc Chatbot API

FastAPI + LangChain RAG. The code lives in the importable `app` package. Run from
this directory so the relative `cache/` and `data/uploads/` paths in
[`config.py`](app/config.py) resolve correctly.

## Run
```bash
pip install -r requirements.txt   # or: pip install -e .  (uses pyproject.toml)
uvicorn app.api:app --reload --port 8000
```

`python-dotenv` walks up from this directory and picks up `../.env` automatically.

## Per-chat isolation

Each chat (the frontend conversation `id`, sent as `session_id`) owns an isolated
index and upload folder. Uploaded files, the FAISS/BM25 retrievers, and the RAG
chain are all keyed by `session_id`, so one chat can never retrieve another
chat's documents. Uploads land in `data/uploads/<session_id>/`.

## Module map

```
app/
  api.py                       FastAPI app + routes
  config.py                    env knobs (chunk size, top-k, cache paths)
  langsmith_tracing.py         single LangSmith entrypoint
  pipeline.py                  load_and_chunk(): files → Documents → chunks

  services/
    schemas.py                   Pydantic request/response models
    resources.py                 lazy embedding / LLM singletons
    indexing.py                  per-session IndexState registry + ingest_pipeline
    chat.py                      @traceable chat_turn SSE generator

  embeddings/embedder.py       cached HF embeddings
  extractors/image_extractor.py parallel VLM image extraction + on-disk cache
  loaders/{pdf,image,tabular}_loader.py
  splitter/text_splitter.py    recursive char splitter
  retriever/
    faiss_retriever.py           per-source FAISS + MultiRetriever
    hybrid_retriever.py          FAISS ⊕ BM25 + per-session BM25 cache
  rag/rag_chain.py             conversational chain w/ per-session memory

pyproject.toml               package metadata (deps read from requirements.txt)
cache/                       BM25 pickles, image cache, LLM cache
data/uploads/<session_id>/   uploaded source files, isolated per chat
```

## Endpoints

| Method | Path     | Body / Query                          |
|--------|----------|---------------------------------------|
| GET    | /health  | —                                     |
| GET    | /files   | `?session_id=…`                       |
| POST   | /upload  | multipart `session_id` + `files`      |
| DELETE | /files   | `?session_id=…` (clears that chat)    |
| POST   | /reset   | `{session_id}`                        |
| POST   | /chat    | `{session_id, message}` (SSE out)     |

## LangSmith

Set `LANGSMITH_API_KEY` in `.env`. One trace per request:
- `/upload` → `ingest_pipeline`
- `/chat` → `chat_turn`

with all sub-steps nested under each parent.
