# Backend — Doc Chatbot API

FastAPI + LangChain RAG. Run from this directory so the relative `cache/` and `data/uploads/` paths in [`config.py`](config.py) resolve correctly.

## Run
```bash
pip install -r requirements.txt
uvicorn api:app --reload --port 8000
```

`python-dotenv` walks up from this directory and picks up `../.env` automatically.

## Module map

```
api.py                       FastAPI app + routes
config.py                    env knobs (chunk size, top-k, cache paths)
langsmith_tracing.py         single LangSmith entrypoint
pipeline.py                  load_and_chunk(): files → Documents → chunks

services/
  schemas.py                   Pydantic request/response models
  resources.py                 lazy embedding / LLM singletons
  indexing.py                  IndexState + @traceable ingest_pipeline
  chat.py                      @traceable chat_turn SSE generator

embeddings/embedder.py       cached HF embeddings
extractors/image_extractor.py parallel VLM image extraction + on-disk cache
loaders/{pdf,image,tabular}_loader.py
splitter/text_splitter.py    recursive char splitter
retriever/
  faiss_retriever.py           per-source FAISS + MultiRetriever
  hybrid_retriever.py          FAISS ⊕ BM25 + persisted BM25 cache
rag/rag_chain.py             conversational chain w/ per-session memory

cache/                       FAISS index, BM25 pickle, image cache, LLM cache
data/uploads/                uploaded source files
```

## Endpoints

| Method | Path     | Body                              |
|--------|----------|-----------------------------------|
| GET    | /health  | —                                 |
| GET    | /files   | —                                 |
| POST   | /upload  | multipart `files`                 |
| DELETE | /files   | —                                 |
| POST   | /reset   | `{session_id}`                    |
| POST   | /chat    | `{session_id, message}` (SSE out) |

## LangSmith

Set `LANGSMITH_API_KEY` in `.env`. One trace per request:
- `/upload` → `ingest_pipeline`
- `/chat` → `chat_turn`

with all sub-steps nested under each parent.
