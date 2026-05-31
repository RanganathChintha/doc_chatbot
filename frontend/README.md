# Frontend — Doc Chatbot UI

Vite + React + plain CSS. Streams `/api/chat` SSE responses from [`../backend/api.py`](../backend/api.py).

## Run (dev)
```bash
npm install
npm run dev
```
Open http://localhost:5173. Vite proxies `/api/*` → `http://localhost:8000` (configured in [`vite.config.js`](vite.config.js)).

The backend must be running separately:
```bash
cd ../backend
uvicorn app.api:app --reload --port 8000
```

## Build (production)
```bash
npm run build
```
Outputs static assets to `dist/`. Serve them from any static host; configure your reverse proxy so `/api/*` → deployed FastAPI.

## Source map

```
src/
├─ main.jsx                React entry
├─ App.jsx                 Top-level state (conversations, streaming, errors)
├─ api.js                  fetch wrapper + SSE parser (chatStream generator)
├─ styles.css              Theme (CHAT A.I+ palette)
└─ components/
   ├─ Sidebar.jsx            New chat, search, conversation groups, doc upload
   ├─ ChatPane.jsx           Message list + empty-state hero
   ├─ Message.jsx            Markdown rendering + collapsible source citations
   └─ InputBox.jsx           Auto-grow textarea, Enter to send, Stop button
```

## State model

- **Conversations** (UI history) live in `localStorage` under key `doc_chatbot.conversations.v1`.
- Each conversation has its own `session_id` (UUID) — the backend keeps per-session chat history.
- **Documents** (the indexed corpus) are server-side and shared across all conversations.

## Backend contract

See [`../README.md`](../README.md#api) for the full endpoint table. Key shapes:

- `POST /chat` request: `{ session_id, message }`
- `POST /chat` SSE frames: `{type: 'session'|'token'|'sources'|'done'|'error', ...}`
