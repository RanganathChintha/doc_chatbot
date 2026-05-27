# app.py — Streamlit interface for the RAG chatbot.
#
# Run with:  streamlit run app.py
#
# Design notes:
#   - HF offline flags MUST be set before any HF library is imported (same
#     constraint as main.py).
#   - Expensive resources (embedding model, LLM) are wrapped with
#     @st.cache_resource so Streamlit's script-rerun model doesn't re-load
#     them on every interaction.
#   - The retriever/chain are kept in st.session_state and rebuilt only when
#     the set of uploaded files actually changes.
#   - Chat history uses our existing per-session_id store inside rag_chain,
#     so memory + history-aware retrieval work exactly as in the CLI.

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("LANGSMITH_TRACING_V2", "true")

import hashlib
import uuid
from pathlib import Path

import streamlit as st
from langsmith_tracing import langsmith_traceable as traceable

from pipeline import load_and_chunk
from embeddings.embedder import get_embedding_model
from retriever.faiss_retriever import build_per_source_retrievers, MultiRetriever
from retriever.hybrid_retriever import build_hybrid_retriever
from vectorstore.chroma_store import clear_chroma_vectorstore, persist_chunks_to_chroma
from rag.rag_chain import get_llm, build_rag_chain
from config import TOP_K

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

st.set_page_config(page_title="Doc Chatbot", page_icon="🤖", layout="wide")


# ─────────────────────────────────────────────
# Cached heavy resources (loaded once per session)
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading embedding model…")
def cached_embedding_model():
    return get_embedding_model()


@st.cache_resource(show_spinner="Connecting to LLM…")
def cached_llm():
    return get_llm()


def _fingerprint(paths: list[Path]) -> str:
    """Stable hash over the uploaded files (path + size + first 4KB)."""
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p).encode())
        h.update(str(p.stat().st_size).encode())
        with open(p, "rb") as f:
            h.update(f.read(4096))
    return h.hexdigest()


@traceable(run_type="tool", name="build_chain_for_files")
def build_chain_for_files(file_paths: list[Path]):
    """Build a global persistent FAISS retriever + hybrid search + the conversational chain."""
    # Support incremental indexing: if a chain already exists in session_state,
    # only chunk and index the newly uploaded files and merge them into the
    # existing retrievers and BM25 index. This avoids rebuilding the entire
    # index on every upload.

    # Resolve file paths and names
    all_paths = [Path(p) for p in file_paths]
    all_names = [p.name for p in all_paths]

    # If there is no existing chain, build everything from scratch
    if not st.session_state.get("chain") or not st.session_state.get("indexed_files"):
        chunks = load_and_chunk([str(p) for p in all_paths], verbose=False)
        if not chunks:
            return None, 0
        embedding_model = cached_embedding_model()
        persist_chunks_to_chroma(
            chunks,
            embedding_model,
            source_files=[str(p) for p in all_paths],
        )

        # Build per-source retrievers (one FAISS per file) and a MultiRetriever
        retrievers_dict, by_source = build_per_source_retrievers(chunks, embedding_model, k_per_source=TOP_K)
        semantic_multi = MultiRetriever(retrievers_dict)

        # Hybrid retriever will build its own BM25 index from `chunks`
        hybrid_retriever = build_hybrid_retriever(
            semantic_retriever=semantic_multi,
            chunks=chunks,
            k=10,
            use_bm25=True,
        )

        source_names = sorted({doc.metadata.get("source", "") for doc in chunks if doc.metadata.get("source")})
        llm = cached_llm()
        chain = build_rag_chain(hybrid_retriever, source_names, llm)

        # Persist pieces in session_state for incremental updates
        st.session_state.chain = chain
        st.session_state.retriever = hybrid_retriever
        st.session_state.semantic_retriever = semantic_multi
        st.session_state.by_source = by_source
        st.session_state.indexed_files = all_names

        return chain, len(chunks)

    # Existing chain present: compute which files are new
    existing_names: list[str] = st.session_state.get("indexed_files", [])
    new_paths = [p for p in all_paths if p.name not in existing_names]
    if not new_paths:
        # Nothing new to index; return existing chain and zero new chunks
        return st.session_state.chain, 0

    # Chunk only the new files
    new_chunks = load_and_chunk([str(p) for p in new_paths], verbose=False)
    if not new_chunks:
        return st.session_state.chain, 0

    embedding_model = cached_embedding_model()
    persist_chunks_to_chroma(
        new_chunks,
        embedding_model,
        source_files=[str(p) for p in new_paths],
    )

    # Build per-source retrievers for the new chunks and merge into existing semantic retriever
    new_retrievers, new_by_source = build_per_source_retrievers(new_chunks, embedding_model, k_per_source=TOP_K)
    semantic_multi: MultiRetriever = st.session_state.semantic_retriever
    if hasattr(semantic_multi, "add_retrievers"):
        semantic_multi.add_retrievers(new_retrievers)
    else:
        # Fallback: if the existing semantic retriever cannot merge, rebuild full chain
        return build_chain_for_files(file_paths)

    # Merge by_source mapping for bookkeeping and BM25 rebuild
    by_source: dict = st.session_state.get("by_source", {})
    by_source.update(new_by_source)
    st.session_state.by_source = by_source

    # Rebuild BM25 retriever from all known chunks (safe and simple)
    all_chunks = []
    for docs in by_source.values():
        all_chunks.extend(docs)

    # Recreate hybrid retriever using the updated semantic retriever and full chunk list
    hybrid_retriever = build_hybrid_retriever(
        semantic_retriever=semantic_multi,
        chunks=all_chunks,
        k=10,
        use_bm25=True,
    )

    # Rebuild the RAG chain with the updated retriever
    source_names = sorted({doc.metadata.get("source", "") for doc in all_chunks if doc.metadata.get("source")})
    llm = cached_llm()
    chain = build_rag_chain(hybrid_retriever, source_names, llm)

    # Update session state
    st.session_state.chain = chain
    st.session_state.retriever = hybrid_retriever
    st.session_state.semantic_retriever = semantic_multi
    st.session_state.indexed_files = existing_names + [p.name for p in new_paths]

    return chain, len(new_chunks)


@traceable(run_type="tool", name="invoke_rag_chain")
def invoke_rag_chain(chain, query, session_id):
    return chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}},
    )


# ─────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role", "content", "sources"}
if "chain" not in st.session_state:
    st.session_state.chain = None
if "files_fingerprint" not in st.session_state:
    st.session_state.files_fingerprint = None
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


# ─────────────────────────────────────────────
# Sidebar — uploads + controls
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("📂 Documents")
    uploads = st.file_uploader(
        "Upload PDF / image / CSV / Excel files",
        type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff", "csv", "xlsx", "xls"],
        accept_multiple_files=True,
    )

    # Persist uploads to disk so the loaders (which take file paths) can read them.
    saved_paths: list[Path] = []
    if uploads:
        for uf in uploads:
            dest = UPLOAD_DIR / uf.name
            with open(dest, "wb") as f:
                f.write(uf.getbuffer())
            saved_paths.append(dest)

    fp = _fingerprint(saved_paths) if saved_paths else None

    # Rebuild the chain only when the uploaded set changed.
    if fp and fp != st.session_state.files_fingerprint:
        with st.spinner(f"Indexing {len(saved_paths)} file(s)…"):
            try:
                chain, n_chunks = build_chain_for_files(saved_paths)
            except Exception as e:
                st.error(f"Failed to index: {e}")
                chain, n_chunks = None, 0
        if chain is not None:
            st.session_state.chain = chain
            st.session_state.files_fingerprint = fp
            st.session_state.indexed_files = [p.name for p in saved_paths]
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.success(f"Indexed {n_chunks} chunk(s) across {len(saved_paths)} file(s).")

    if st.session_state.indexed_files:
        st.markdown("**Currently indexed:**")
        for name in st.session_state.indexed_files:
            st.markdown(f"- `{name}`")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🧹 Clear chat"):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🗑️ Clear files"):
            # Wipe uploaded files and reset the chain.
            for p in UPLOAD_DIR.glob("*"):
                try:
                    p.unlink()
                except OSError:
                    pass
            st.session_state.chain = None
            st.session_state.files_fingerprint = None
            st.session_state.indexed_files = []
            try:
                clear_chroma_vectorstore(cached_embedding_model())
            except Exception:
                pass
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.rerun()

    st.caption(f"Session: `{st.session_state.session_id[:8]}`")


# ─────────────────────────────────────────────
# Main pane — chat
# ─────────────────────────────────────────────
st.title("🤖 Doc Chatbot")

if st.session_state.chain is None:
    st.info("👈 Upload one or more documents in the sidebar to start chatting.")
else:
    # Replay history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📚 Sources used ({len(msg['sources'])})"):
                    for i, doc in enumerate(msg["sources"], 1):
                        src = doc.metadata.get("source", "unknown")
                        stype = doc.metadata.get("source_type", "?")
                        page = doc.metadata.get("page")
                        header = f"**[{i}] {stype.upper()}** — `{src}`"
                        if page is not None:
                            header += f" (page {page})"
                        st.markdown(header)
                        st.markdown(f"> {doc.page_content[:500]}"
                                    + ("…" if len(doc.page_content) > 500 else ""))

    # New input
    query = st.chat_input("Ask a question about your documents…")
    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("_Thinking…_")
            try:
                response = invoke_rag_chain(
                    st.session_state.chain,
                    query,
                    st.session_state.session_id,
                )
                answer = response["answer"]
                sources = response.get("docs", [])
            except Exception as e:
                answer = f"⚠️ Error: {e}"
                sources = []
            placeholder.markdown(answer)
            if sources:
                with st.expander(f"📚 Sources used ({len(sources)})"):
                    for i, doc in enumerate(sources, 1):
                        src = doc.metadata.get("source", "unknown")
                        stype = doc.metadata.get("source_type", "?")
                        page = doc.metadata.get("page")
                        header = f"**[{i}] {stype.upper()}** — `{src}`"
                        if page is not None:
                            header += f" (page {page})"
                        st.markdown(header)
                        st.markdown(f"> {doc.page_content[:500]}"
                                    + ("…" if len(doc.page_content) > 500 else ""))

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })
