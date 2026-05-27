# main.py

import os

# IMPORTANT: these must be set BEFORE any huggingface_hub / transformers /
# sentence_transformers import, otherwise the offline flags are ignored
# (the libraries cache these env vars at module import time).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("LANGSMITH_TRACING_V2", "true")

import uuid
from pipeline import load_and_chunk
from embeddings.embedder import get_embedding_model
from retriever.faiss_retriever import build_faiss_retriever
from retriever.hybrid_retriever import build_hybrid_retriever
from vectorstore.chroma_store import persist_chunks_to_chroma
from rag.rag_chain import get_llm, build_rag_chain
from langsmith_tracing import langsmith_traceable as traceable

# ─────────────────────────────────────────────
# STEP 1: Define your input files
# ─────────────────────────────────────────────
input_files = [
    "data/python.pdf",
    # "data/ocr.png",
    # "data/breast-cancer.csv",
    # "data/clothes.jpg",
]

# ─────────────────────────────────────────────
# STEP 2: Load, chunk, embed, and build a persistent global FAISS index
# (image + embedding caches make repeat runs fast)
# ─────────────────────────────────────────────
embedding_model = get_embedding_model()
chunks = load_and_chunk(input_files)
persist_chunks_to_chroma(chunks, embedding_model, source_files=input_files)

# Build FAISS retriever
retriever = build_faiss_retriever(chunks, embedding_model, source_files=input_files)

# Wrap in hybrid retriever for enhanced retrieval (semantic + keyword search)
retriever = build_hybrid_retriever(
    semantic_retriever=retriever,
    chunks=chunks,
    k=10,  # Retrieve top 10 candidates
    use_bm25=True,  # Use keyword search alongside semantic
)

source_names = sorted({doc.metadata.get("source", "") for doc in chunks if doc.metadata.get("source")})

# ─────────────────────────────────────────────
# STEP 3: LLM + conversational RAG chain
# ─────────────────────────────────────────────
llm = get_llm()
rag_chain = build_rag_chain(retriever, source_names, llm)

# ─────────────────────────────────────────────
# STEP 4: Chat loop — one session_id per run
# ─────────────────────────────────────────────
session_id = str(uuid.uuid4())
print(f"\nRAG System Ready! (session {session_id[:8]}) Ask your questions below.")
print("Type 'exit' to quit, 'reset' to clear chat history.\n")

@traceable(run_type="tool", name="invoke_rag_chain")
def invoke_rag_chain(chain, query, session_id):
    return chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}},
    )


while True:
    try:
        query = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        break

    if query.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break
    if query.lower() == "reset":
        session_id = str(uuid.uuid4())
        print(f"Chat history cleared. New session: {session_id[:8]}\n")
        continue
    if not query:
        continue

    response = invoke_rag_chain(rag_chain, query, session_id)
    print(f"\nAssistant: {response['answer']}\n")
    print("-" * 60)
