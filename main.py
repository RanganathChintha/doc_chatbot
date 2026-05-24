# main.py

import os

# IMPORTANT: these must be set BEFORE any huggingface_hub / transformers /
# sentence_transformers import, otherwise the offline flags are ignored
# (the libraries cache these env vars at module import time).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import uuid
from loaders.pdf_loader import load_pdf_text, extract_images_from_pdf
from loaders.image_loader import load_image
from loaders.tabular_loader import load_csv, load_excel
from extractors.image_extractor import extract_text_from_images
from splitter.text_splitter import split_documents
from embeddings.embedder import get_embedding_model
from vectorstore.chroma_store import build_chroma_vectorstore, load_chroma_vectorstore
from retriever.faiss_retriever import build_faiss_retriever
from rag.rag_chain import get_llm, build_rag_chain
from vectorstore.manifest import manifest_matches, build_manifest, save_manifest
from config import CHROMA_PERSIST_DIR

# ─────────────────────────────────────────────
# STEP 1: Define your input files
# ─────────────────────────────────────────────
input_files = [
    "data/pdf_img.pdf",
    "data/ocr.png",
    "data/breast-cancer.csv",
    # "data/clothes.jpg",
]


def load_and_chunk(input_files: list[str]):
    """Parse all input files into Documents, then split into chunks."""
    all_documents = []
    for file_path in input_files:
        ext = os.path.splitext(file_path)[-1].lower()
        print(f"\n📂 Processing: {file_path}")

        if ext == ".pdf":
            text_docs = load_pdf_text(file_path)
            all_documents.extend(text_docs)
            images = extract_images_from_pdf(file_path)
            if images:
                image_docs = extract_text_from_images(images, source=file_path)
                all_documents.extend(image_docs)

        elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
            images = load_image(file_path)
            image_docs = extract_text_from_images(images, source=file_path)
            all_documents.extend(image_docs)

        elif ext == ".csv":
            all_documents.extend(load_csv(file_path))

        elif ext in [".xlsx", ".xls"]:
            all_documents.extend(load_excel(file_path))

        else:
            print(f"⚠️  Unsupported file type: {ext} — skipping.")

    print(f"\n✅ Total documents collected: {len(all_documents)}")
    return split_documents(all_documents)


# ─────────────────────────────────────────────
# STEP 2: Load embedding model (cached)
# ─────────────────────────────────────────────
embedding_model = get_embedding_model()

# ─────────────────────────────────────────────
# STEP 3: Vectorstore — reuse if source files unchanged
# ─────────────────────────────────────────────
chunks = None
if manifest_matches(input_files) and os.path.isdir(CHROMA_PERSIST_DIR):
    print("♻️  Source files unchanged — loading vectorstore from disk.")
    vectorstore = load_chroma_vectorstore(embedding_model)
else:
    print("🔨 Source files changed (or first run) — rebuilding vectorstore.")
    chunks = load_and_chunk(input_files)
    vectorstore = build_chroma_vectorstore(chunks, embedding_model)
    save_manifest(build_manifest(input_files))

# ─────────────────────────────────────────────
# STEP 4: Retriever
# Use Chroma directly when we loaded from disk (no chunks in memory);
# otherwise keep your existing FAISS retriever for the freshly built chunks.
# ─────────────────────────────────────────────
if chunks is None:
    from config import TOP_K
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": TOP_K},
    )
    print(f"✅ Chroma retriever ready — will fetch top {TOP_K} chunks.")
else:
    retriever = build_faiss_retriever(chunks, embedding_model)

# ─────────────────────────────────────────────
# STEP 5: LLM + conversational RAG chain
# ─────────────────────────────────────────────
llm = get_llm()
rag_chain = build_rag_chain(retriever, llm)

# ─────────────────────────────────────────────
# STEP 6: Chat loop — one session_id per run
# ─────────────────────────────────────────────
session_id = str(uuid.uuid4())
print(f"\n🤖 RAG System Ready! (session {session_id[:8]}) Ask your questions below.")
print("Type 'exit' to quit, 'reset' to clear chat history.\n")

while True:
    try:
        query = input("You: ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n👋 Goodbye!")
        break

    if query.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break
    if query.lower() == "reset":
        session_id = str(uuid.uuid4())
        print(f"🧹 Chat history cleared. New session: {session_id[:8]}\n")
        continue
    if not query:
        continue

    response = rag_chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}},
    )
    print(f"\n🤖 Assistant: {response['answer']}\n")
    print("-" * 60)
