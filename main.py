# main.py

import os
from loaders.pdf_loader import load_pdf_text, extract_images_from_pdf
from loaders.image_loader import load_image
from loaders.tabular_loader import load_csv, load_excel
from extractors.image_extractor import extract_text_from_images
from splitter.text_splitter import split_documents
from embeddings.embedder import get_embedding_model
from vectorstore.chroma_store import build_chroma_vectorstore
from retriever.faiss_retriever import build_faiss_retriever
from rag.rag_chain import get_llm, build_rag_chain

# ─────────────────────────────────────────────
# STEP 1: Define your input files
# ─────────────────────────────────────────────
input_files = [
    "data/report.pdf",
    "data/chart.png",
    "data/sales_data.csv",
    "data/financials.xlsx",
]

all_documents = []

# ─────────────────────────────────────────────
# STEP 2: Load & Process Each File
# ─────────────────────────────────────────────
for file_path in input_files:
    ext = os.path.splitext(file_path)[-1].lower()
    print(f"\n📂 Processing: {file_path}")

    if ext == ".pdf":
        # Extract text from PDF
        text_docs = load_pdf_text(file_path)
        all_documents.extend(text_docs)

        # Extract images from PDF → Scout model → text
        images = extract_images_from_pdf(file_path)
        if images:
            image_docs = extract_text_from_images(images, source=file_path)
            all_documents.extend(image_docs)

    elif ext in [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]:
        # Standalone image → Scout model → text
        images = load_image(file_path)
        image_docs = extract_text_from_images(images, source=file_path)
        all_documents.extend(image_docs)

    elif ext == ".csv":
        csv_docs = load_csv(file_path)
        all_documents.extend(csv_docs)

    elif ext in [".xlsx", ".xls"]:
        excel_docs = load_excel(file_path)
        all_documents.extend(excel_docs)

    else:
        print(f"⚠️  Unsupported file type: {ext} — skipping.")

print(f"\n✅ Total documents collected: {len(all_documents)}")

# ─────────────────────────────────────────────
# STEP 3: Split into Chunks
# ─────────────────────────────────────────────
chunks = split_documents(all_documents)

# ─────────────────────────────────────────────
# STEP 4: Load Embedding Model
# ─────────────────────────────────────────────
embedding_model = get_embedding_model()

# ─────────────────────────────────────────────
# STEP 5: Build Chroma Vector Store (persistent)
# ─────────────────────────────────────────────
vectorstore = build_chroma_vectorstore(chunks, embedding_model)

# ─────────────────────────────────────────────
# STEP 6: Build FAISS Retriever (fast retrieval)
# ─────────────────────────────────────────────
retriever = build_faiss_retriever(chunks, embedding_model)

# ─────────────────────────────────────────────
# STEP 7: Initialize LLM & RAG Chain
# ─────────────────────────────────────────────
llm = get_llm()
rag_chain = build_rag_chain(retriever, llm)

# ─────────────────────────────────────────────
# STEP 8: Query the RAG System
# ─────────────────────────────────────────────
print("\n🤖 RAG System Ready! Ask your questions below.")
print("Type 'exit' to quit.\n")

while True:
    query = input("You: ").strip()
    if query.lower() in ["exit", "quit"]:
        print("👋 Goodbye!")
        break
    if not query:
        continue

    response = rag_chain.invoke(query)
    print(f"\n🤖 Assistant: {response}\n")
    print("-" * 60)