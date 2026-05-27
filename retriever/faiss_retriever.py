# retriever/faiss_retriever.py

import json
import os
from typing import Any

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from config import CACHE_DIR, MANIFEST_FILE, TOP_K
from langsmith_tracing import langsmith_traceable as traceable


def _build_manifest(file_paths: list[str]) -> dict[str, dict[str, Any]]:
    manifest: dict[str, dict[str, Any]] = {}
    for path in sorted({p for p in file_paths if p}):
        abs_path = os.path.abspath(path)
        try:
            stat = os.stat(abs_path)
        except FileNotFoundError:
            continue
        manifest[abs_path] = {
            "size": stat.st_size,
            "mtime": stat.st_mtime,
        }
    return manifest


def _load_manifest(path: str) -> dict[str, dict[str, Any]]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _save_manifest(manifest: dict[str, dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _manifest_matches(file_paths: list[str], manifest_path: str) -> bool:
    existing_manifest = _load_manifest(manifest_path)
    if not existing_manifest:
        return False
    return _build_manifest(file_paths) == existing_manifest


def _faiss_cache_path() -> str:
    path = os.path.join(CACHE_DIR, "faiss_index")
    os.makedirs(path, exist_ok=True)
    return path


def build_faiss_retriever(
    chunks: list[Document],
    embedding_model,
    source_files: list[str] | None = None,
):
    """
    Build or load a persistent global FAISS index.

    Uses a manifest of the source files to avoid rebuilding the index when
    the underlying documents have not changed.
    """
    persist_dir = _faiss_cache_path()
    if source_files:
        files_to_check = source_files
    else:
        files_to_check = [doc.metadata.get("source", "") for doc in chunks if doc.metadata.get("source")]

    if _manifest_matches(files_to_check, MANIFEST_FILE) and os.path.isdir(persist_dir):
        try:
            store = FAISS.load_local(persist_dir, embedding_model)
            print("Loaded persistent FAISS index from disk.")
            return store
        except Exception:
            print("Failed to load existing FAISS index, rebuilding.")

    store = FAISS.from_documents(documents=chunks, embedding=embedding_model)
    store.save_local(persist_dir)
    _save_manifest(_build_manifest(files_to_check), MANIFEST_FILE)
    print(f"Built new FAISS index and persisted it to {persist_dir}.")
    return store


@traceable(run_type="tool", name="build_per_source_retrievers")
def build_per_source_retrievers(
    chunks: list[Document],
    embedding_model,
    k_per_source: int = 3,
) -> tuple[dict, dict]:
    """
    Build one FAISS retriever per source file so retrieval is guaranteed to
    cover every uploaded file (and not just the globally most-similar chunks).

    Returns:
        retrievers: {source_path -> Retriever} — for query-time per-file similarity
        by_source: {source_path -> list[Document]} — all chunks per file, used
                   when a query asks for full per-file context (e.g. summaries)
    """
    from collections import defaultdict

    by_source: dict[str, list[Document]] = defaultdict(list)
    for c in chunks:
        src = c.metadata.get("source", "unknown")
        by_source[src].append(c)

    retrievers: dict = {}
    for source, source_chunks in by_source.items():
        k = min(k_per_source, len(source_chunks))
        if k == 0:
            continue
        store = FAISS.from_documents(source_chunks, embedding_model)
        retrievers[source] = store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    print(
        f"Per-file retrievers ready: {len(retrievers)} file(s), "
        f"top-{k_per_source} per file."
    )
    return retrievers, dict(by_source)


class MultiRetriever:
    """Combine many per-source retrievers into a single semantic retriever.

    Exposes `similarity_search_with_score(query, k=...)` and a helper
    `add_retrievers()` for incremental merging.
    """

    def __init__(self, retrievers: dict[str, Any] | None = None):
        self.retrievers: dict[str, Any] = retrievers or {}

    def add_retrievers(self, new: dict[str, Any]) -> None:
        self.retrievers.update(new)

    def similarity_search_with_score(self, query: str, k: int | None = None):
        results: list[tuple[Document, float]] = []
        per_k = k or TOP_K
        for src, r in self.retrievers.items():
            try:
                res = r.similarity_search_with_score(query, k=per_k)
                results.extend(res)
            except Exception:
                try:
                    docs = r.invoke(query)
                    results.extend([(d, 0.5) for d in docs])
                except Exception:
                    continue

        # Deduplicate by (source,page,content-hash) preserving highest score
        seen = set()
        unique: list[tuple[Document, float]] = []
        for doc, score in sorted(results, key=lambda x: x[1], reverse=True):
            key = (
                doc.metadata.get("source", ""),
                doc.metadata.get("page", ""),
                hash(doc.page_content) % 100000,
            )
            if key in seen:
                continue
            seen.add(key)
            unique.append((doc, score))

        return unique[: per_k * 2]
