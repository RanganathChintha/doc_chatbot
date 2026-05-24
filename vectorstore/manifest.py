"""Source-file fingerprint manifest.

Tracks (path, size, mtime, sha256) of input files so we can decide whether
the persisted Chroma vectorstore is still valid or must be rebuilt.
"""
from __future__ import annotations

import hashlib
import json
import os
from config import MANIFEST_FILE, CACHE_DIR


def _file_fingerprint(path: str) -> dict:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    st = os.stat(path)
    return {"size": st.st_size, "mtime": st.st_mtime, "sha256": h.hexdigest()}


def build_manifest(paths: list[str]) -> dict:
    return {p: _file_fingerprint(p) for p in paths if os.path.exists(p)}


def load_manifest() -> dict | None:
    if not os.path.exists(MANIFEST_FILE):
        return None
    with open(MANIFEST_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: dict) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def manifest_matches(paths: list[str]) -> bool:
    existing = load_manifest()
    if existing is None:
        return False
    current = build_manifest(paths)
    if set(existing.keys()) != set(current.keys()):
        return False
    for path, fp in current.items():
        old = existing.get(path)
        if not old or old.get("sha256") != fp["sha256"]:
            return False
    return True
