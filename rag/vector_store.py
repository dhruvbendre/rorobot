"""
A small numpy vector store.

The archive is a few hundred chunks at most, so an in-memory matrix with a
dot-product search is faster, simpler and more portable than a vector
database. The index persists to `INDEX_DIR` as three files (vectors, chunks,
manifest) and is rebuilt automatically when the knowledge, the embedding
provider or the model changes.

Swapping in FAISS or Chroma later only means re-implementing `search`.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .chunker import Chunk

VECTORS_FILE = "vectors.npy"
CHUNKS_FILE = "chunks.json"
MANIFEST_FILE = "manifest.json"


class VectorStore:
    def __init__(self, vectors: np.ndarray, chunks: list[Chunk], manifest: dict):
        assert vectors.shape[0] == len(chunks), "vectors and chunks are out of step"
        self.vectors = vectors.astype(np.float32)
        self.chunks = chunks
        self.manifest = manifest

    def __len__(self) -> int:
        return len(self.chunks)

    # --- search -------------------------------------------------------------

    def search(self, query_vector: np.ndarray, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if len(self.chunks) == 0:
            return []
        scores = self.vectors @ query_vector.astype(np.float32)
        order = np.argsort(-scores)[:top_k]
        return [(self.chunks[i], float(scores[i])) for i in order]

    # --- persistence --------------------------------------------------------

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / VECTORS_FILE, self.vectors)
        (directory / CHUNKS_FILE).write_text(
            json.dumps([asdict(c) for c in self.chunks], ensure_ascii=False, indent=0), encoding="utf-8"
        )
        (directory / MANIFEST_FILE).write_text(json.dumps(self.manifest, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "VectorStore | None":
        try:
            manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
            raw_chunks = json.loads((directory / CHUNKS_FILE).read_text(encoding="utf-8"))
            vectors = np.load(directory / VECTORS_FILE)
        except (OSError, ValueError, json.JSONDecodeError):
            return None
        chunks = [Chunk(**c) for c in raw_chunks]
        if vectors.shape[0] != len(chunks):
            return None
        return cls(vectors, chunks, manifest)

    @staticmethod
    def manifest_matches(directory: Path, expected: dict) -> bool:
        try:
            manifest = json.loads((directory / MANIFEST_FILE).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return all(manifest.get(k) == v for k, v in expected.items())
