"""
Embedding providers.

    OpenAIEmbeddings   text-embedding-3-* through the OpenAI SDK (or any
                       OpenAI-compatible endpoint via OPENAI_BASE_URL)
    LocalEmbeddings    a dependency-free hashed bag-of-words + bigram vector.
                       Not a semantic model; good enough to exercise retrieval,
                       thresholds and the refusal path without any API key.

Both return L2-normalised float32 arrays so cosine similarity is a dot product.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol

import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is", "are", "was", "were",
    "be", "been", "it", "its", "this", "that", "these", "those", "as", "at", "by", "from", "he", "his",
    "him", "she", "her", "they", "them", "their", "what", "which", "who", "whom", "does", "do", "did",
    "has", "have", "had", "about", "tell", "me", "s", "not", "yet", "any", "some", "into", "over",
}


class EmbeddingProvider(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


def _normalise(matrix: np.ndarray) -> np.ndarray:
    matrix = matrix.astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


class LocalEmbeddings:
    """Deterministic hashed features: unigrams + bigrams, sublinear tf, L2-normalised."""

    def __init__(self, dim: int = 512):
        self.name = f"local-hash-{dim}"
        self.dim = dim

    def _slot(self, feature: str) -> tuple[int, float]:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        index = int.from_bytes(digest[:4], "little") % self.dim
        sign = 1.0 if digest[4] & 1 else -1.0
        return index, sign

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        tokens = tokenize(text)
        counts: dict[str, float] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0.0) + 1.0
        for a, b in zip(tokens, tokens[1:]):
            counts[f"{a}_{b}"] = counts.get(f"{a}_{b}", 0.0) + 0.6
        for feature, count in counts.items():
            index, sign = self._slot(feature)
            vec[index] += sign * (1.0 + math.log(count))
        return vec

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return _normalise(np.stack([self._vector(t) for t in texts]))


class OpenAIEmbeddings:
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", base_url: str | None = None, batch_size: int = 64):
        from openai import OpenAI  # imported lazily so the local provider needs no SDK

        self.client = OpenAI(api_key=api_key, base_url=base_url or None)
        self.model = model
        self.name = f"openai:{model}"
        self.batch_size = batch_size
        self.dim = 0  # learned from the first response

    def embed(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim or 1536), dtype=np.float32)
        rows: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = [t.replace("\n", " ") for t in texts[start : start + self.batch_size]]
            response = self.client.embeddings.create(model=self.model, input=batch)
            rows.extend(item.embedding for item in sorted(response.data, key=lambda d: d.index))
        matrix = _normalise(np.asarray(rows, dtype=np.float32))
        self.dim = matrix.shape[1]
        return matrix


def make_embeddings(provider: str, *, api_key: str | None, model: str, base_url: str | None, local_dim: int) -> EmbeddingProvider:
    if provider == "openai":
        if not api_key:
            raise RuntimeError("EMBEDDING_PROVIDER=openai needs OPENAI_API_KEY.")
        return OpenAIEmbeddings(api_key=api_key, model=model, base_url=base_url)
    return LocalEmbeddings(dim=local_dim)
