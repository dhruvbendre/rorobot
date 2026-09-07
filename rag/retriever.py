"""
Retrieval with a grounding threshold.

`retrieve` returns the best chunks for a query. `grounded` is True only when
the strongest hit clears `min_score`; below it, the pipeline refuses rather
than letting the model improvise from weak context. Hits from the same
document section are deduplicated so the context window is not five copies of
one paragraph.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .chunker import Chunk
from .embeddings import EmbeddingProvider, tokenize
from .vector_store import VectorStore

# How much an exact-word match can add on top of the embedding similarity.
LEXICAL_WEIGHT = 0.35
# Coverage below this adds nothing: a stray shared word is not evidence.
LEXICAL_FLOOR = 0.3
# Question filler that the embedding stopword list keeps but carries no topic.
LEXICAL_STOP = {
    "how", "why", "when", "where", "who", "which", "what",
    "done", "tell", "know", "like", "thing", "things", "something", "anything", "give", "get", "list", "show", "mean", "kind", "sort", "please", "also",
}
# A query word found in the chunk's own heading trail counts this much more.
HEADING_BOOST = 2.2


def stem(token: str) -> str:
    """A very small stemmer so 'internships' meets 'intern' and 'worked' meets 'work'."""
    if len(token) <= 4:
        return token
    if token.endswith("ss"):
        return token
    for suffix, repl in (("ships", ""), ("ing", ""), ("ies", "y"), ("ed", ""), ("es", ""), ("s", "")):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            return token[: -len(suffix)] + repl
    return token


@dataclass
class Hit:
    chunk: Chunk
    score: float


@dataclass
class Retrieval:
    query: str
    hits: list[Hit]
    min_score: float

    @property
    def grounded(self) -> bool:
        return bool(self.hits) and self.hits[0].score >= self.min_score

    @property
    def strong_hits(self) -> list[Hit]:
        """Hits above the threshold, plus near-misses that support them."""
        if not self.grounded:
            return []
        floor = max(self.min_score * 0.75, self.hits[0].score * 0.55)
        return [h for h in self.hits if h.score >= floor]


class Retriever:
    """
    Hybrid scoring: embedding similarity plus an IDF-weighted exact-word
    coverage of the query. The lexical part rescues the questions the hashed
    local embeddings handle worst (a single project name, a plural), and only
    counts once at least LEXICAL_FLOOR of the query's word mass is matched, so
    an unrelated question that merely mentions the owner's name gains nothing.
    """

    def __init__(self, store: VectorStore, embeddings: EmbeddingProvider, *, top_k: int = 5, min_score: float = 0.3):
        self.store = store
        self.embeddings = embeddings
        self.top_k = top_k
        self.min_score = min_score
        self._terms: list[set[str]] = [set(stem(t) for t in tokenize(c.text)) for c in store.chunks]
        # Words in a chunk's own heading trail count extra: a question naming a
        # project should land on that project's section, not a passing mention.
        self._headings: list[set[str]] = [set(stem(t) for t in tokenize(c.breadcrumb)) for c in store.chunks]
        n = max(1, len(self._terms))
        df: dict[str, int] = {}
        for terms in self._terms:
            for t in terms:
                df[t] = df.get(t, 0) + 1
        self._idf = {t: math.log((n + 1) / (d + 0.5)) for t, d in df.items()}
        # Words in a third or more of the chunks (the owner's name, "project")
        # say nothing about *which* section is meant, so they never earn the
        # heading boost.
        self._common = {t for t, d in df.items() if d * 3 >= n}
        # A word the archive has never seen is the most specific word possible:
        # it weighs the most and can never be matched, which is exactly how an
        # unrelated question ("marine biology") fails to earn a boost.
        self._unseen = math.log((n + 1) / 0.5)

    def _lexical(self, query: str) -> list[float]:
        """Per-chunk coverage in [0, 1] of the query's IDF-weighted stems."""
        stems = {stem(t) for t in tokenize(query) if t not in LEXICAL_STOP}
        weights = {t: self._idf.get(t, self._unseen) for t in stems}
        # +1 so a query made only of common words ("what does he do") cannot
        # reach full coverage on whatever chunk happens to share them.
        total = sum(weights.values()) + 1.0
        if total <= 1.0:
            return [0.0] * len(self._terms)
        out: list[float] = []
        for terms, headings in zip(self._terms, self._headings):
            matched = sum(w * (HEADING_BOOST if t in headings and t not in self._common else 1.0) for t, w in weights.items() if t in terms)
            coverage = min(1.0, matched / total)
            out.append(max(0.0, coverage - LEXICAL_FLOOR) / (1 - LEXICAL_FLOOR))
        return out

    def retrieve(self, query: str) -> Retrieval:
        query = query.strip()
        if not query or len(self.store) == 0:
            return Retrieval(query=query, hits=[], min_score=self.min_score)
        vector = self.embeddings.embed([query])[0]
        lexical = self._lexical(query)
        index = {id(c): i for i, c in enumerate(self.store.chunks)}
        # A negative cosine is noise from the hashed fallback, not evidence against a chunk.
        raw = [(chunk, max(0.0, score) + LEXICAL_WEIGHT * lexical[index[id(chunk)]]) for chunk, score in self.store.search(vector, top_k=len(self.store))]
        raw.sort(key=lambda pair: -pair[1])
        hits: list[Hit] = []
        seen: set[tuple[str, str]] = set()
        for chunk, score in raw:
            key = (chunk.metadata.get("source", ""), chunk.section)
            if key in seen:
                continue
            seen.add(key)
            hits.append(Hit(chunk=chunk, score=score))
            if len(hits) >= self.top_k:
                break
        return Retrieval(query=query, hits=hits, min_score=self.min_score)
