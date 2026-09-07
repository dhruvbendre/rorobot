"""
The archive, end to end.

    Archive(settings)
        .status        what the archive holds (documents, placeholders, chunks)
        .rebuild()     re-index the knowledge folder
        .ask(question, history) -> Answer

An answer is grounded only when retrieval clears the threshold. Otherwise no
model is called at all and the archive says it does not hold the answer. With
LLM_PROVIDER=none the grounded path returns the passages themselves.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Iterator

from config.settings import Settings

from .chunker import chunk_documents
from .embeddings import EmbeddingProvider, make_embeddings
from .llm import ChatProvider, Message, ProviderError, make_chat
from .loader import Document, knowledge_fingerprint, load_documents
from .prompts import (
    CONDENSE_PROMPT,
    EMPTY_ARCHIVE_NOTE,
    NOT_ENOUGH_VARIANTS,
    format_context,
    system_prompt,
    user_turn_with_context,
)
from .retriever import Hit, Retrieval, Retriever
from .vector_store import VectorStore


@dataclass
class Source:
    label: str
    score: float
    breadcrumb: str


@dataclass
class Answer:
    question: str
    grounded: bool
    text: str = ""
    stream: Iterator[str] | None = None
    sources: list[Source] = field(default_factory=list)
    mode: str = "generated"  # generated | extractive | refusal | error
    retrieval: Retrieval | None = None

    def collect(self) -> str:
        """Drain the stream (if any) into `text` and return it."""
        if self.stream is not None:
            self.text = "".join(self.stream)
            self.stream = None
        return self.text


@dataclass
class ArchiveStatus:
    documents: int
    placeholders: int
    live_documents: int
    chunks: int
    embedding: str
    llm: str
    generates: bool
    test_data: bool

    @property
    def empty(self) -> bool:
        return self.chunks == 0


class Archive:
    def __init__(self, settings: Settings, *, embeddings: EmbeddingProvider | None = None, chat: ChatProvider | None = None):
        self.settings = settings
        self.embeddings = embeddings or make_embeddings(
            settings.embedding_provider,
            api_key=settings.openai_api_key,
            model=settings.embedding_model,
            base_url=settings.openai_base_url,
            local_dim=settings.local_embedding_dim,
        )
        self.chat = chat or make_chat(
            settings.llm_provider,
            model=settings.model_name,
            openai_key=settings.openai_api_key,
            openai_base_url=settings.openai_base_url,
            anthropic_key=settings.anthropic_api_key,
        )
        self.documents: list[Document] = []
        self.store: VectorStore | None = None
        self.retriever: Retriever | None = None
        self._rng = random.Random(7)
        self.load()

    # --- indexing -----------------------------------------------------------

    def _manifest(self, docs: list[Document]) -> dict:
        return {
            "knowledge": knowledge_fingerprint(docs),
            "embedding": self.embeddings.name,
            "test_data": self.settings.include_test_data,
        }

    def load(self) -> None:
        """Load the persisted index if it matches the current knowledge; otherwise rebuild."""
        self.documents = load_documents(self.settings.knowledge_dir, include_test_data=self.settings.include_test_data)
        expected = self._manifest(self.documents)
        store = None
        if VectorStore.manifest_matches(self.settings.index_dir, expected):
            store = VectorStore.load(self.settings.index_dir)
        if store is None:
            store = self._build(self.documents, expected)
            store.save(self.settings.index_dir)
        self._attach(store)

    def rebuild(self) -> ArchiveStatus:
        self.documents = load_documents(self.settings.knowledge_dir, include_test_data=self.settings.include_test_data)
        store = self._build(self.documents, self._manifest(self.documents))
        store.save(self.settings.index_dir)
        self._attach(store)
        return self.status

    def _build(self, docs: list[Document], manifest: dict) -> VectorStore:
        chunks = chunk_documents(docs)
        vectors = self.embeddings.embed([c.text for c in chunks])
        manifest = dict(manifest, chunks=len(chunks), dim=int(vectors.shape[1]) if len(chunks) else 0)
        return VectorStore(vectors, chunks, manifest)

    def _attach(self, store: VectorStore) -> None:
        self.store = store
        self.retriever = Retriever(store, self.embeddings, top_k=self.settings.top_k, min_score=self.settings.min_score)

    @property
    def status(self) -> ArchiveStatus:
        placeholders = sum(1 for d in self.documents if d.placeholder)
        return ArchiveStatus(
            documents=len(self.documents),
            placeholders=placeholders,
            live_documents=len(self.documents) - placeholders,
            chunks=len(self.store) if self.store else 0,
            embedding=self.embeddings.name,
            llm=self.chat.name,
            generates=self.chat.generates,
            test_data=self.settings.include_test_data,
        )

    # --- asking ---------------------------------------------------------------

    def _system(self) -> str:
        s = self.settings
        return system_prompt(archive_name=s.archive_name, owner_name=s.owner_name, owner_short=s.owner_short)

    def _recent(self, history: list[Message]) -> list[Message]:
        """The last N turns, roles alternating, without the current question."""
        turns = [m for m in history if m.get("role") in {"user", "assistant"} and m.get("content")]
        return turns[-(self.settings.history_turns * 2):]

    def condense(self, question: str, history: list[Message]) -> str:
        """Turn a follow-up into a standalone query for retrieval. Falls back to the raw question."""
        recent = self._recent(history)
        if not recent or not self.settings.condense_followups or not self.chat.generates:
            return question
        transcript = "\n".join(f"{m['role']}: {m['content']}" for m in recent[-6:])
        prompt = CONDENSE_PROMPT.format(owner_name=self.settings.owner_name)
        try:
            rewritten = self.chat.complete(
                prompt,
                [{"role": "user", "content": f"Conversation so far:\n{transcript}\n\nLatest message: {question}"}],
                max_tokens=120,
            )
        except ProviderError:
            return question
        rewritten = rewritten.strip().strip('"')
        return rewritten if 0 < len(rewritten) <= 300 else question

    def not_enough(self) -> str:
        return self._rng.choice(NOT_ENOUGH_VARIANTS).format(owner_short=self.settings.owner_short)

    def ask(self, question: str, history: list[Message] | None = None) -> Answer:
        history = history or []
        question = question.strip()
        if not question:
            return Answer(question=question, grounded=False, text=self.not_enough(), mode="refusal")
        assert self.retriever is not None

        query = self.condense(question, history)
        retrieval = self.retriever.retrieve(query)
        if not retrieval.grounded:
            text = self.not_enough()
            if self.status.empty:
                text += "\n\n" + EMPTY_ARCHIVE_NOTE.format(owner_short=self.settings.owner_short)
            return Answer(question=question, grounded=False, text=text, mode="refusal", retrieval=retrieval)

        hits = retrieval.strong_hits
        sources = _sources(hits)

        if not self.chat.generates:
            return Answer(question=question, grounded=True, text=self._extractive(hits), sources=sources, mode="extractive", retrieval=retrieval)

        context = format_context(hits, max_chars=self.settings.max_context_chars)
        messages: list[Message] = [*self._recent(history), {"role": "user", "content": user_turn_with_context(question, context)}]
        try:
            stream = self.chat.stream(self._system(), messages, max_tokens=self.settings.max_answer_tokens)
        except ProviderError as exc:
            return Answer(question=question, grounded=True, text=str(exc), sources=sources, mode="error", retrieval=retrieval)
        return Answer(question=question, grounded=True, stream=stream, sources=sources, mode="generated", retrieval=retrieval)

    def _extractive(self, hits: list[Hit]) -> str:
        """No model configured: return what the archive holds, verbatim and labelled."""
        parts = ["Here is what the archive holds on that (no model is connected, so these are the passages themselves):"]
        for hit in hits[:3]:
            body = hit.chunk.text.split("\n\n", 1)[-1].strip()
            parts.append(f"**{hit.chunk.source_label()}**\n\n{body}")
        return "\n\n".join(parts)


def _sources(hits: list[Hit]) -> list[Source]:
    seen: set[str] = set()
    out: list[Source] = []
    for hit in hits:
        label = hit.chunk.source_label()
        if label in seen:
            continue
        seen.add(label)
        out.append(Source(label=label, score=hit.score, breadcrumb=hit.chunk.breadcrumb))
    return out
