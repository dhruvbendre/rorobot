"""
Section-aware chunking.

A document is split on its markdown headings first, so a chunk never straddles
two topics. Long sections are then packed paragraph by paragraph up to a word
budget, with the previous paragraph repeated as overlap so a fact that sits on
a boundary is retrievable from either side. Every chunk carries a breadcrumb
("Projects / Project X / What it taught") so the model, and the sources panel,
know where it came from.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .loader import Document

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*?)\s*#*\s*$")


@dataclass
class Chunk:
    id: str
    text: str
    breadcrumb: str
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def category(self) -> str:
        return self.metadata.get("category", "")

    @property
    def title(self) -> str:
        return self.metadata.get("title", "")

    @property
    def section(self) -> str:
        return self.metadata.get("section", "")

    def source_label(self) -> str:
        """The label shown in the 'from the archive' line, e.g. 'Projects / Project X'."""
        parts = [self.category]
        if self.section and self.section.lower() != self.category.lower():
            parts.append(self.section)
        elif self.title and self.title.lower() != self.category.lower():
            parts.append(self.title)
        return " / ".join(p for p in parts if p)


def _split_sections(text: str) -> list[tuple[str, list[str], str]]:
    """
    Return (document heading, sub-heading path, body) triples in document order.
    The level-1 heading names the document; levels 2-4 form the section path.
    """
    sections: list[tuple[str, list[str], str]] = []
    h1 = ""
    path: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        body = "\n".join(buffer).strip()
        if body:
            sections.append((h1, list(path), body))
        buffer.clear()

    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            title = match.group(2).strip()
            if level == 1:
                h1 = title
                path = []
            else:
                path = path[: level - 2] + [title]
        else:
            buffer.append(line)
    flush()
    return sections


def _paragraphs(body: str) -> list[str]:
    parts = re.split(r"\n\s*\n", body)
    out: list[str] = []
    for part in parts:
        part = part.strip()
        if part:
            out.append(part)
    return out


def _words(text: str) -> int:
    return len(text.split())


def chunk_document(doc: Document, *, max_words: int = 220, min_words: int = 25) -> list[Chunk]:
    chunks: list[Chunk] = []
    base_meta = dict(doc.metadata)
    base_meta.setdefault("title", doc.title)
    base_meta.setdefault("category", doc.category)
    title = base_meta["title"]

    for h1, path, body in _split_sections(doc.text):
        section = " / ".join(p for p in path if p) or title
        heading = [h1] if h1 and h1.lower() != title.lower() else []
        breadcrumb = " / ".join(dict.fromkeys([base_meta["category"], title, *heading, *path]))
        paragraphs = _paragraphs(body)
        packs: list[list[str]] = []
        current: list[str] = []
        current_words = 0
        for para in paragraphs:
            w = _words(para)
            if current and current_words + w > max_words:
                packs.append(current)
                # overlap: carry the last paragraph forward if it is short
                carry = [current[-1]] if _words(current[-1]) <= max_words // 3 else []
                current = carry + [para]
                current_words = sum(_words(p) for p in current)
            else:
                current.append(para)
                current_words += w
        if current:
            packs.append(current)

        # Merge a trailing sliver into the previous pack so no chunk is a stub.
        merged: list[list[str]] = []
        for pack in packs:
            if merged and sum(_words(p) for p in pack) < min_words:
                merged[-1] = merged[-1] + pack
            else:
                merged.append(pack)

        for i, pack in enumerate(merged):
            text = "\n\n".join(pack)
            meta = dict(base_meta)
            meta["section"] = section
            chunk_id = f"{doc.path.stem}:{len(chunks):03d}"
            chunks.append(Chunk(id=chunk_id, text=f"{breadcrumb}\n\n{text}", breadcrumb=breadcrumb, metadata=meta))
    return chunks


def chunk_documents(docs: list[Document], **kwargs) -> list[Chunk]:
    out: list[Chunk] = []
    for doc in docs:
        if doc.placeholder:
            continue
        out.extend(chunk_document(doc, **kwargs))
    return out
