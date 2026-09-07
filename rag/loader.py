"""
Document loading.

Knowledge lives in `knowledge/*.md`. Each file may start with a frontmatter
block of `key: value` lines between `---` fences:

    ---
    category: Projects
    title: Projects
    source: knowledge/projects.md
    date: 2026-01-01
    project:
    document_type: catalogue
    status: placeholder
    ---

`status: placeholder` (or a body that is nothing but TODO markers) keeps a
document out of the index entirely, so a placeholder can never be retrieved
and read back as a fact. Files under `knowledge/_test_data/` are only loaded
when explicitly asked for (INCLUDE_TEST_DATA=true) and are tagged TEST_DATA_ONLY.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
TODO_RE = re.compile(r"^\s*(TODO|TBD)\b", re.IGNORECASE | re.MULTILINE)
TEST_DIR_NAME = "_test_data"
TEST_TAG = "TEST_DATA_ONLY"

META_KEYS = ("category", "title", "source", "date", "project", "document_type", "status")


@dataclass
class Document:
    path: Path
    text: str
    metadata: dict[str, str] = field(default_factory=dict)
    placeholder: bool = False
    test_data: bool = False

    @property
    def title(self) -> str:
        return self.metadata.get("title") or self.path.stem.replace("_", " ").title()

    @property
    def category(self) -> str:
        return self.metadata.get("category") or self.title


def parse_frontmatter(raw: str) -> tuple[dict[str, str], str]:
    match = FRONTMATTER_RE.match(raw)
    if not match:
        return {}, raw
    meta: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip().strip('"').strip("'")
        if key:
            meta[key] = value
    return meta, raw[match.end():]


def clean_text(text: str) -> str:
    """Normalise whitespace without touching markdown structure."""
    text = text.replace("\r\n", "\n").replace("\t", "  ")
    text = re.sub(r"[  ]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def substantive_words(text: str) -> int:
    """Words that are not headings, TODO lines, or markdown furniture."""
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or TODO_RE.match(stripped):
            continue
        if stripped.startswith("<!--"):
            continue
        count += len(re.findall(r"[A-Za-z0-9']+", stripped))
    return count


def is_placeholder(meta: dict[str, str], body: str) -> bool:
    if meta.get("status", "").lower() in {"placeholder", "todo", "pending"}:
        return True
    # A file that only carries TODO markers and a heading or two is not knowledge.
    return bool(TODO_RE.search(body)) and substantive_words(body) < 40


def load_document(path: Path, *, test_data: bool = False) -> Document:
    raw = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(raw)
    body = clean_text(body)
    meta.setdefault("source", path.as_posix())
    if test_data:
        meta["status"] = TEST_TAG
        meta["category"] = f"{TEST_TAG} / {meta.get('category', path.stem)}"
    return Document(
        path=path,
        text=body,
        metadata={k: v for k, v in meta.items() if k in META_KEYS},
        placeholder=(not test_data) and is_placeholder(meta, body),
        test_data=test_data,
    )


def load_documents(knowledge_dir: Path, *, include_test_data: bool = False) -> list[Document]:
    if not knowledge_dir.exists():
        return []
    docs: list[Document] = []
    for path in sorted(knowledge_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        docs.append(load_document(path))
    if include_test_data:
        test_dir = knowledge_dir / TEST_DIR_NAME
        for path in sorted(test_dir.glob("*.md")) if test_dir.exists() else []:
            docs.append(load_document(path, test_data=True))
    return docs


def knowledge_fingerprint(docs: list[Document]) -> str:
    """Changes whenever any indexed document changes; used to invalidate the index."""
    h = hashlib.sha256()
    for doc in docs:
        h.update(doc.path.name.encode())
        h.update(b"\0placeholder" if doc.placeholder else b"\0live")
        h.update(doc.text.encode("utf-8"))
    return h.hexdigest()[:16]
