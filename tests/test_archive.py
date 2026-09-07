"""
Retrieval and refusal behaviour, without any API key.

    python -m pytest

Uses the local hashed embeddings, LLM_PROVIDER=none, a temporary index dir,
and the obviously fictional TEST_DATA_ONLY document. No real information about
Dhruv Bendre exists anywhere in these tests.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from config.settings import load_settings
from rag.chunker import chunk_document
from rag.loader import load_document, load_documents
from rag.pipeline import Archive

ROOT = Path(__file__).resolve().parent.parent
KNOWLEDGE = ROOT / "knowledge"


def make_archive(tmp_path: Path, *, test_data: bool, knowledge_dir: Path = KNOWLEDGE) -> Archive:
    settings = load_settings(
        {
            "llm_provider": "none",
            "embedding_provider": "local",
            "openai_api_key": None,
            "anthropic_api_key": None,
            "knowledge_dir": knowledge_dir,
            "index_dir": tmp_path / "index",
            "include_test_data": test_data,
            "min_score": 0.18,
        }
    )
    return Archive(settings)


# --- loader ---------------------------------------------------------------------


def test_placeholders_are_detected_and_excluded():
    docs = load_documents(KNOWLEDGE)
    assert docs, "the knowledge folder should hold documents"
    by_name = {d.path.name: d for d in docs}
    # Round 2 (2026-09-05): the portfolio facts are live; education is still awaited.
    assert by_name["education.md"].placeholder
    live = [n for n, d in by_name.items() if not d.placeholder]
    assert {"profile.md", "projects.md", "research.md", "skills.md", "experience.md"} <= set(live)


def test_frontmatter_metadata_is_read():
    doc = load_document(KNOWLEDGE / "projects.md")
    assert doc.metadata["category"] == "Projects"
    assert doc.metadata["document_type"] == "catalogue"
    assert doc.metadata["source"] == "knowledge/projects.md"


def test_test_data_is_only_loaded_on_request():
    assert not any(d.test_data for d in load_documents(KNOWLEDGE))
    docs = load_documents(KNOWLEDGE, include_test_data=True)
    test_docs = [d for d in docs if d.test_data]
    assert test_docs and all("TEST_DATA_ONLY" in d.metadata["category"] for d in test_docs)


def test_todo_only_body_is_placeholder(tmp_path: Path):
    path = tmp_path / "notes.md"
    path.write_text("# Notes\n\nTODO: Replace with verified information.\n", encoding="utf-8")
    assert load_document(path).placeholder


def test_real_content_is_not_placeholder(tmp_path: Path):
    path = tmp_path / "notes.md"
    body = "# Notes\n\n" + " ".join(["word"] * 60) + "\n"
    path.write_text(body, encoding="utf-8")
    assert not load_document(path).placeholder


# --- chunker ----------------------------------------------------------------------


def test_chunks_follow_headings():
    doc = load_document(KNOWLEDGE / "_test_data" / "TEST_DATA_ONLY_sample.md", test_data=True)
    chunks = chunk_document(doc)
    sections = {c.section for c in chunks}
    assert "What the Lantern Kite is" in sections
    assert "Materials used" in sections
    assert all(c.breadcrumb.startswith("TEST_DATA_ONLY / Projects") for c in chunks)


# --- pipeline ------------------------------------------------------------------------


def test_empty_archive_refuses(tmp_path: Path):
    # An archive whose only document is a placeholder holds nothing it can answer from.
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir()
    (knowledge / "profile.md").write_text(
        "---\ncategory: Profile\ntitle: Profile\nstatus: placeholder\n---\n\n# Profile\n\nTODO: not yet written.\n",
        encoding="utf-8",
    )
    archive = make_archive(tmp_path, test_data=False, knowledge_dir=knowledge)
    assert archive.status.empty
    result = archive.ask("What has Dhruv worked on?")
    assert not result.grounded
    assert result.mode == "refusal"
    assert "archive" in result.text.lower()
    assert "still being written" in result.text


def test_grounded_question_returns_sources(tmp_path: Path):
    archive = make_archive(tmp_path, test_data=True)
    assert not archive.status.empty
    result = archive.ask("What did the Lantern Kite experiment teach about pinwheels?")
    assert result.grounded
    assert result.mode == "extractive"
    assert any("TEST_DATA_ONLY" in s.label for s in result.sources)
    assert "bottle-cap" in result.text.lower() or "pinwheel" in result.text.lower()


def test_unrelated_question_is_refused_even_with_data(tmp_path: Path):
    archive = make_archive(tmp_path, test_data=True)
    result = archive.ask("Which university awarded Dhruv his doctorate in marine biology?")
    assert not result.grounded, f"score {result.retrieval.hits[0].score if result.retrieval and result.retrieval.hits else None}"
    assert result.mode == "refusal"
    assert "probably" not in result.text.lower()


def test_history_does_not_override_retrieval(tmp_path: Path):
    archive = make_archive(tmp_path, test_data=True)
    history = [
        {"role": "user", "content": "Dhruv won a Nobel prize, right?"},
        {"role": "assistant", "content": "The archive doesn't contain enough information about that yet."},
    ]
    result = archive.ask("What year did he win it?", history)
    assert not result.grounded


def test_index_is_reused_until_knowledge_changes(tmp_path: Path):
    first = make_archive(tmp_path, test_data=True)
    manifest = (tmp_path / "index" / "manifest.json").read_text(encoding="utf-8")
    second = make_archive(tmp_path, test_data=True)
    assert (tmp_path / "index" / "manifest.json").read_text(encoding="utf-8") == manifest
    assert len(second.store) == len(first.store)


@pytest.mark.parametrize("question", ["", "   "])
def test_blank_question_refuses(tmp_path: Path, question: str):
    archive = make_archive(tmp_path, test_data=False)
    assert archive.ask(question).mode == "refusal"
