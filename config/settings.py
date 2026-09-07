"""
Runtime configuration for the archive.

Every value is read from the environment (a local `.env` is loaded if present)
and, when running under Streamlit, from `st.secrets` as a fallback. Nothing is
hard-coded: no keys, no domains, no model names that cannot be overridden.

Provider selection
------------------
LLM_PROVIDER        openai | anthropic | none      (default: auto-detect from keys)
EMBEDDING_PROVIDER  openai | local                 (default: openai if a key exists, else local)

"none" is a real mode: the archive still retrieves, and answers by quoting the
retrieved passages instead of generating prose. It is how retrieval and the
refusal behaviour are tested without any API key.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:  # optional: a local .env for development
    from dotenv import load_dotenv

    load_dotenv(override=False)
except Exception:  # pragma: no cover - dotenv is optional
    pass

ROOT = Path(__file__).resolve().parent.parent


def _secret(name: str) -> str | None:
    """Streamlit Cloud injects secrets through st.secrets, not the environment."""
    try:
        import streamlit as st  # noqa: WPS433 (runtime import keeps tests free of Streamlit)

        if name in st.secrets:  # type: ignore[operator]
            value = st.secrets[name]
            return str(value) if value is not None else None
    except Exception:
        return None
    return None


def env(name: str, default: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        value = _secret(name)
    if value is None or value.strip() == "":
        return default
    return value.strip()


def env_bool(name: str, default: bool) -> bool:
    value = env(name)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def env_float(name: str, default: float) -> float:
    value = env(name)
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    value = env(name)
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    # --- identity ---------------------------------------------------------
    archive_name: str = "Mnemora"
    owner_name: str = "Dhruv Bendre"
    owner_short: str = "Dhruv"
    tagline: str = "Ask the archive about Dhruv."
    empty_state: str = "Ask me anything about Dhruv."
    portfolio_url: str | None = None
    # The character who answers. Roro keeps the archive; Mnemora is the world.
    bot_name: str = "Roro"
    bot_tagline: str = "keeper of the archive"

    # --- providers --------------------------------------------------------
    llm_provider: str = "none"
    model_name: str = ""
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    anthropic_api_key: str | None = None
    embedding_provider: str = "local"
    embedding_model: str = "text-embedding-3-small"
    local_embedding_dim: int = 512

    # --- retrieval --------------------------------------------------------
    top_k: int = 5
    min_score: float = 0.0
    max_context_chars: int = 9000
    history_turns: int = 6
    condense_followups: bool = True
    max_answer_tokens: int = 160

    # --- storage ----------------------------------------------------------
    knowledge_dir: Path = field(default_factory=lambda: ROOT / "knowledge")
    index_dir: Path = field(default_factory=lambda: ROOT / ".index")
    include_test_data: bool = False


def default_min_score(embedding_provider: str) -> float:
    """
    Cosine-similarity floor below which the archive refuses to answer.
    Real embedding models cluster higher than the hashed local fallback.
    """
    return 0.30 if embedding_provider == "openai" else 0.18


def load_settings(overrides: dict | None = None) -> Settings:
    overrides = overrides or {}
    openai_key = env("OPENAI_API_KEY")
    anthropic_key = env("ANTHROPIC_API_KEY")

    llm_provider = (env("LLM_PROVIDER") or "").lower()
    if llm_provider not in {"openai", "anthropic", "none"}:
        llm_provider = "anthropic" if anthropic_key else "openai" if openai_key else "none"

    openai_base_url = env("OPENAI_BASE_URL")
    # OpenAI-compatible chat endpoints (xAI Grok, Groq, Together, Ollama ...) usually
    # have no embeddings endpoint, so only default to OpenAI embeddings on OpenAI itself.
    real_openai = bool(openai_key) and (not openai_base_url or "api.openai.com" in openai_base_url)
    embedding_provider = (env("EMBEDDING_PROVIDER") or "").lower()
    if embedding_provider not in {"openai", "local"}:
        embedding_provider = "openai" if real_openai else "local"

    model_name = env("MODEL_NAME") or ("claude-opus-5" if llm_provider == "anthropic" else "gpt-4o-mini")

    knowledge_dir = Path(env("KNOWLEDGE_DIR") or (ROOT / "knowledge"))
    index_dir = Path(env("INDEX_DIR") or (ROOT / ".index"))

    values = dict(
        archive_name=env("ARCHIVE_NAME", "Mnemora"),
        owner_name=env("OWNER_NAME", "Dhruv Bendre"),
        owner_short=env("OWNER_SHORT", "Dhruv"),
        tagline=env("ARCHIVE_TAGLINE", "Ask the archive about Dhruv."),
        empty_state=env("ARCHIVE_EMPTY_STATE", "Ask me anything about Dhruv."),
        portfolio_url=env("PORTFOLIO_URL"),
        bot_name=env("BOT_NAME", "Roro"),
        bot_tagline=env("BOT_TAGLINE", "keeper of the archive"),
        llm_provider=llm_provider,
        model_name=model_name,
        openai_api_key=openai_key,
        openai_base_url=openai_base_url,
        anthropic_api_key=anthropic_key,
        embedding_provider=embedding_provider,
        embedding_model=env("EMBEDDING_MODEL", "text-embedding-3-small"),
        local_embedding_dim=env_int("LOCAL_EMBEDDING_DIM", 512),
        top_k=env_int("RETRIEVAL_TOP_K", 5),
        min_score=env_float("RETRIEVAL_MIN_SCORE", default_min_score(embedding_provider)),
        max_context_chars=env_int("MAX_CONTEXT_CHARS", 9000),
        history_turns=env_int("HISTORY_TURNS", 6),
        condense_followups=env_bool("CONDENSE_FOLLOWUPS", True),
        max_answer_tokens=env_int("MAX_ANSWER_TOKENS", 160),
        knowledge_dir=knowledge_dir,
        index_dir=index_dir,
        include_test_data=env_bool("INCLUDE_TEST_DATA", False),
    )
    values.update(overrides)
    return Settings(**values)
