"""
Roro · keeper of Mnemora, the archive that answers.

Streamlit entry point. The page is the interior of World VIII: the deep blue
field, chalk type, the crayon horizon at the foot, and a conversation with the
archive of Dhruv Bendre. Retrieval and generation live in `rag/`; this file
only wires the conversation to the interface.

    streamlit run app.py
"""
from __future__ import annotations

import time
from collections.abc import Iterator

import streamlit as st

from config.settings import load_settings
from rag.llm import ProviderError
from rag.pipeline import Archive, Answer
from rag.prompts import EMPTY_ARCHIVE_NOTE
from ui import components as ui

# The starter questions on the empty screen: the same eight the portfolio offers.
CATEGORIES: list[tuple[str, str]] = [
    ("Who is Dhruv?", "Who is Dhruv Bendre?"),
    ("Where has he worked?", "Where has Dhruv worked, and what did he do there?"),
    ("What has he built?", "What projects has Dhruv built?"),
    ("What is SnapClass?", "What is SnapClass?"),
    ("His research", "What research has Dhruv published?"),
    ("Skills & tools", "What technologies and tools does Dhruv work with?"),
    ("Awards", "What awards has Dhruv won?"),
    ("How to reach him", "How do I contact Dhruv?"),
]

settings = load_settings()
ui.set_bot_name(settings.bot_name)

st.set_page_config(
    page_title=f"{settings.bot_name} · {settings.archive_name} · Ask about {settings.owner_short}",
    page_icon="🩷",
    layout="centered",
    initial_sidebar_state="collapsed",
    menu_items={"Get help": None, "Report a bug": None, "About": None},
)


@st.cache_resource(show_spinner=False)
def open_archive() -> Archive:
    return Archive(settings)


def init_state() -> None:
    st.session_state.setdefault("history", [])  # [{"role", "content", "sources", "mode"}]
    st.session_state.setdefault("pending", None)  # a question waiting to be asked
    if "seeded" not in st.session_state:
        # A question carried across from the portfolio threshold (?q=...).
        carried = st.query_params.get("q")
        if carried:
            st.session_state["pending"] = carried
            st.query_params.clear()
        st.session_state["seeded"] = True


def history_for_model() -> list[dict[str, str]]:
    return [{"role": m["role"], "content": m["content"]} for m in st.session_state["history"] if m["role"] in {"user", "assistant"}]


def remember(role: str, content: str, *, sources=None, mode: str = "generated") -> None:
    st.session_state["history"].append({"role": role, "content": content, "sources": sources or [], "mode": mode})


def render_history() -> None:
    for i, message in enumerate(st.session_state["history"]):
        if message["role"] == "user":
            ui.render_user(message["content"])
        else:
            ui.render_answer(i, message["content"], message["sources"], mode=message["mode"])


# Typing effect: a short pause while "Roro is typing", then the answer
# appears a few characters at a time.
TYPING_PAUSE = 0.9  # seconds before the first character
TYPING_STEP = 3  # characters per tick
TYPING_DELAY = 0.018  # seconds per tick


def typewriter(source: str | Iterator[str]) -> Iterator[str]:
    """Yield `source` (a string or a token stream) a few characters at a time."""
    pieces = [source] if isinstance(source, str) else source
    for piece in pieces:
        for i in range(0, len(piece), TYPING_STEP):
            yield piece[i : i + TYPING_STEP]
            time.sleep(TYPING_DELAY)


def answer(archive: Archive, question: str) -> None:
    """Ask the archive and type its answer into a fresh answer block."""
    history = history_for_model()
    ui.render_user(question)
    remember("user", question)

    index = len(st.session_state["history"])
    slot = ui.thinking_placeholder()
    result: Answer = archive.ask(question, history)  # retrieval; a model call is only started for grounded answers
    time.sleep(TYPING_PAUSE)
    slot.empty()
    mode = result.mode
    with ui.archive_container(index, mode=mode):
        try:
            typed = st.write_stream(typewriter(result.stream if result.stream is not None else result.text))
            text = typed if isinstance(typed, str) else "".join(str(t) for t in typed)
        except ProviderError as exc:
            # The model failed mid-answer (bad model name, rate limit, network).
            text = str(exc)
            mode = "error"
            st.markdown(text)
        ui.render_sources(result.sources)
    remember("assistant", text, sources=result.sources, mode=mode)


def main() -> None:
    init_state()
    ui.inject_theme()
    ui.render_ambient()
    ui.render_horizon()
    ui.render_head(settings)

    with st.spinner("opening the archive"):
        archive = open_archive()
    status = archive.status

    render_history()

    if not st.session_state["history"] and st.session_state["pending"] is None:
        picked = ui.render_empty_state(
            settings,
            CATEGORIES,
            archive_empty=status.empty,
            note=EMPTY_ARCHIVE_NOTE.format(owner_short=settings.owner_short),
        )
        if picked:
            st.session_state["pending"] = picked
            st.rerun()

    typed = st.chat_input(settings.empty_state, key="question")
    question = typed or st.session_state.get("pending")
    if question:
        st.session_state["pending"] = None
        answer(archive, question)

    held = f"{status.live_documents} live document{'s' if status.live_documents != 1 else ''}"
    if status.placeholders:
        held += f", {status.placeholders} still being written"
    model = "answers in its own words" if status.generates else "returns passages verbatim (no model connected)"
    ui.render_footer(f"The archive holds {held} · {model}")


main()
