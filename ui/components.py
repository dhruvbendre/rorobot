"""
The pieces of Mnemora's interface, each a small function that renders HTML
into Streamlit. Nothing here knows about retrieval; it only knows how the
world looks.
"""
from __future__ import annotations

import html
import random
from pathlib import Path

import streamlit as st

from config.settings import Settings
from rag.pipeline import Source
from .horizon import chalk_star, glyph_memory, horizon_svg, roro

STYLES = Path(__file__).resolve().parent.parent / "styles" / "streamlit.css"

# Set once by app.py from settings so answer blocks carry the bot's name.
BOT_NAME = "Roro"


def set_bot_name(name: str) -> None:
    global BOT_NAME
    BOT_NAME = name


def inject_theme() -> None:
    css = STYLES.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def _rule(seed: int, width: int = 600) -> str:
    rng = random.Random(seed)
    d = f"M0 {rng.uniform(-1.6, 1.6):.1f}"
    n = 6
    for i in range(1, n + 1):
        x = i / n * width
        d += f"Q{x - width / n / 2 + rng.uniform(-6, 6):.1f} {rng.uniform(-2.6, 2.6):.1f} {x:.1f} {rng.uniform(-1.6, 1.6):.1f}"
    return f'<svg class="mn-rule" viewBox="-2 -6 604 12" preserveAspectRatio="none" aria-hidden="true"><path d="{d}" vector-effect="non-scaling-stroke"/></svg>'


def render_ambient(seed: int = 11) -> None:
    """Fixed layer: two chalk orbit arcs in the top-right corner, sparse stars, blue dust."""
    rng = random.Random(seed)
    stars = "".join(
        f'<circle class="mn-ambient__star" cx="{rng.uniform(0, 1600):.0f}" cy="{rng.uniform(0, 1000):.0f}" r="{rng.uniform(0.7, 1.7):.2f}" opacity="{rng.uniform(0.25, 0.7):.2f}"/>'
        for _ in range(46)
    )
    dust = "".join(
        f'<circle cx="{rng.uniform(800, 1600):.0f}" cy="{rng.uniform(0, 600):.0f}" r="{rng.uniform(0.6, 2.2):.2f}" fill="#1889c0" opacity="{rng.uniform(0.08, 0.3):.2f}"/>'
        for _ in range(90)
    )
    st.markdown(
        f"""<div class="mn-ambient" aria-hidden="true"><svg viewBox="0 0 1600 1000" preserveAspectRatio="xMidYMid slice">
        <path class="mn-ambient__arc" d="M860 -120 C 1300 -80 1720 250 1660 720" stroke-dasharray="420 14 260 9 600 18"/>
        <path class="mn-ambient__arc" d="M700 -160 C 1320 -120 1860 340 1760 900" stroke-dasharray="700 12 380 16 500 10"/>
        {stars}{dust}</svg></div>""",
        unsafe_allow_html=True,
    )


def render_horizon() -> None:
    st.markdown(f'<div class="mn-horizon" aria-hidden="true">{horizon_svg()}</div>', unsafe_allow_html=True)


def render_head(settings: Settings) -> None:
    back = ""
    if settings.portfolio_url:
        back = (
            f'<a class="mn-back" href="{html.escape(settings.portfolio_url)}">'
            '<svg class="mn-back__sun" viewBox="0 0 40 40" aria-hidden="true"><g fill="#f5a82c" opacity="0.9">'
            '<path d="M20 4 l2.4 5 l-4.6 0.3z"/><path d="M31 8.6 l-0.4 5.4 l-3.8 -2.6z"/><path d="M35.4 19.6 l-4.8 2.6 l-0.2 -4.6z"/>'
            '<path d="M31.6 31 l-5.4 -0.8 l2.4 -3.8z"/><path d="M20.4 36 l-2.8 -4.8 l4.6 -0.4z"/><path d="M8.6 31.4 l0.6 -5.4 l3.8 2.4z"/>'
            '<path d="M4.6 20.2 l4.8 -2.4 l0 4.6z"/><path d="M8.2 8.8 l5.4 0.6 l-2.2 4z"/></g><circle cx="20" cy="20" r="10.5" fill="#ffd84a"/>'
            '<path d="M20.4 8.2 a12 12 0 1 1 -0.6 0" fill="none" stroke="#f1f1e8" stroke-width="1.3" stroke-linecap="round" stroke-dasharray="30 4 22 3 12 5" opacity="0.85"/></svg>'
            "<span>Back to the system</span></a>"
        )
    st.markdown(
        f"""<header class="mn-head">
        <div class="mn-head__row">{back or '<span></span>'}<span class="mn-catalog">{glyph_memory(16)}<span>World VIII</span></span></div>
        <p class="mn-eyebrow">VIII · The archive · {html.escape(settings.archive_name)}</p>
        <div class="mn-roro">{roro(72, "mn-roro__body")}<div><h1 class="mn-name">{html.escape(settings.bot_name)}</h1>
        <p class="mn-bot-tagline">{html.escape(settings.bot_tagline)}</p></div></div>
        <p class="mn-tagline">{html.escape(settings.tagline)}</p>
        {_rule(84)}
        </header>""",
        unsafe_allow_html=True,
    )


def render_empty_state(settings: Settings, categories: list[tuple[str, str]], *, archive_empty: bool, note: str) -> str | None:
    """The opening screen. Returns a question if the visitor picked a category."""
    st.markdown(
        f"""<section class="mn-empty"><p class="mn-empty__prompt">{html.escape(settings.empty_state)}</p>
        <p class="mn-empty__hint">Start from a category</p></section>""",
        unsafe_allow_html=True,
    )
    picked: str | None = None
    per_row = 4
    for row_start in range(0, len(categories), per_row):
        row = categories[row_start : row_start + per_row]
        cols = st.columns(per_row)
        for col, (label, question) in zip(cols, row):
            slug = "".join(ch if ch.isalnum() else "-" for ch in label.lower())
            with col:
                with st.container(key=f"chip-{slug}"):
                    if st.button(label, key=f"chip-btn-{slug}", use_container_width=True):
                        picked = question
    if archive_empty:
        st.markdown(f'<p class="mn-note">{html.escape(note)}</p>', unsafe_allow_html=True)
    return picked


def render_user(text: str) -> None:
    st.markdown(
        f'<div class="mn-msg mn-msg--you"><span class="mn-msg__who">you</span><p class="mn-msg__body">{html.escape(text)}</p></div>',
        unsafe_allow_html=True,
    )


def _sources_html(sources: list[Source]) -> str:
    if not sources:
        return ""
    items = "".join(f'<span class="mn-sources__item">{html.escape(s.label)}</span>' for s in sources[:4])
    return f'<p class="mn-sources"><span class="mn-sources__label">from the archive</span>{items}</p>'


def archive_container(index: int, *, mode: str):
    """A keyed container styled as an archive answer, with its chalk mark."""
    container = st.container(key=f"archive-{index}")
    with container:
        mark = chalk_star(15) if mode in {"refusal", "error"} else roro(22)
        label = BOT_NAME if mode != "error" else f"{BOT_NAME} · interrupted"
        st.markdown(
            f'<div class="mn-answer-head{" mn-refusal" if mode == "refusal" else ""}"><span class="mn-mark">{mark}</span>{label}</div>',
            unsafe_allow_html=True,
        )
    return container


def render_answer(index: int, text: str, sources: list[Source], *, mode: str) -> None:
    with archive_container(index, mode=mode):
        st.markdown(text)
        if sources:
            st.markdown(_sources_html(sources), unsafe_allow_html=True)


def render_sources(sources: list[Source]) -> None:
    if sources:
        st.markdown(_sources_html(sources), unsafe_allow_html=True)


def thinking_placeholder():
    slot = st.empty()
    slot.markdown(f'<span class="mn-thinking">{html.escape(BOT_NAME)} is remembering<i></i><i></i><i></i></span>', unsafe_allow_html=True)
    return slot


def render_footer(status_line: str) -> None:
    st.markdown(f'<p class="mn-foot">{html.escape(status_line)}</p>', unsafe_allow_html=True)
