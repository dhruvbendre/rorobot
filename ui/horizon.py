"""
Mnemora's crayon horizon, generated as SVG.

The same construction as the web scene (art.md §3 and §4): a hand-adjusted
disc, broad directional crayon streaks, a few fine scribbles, pressure
highlights, a paper-tooth pattern, and a chalk perimeter made of three
imperfect rings with breaks and dust. Seeded, so it is the same drawing on
every load.
"""
from __future__ import annotations

import math
import random
from functools import lru_cache
from pathlib import Path

# Mnemora's palette (plot.md §4, art.md §2.4)
BODY = "#1889c0"
STREAKS = ["#247fb2", "#146c9d", "#3699c9", "#1f7fb5"]
LIGHT = "#5fb3dd"
SHADE = "#0e4f78"
CHALK = "#f1f1e8"

W, H = 1000, 420
R = 430
TOP = 34
CX, CY = W / 2, TOP + R


def _r(v: float) -> str:
    return f"{v:.1f}"


def rough_disc(cx: float, cy: float, r: float, rng: random.Random, points: int = 22, wobble: float = 0.018) -> str:
    step = 2 * math.pi / points
    kappa = (4 / 3) * math.tan(step / 4)
    pts = []
    for i in range(points):
        a = i * step
        rr = r * (1 + (rng.random() * 2 - 1) * wobble + math.sin(a * 3) * wobble * 0.4)
        pts.append((cx + math.cos(a) * rr, cy + math.sin(a) * rr, -math.sin(a) * rr * kappa, math.cos(a) * rr * kappa))
    d = f"M{_r(pts[0][0])} {_r(pts[0][1])}"
    for i in range(points):
        ax, ay, atx, aty = pts[i]
        bx, by, btx, bty = pts[(i + 1) % points]
        d += f"C{_r(ax + atx)} {_r(ay + aty)} {_r(bx - btx)} {_r(by - bty)} {_r(bx)} {_r(by)}"
    return d + "Z"


def hatch(cx: float, cy: float, r: float, angle: float, count: int, rng: random.Random, colors: list[str], width: tuple[float, float], opacity: tuple[float, float]) -> list[str]:
    a = math.radians(angle)
    ux, uy = math.cos(a), math.sin(a)
    nx, ny = -uy, ux
    out = []
    for i in range(count):
        t = (i + 0.5) / count - 0.5 + (rng.random() - 0.5) * 0.15
        off = t * 2 * r * 0.95
        half = r * rng.uniform(0.55, 1.1) * math.sqrt(max(0.05, 1 - (off / r) ** 2))
        shift = (rng.random() - 0.5) * r * 0.3
        sx, sy = cx + nx * off - ux * half + ux * shift, cy + ny * off - uy * half + uy * shift
        ex, ey = cx + nx * off + ux * half + ux * shift, cy + ny * off + uy * half + uy * shift
        bend = (rng.random() - 0.5) * 2 * 0.07 * r
        mx, my = (sx + ex) / 2 + nx * bend, (sy + ey) / 2 + ny * bend
        out.append(
            f'<path d="M{_r(sx)} {_r(sy)}Q{_r(mx)} {_r(my)} {_r(ex)} {_r(ey)}" stroke="{rng.choice(colors)}" '
            f'stroke-width="{rng.uniform(*width):.1f}" opacity="{rng.uniform(*opacity):.2f}"/>'
        )
    return out


def chalk_ring(r: float, rng: random.Random) -> str:
    """Three imperfect rings with breaks, plus dust. Mirrors web/src/art/chalk.ts."""
    def breaks(circ: float, count: int, gap: tuple[float, float]) -> str:
        parts = []
        remaining = circ
        for i in range(count):
            g = circ * rng.uniform(*gap)
            share = remaining / (count - i)
            run = max(4.0, share * rng.uniform(0.65, 1.15) - g)
            parts += [run, g]
            remaining -= run + g
        if remaining > 0:
            parts[-2] += remaining
        return " ".join(f"{max(0.5, p):.1f}" for p in parts)

    outer = r * 1.035
    weight = 3.0
    rings = [
        (outer, weight, 0.86, breaks(2 * math.pi * outer, rng.randint(2, 3), (0.015, 0.045)), 0.022),
        (outer * 1.012, weight * 0.55, 0.42, breaks(2 * math.pi * outer * 1.012, rng.randint(3, 5), (0.03, 0.08)), 0.03),
        (outer * 0.985, weight * 1.7, 0.12, breaks(2 * math.pi * outer * 0.985, rng.randint(2, 4), (0.05, 0.12)), 0.035),
    ]
    parts = []
    for rr, w, op, dash, wob in rings:
        d = rough_disc(0, 0, rr, rng, 36, wob)
        parts.append(
            f'<path d="{d}" stroke="{CHALK}" stroke-width="{w:.2f}" opacity="{op}" stroke-dasharray="{dash}" '
            f'stroke-dashoffset="{rng.uniform(0, 400):.1f}" transform="translate({rng.uniform(-1.2, 1.2):.2f} {rng.uniform(-1.2, 1.2):.2f})"/>'
        )
    dust = []
    for _ in range(22):
        a = rng.uniform(0, 2 * math.pi)
        rr = outer * rng.uniform(0.96, 1.12)
        dust.append(f'<circle cx="{_r(math.cos(a) * rr)}" cy="{_r(math.sin(a) * rr)}" r="{rng.uniform(0.6, 1.8):.2f}" opacity="{rng.uniform(0.18, 0.5):.2f}"/>')
    return (
        '<g fill="none" stroke-linecap="round" stroke-linejoin="round" filter="url(#mn-chalk-grain)">'
        + "".join(parts)
        + f'<g fill="{CHALK}" stroke="none">' + "".join(dust) + "</g></g>"
    )


@lru_cache(maxsize=1)
def horizon_svg(seed: int = 84) -> str:
    rng = random.Random(seed)
    disc = rough_disc(0, 0, R, rng, 26, 0.014)
    streaks = hatch(0, 0, R, -38, 26, rng, STREAKS, (8, 18), (0.22, 0.42))
    lights = hatch(-R * 0.35, -R * 0.4, R * 0.55, -38, 9, rng, [LIGHT], (5, 10), (0.18, 0.32))
    shade = hatch(R * 0.45, R * 0.3, R * 0.6, -34, 8, rng, [SHADE], (6, 12), (0.16, 0.28))
    cross = hatch(0, 0, R * 0.9, 52, 7, rng, ["#3d8fdc", "#146c9d"], (2.5, 5), (0.14, 0.24))
    # horizon "bands": three soft curved arcs like Neptune's texture bands
    bands = []
    for i in range(3):
        y = -R * 0.85 + i * R * 0.22
        bands.append(
            f'<path d="M{_r(-R * 0.9)} {_r(y)}Q{_r(rng.uniform(-60, 60))} {_r(y + rng.uniform(-30, 30))} {_r(R * 0.9)} {_r(y + rng.uniform(-20, 20))}" '
            f'stroke="#146c9d" stroke-width="{rng.uniform(10, 18):.1f}" opacity="{rng.uniform(0.12, 0.22):.2f}"/>'
        )
    return f"""<svg class="mn-horizon__svg" viewBox="0 0 {W} {H}" preserveAspectRatio="xMidYMin slice" aria-hidden="true">
  <defs>
    <clipPath id="mn-disc"><path d="{disc}"/></clipPath>
    <pattern id="mn-tooth" patternUnits="userSpaceOnUse" width="26" height="26" patternTransform="rotate(23) scale(0.7)">
      <g fill="#fffbe9" opacity="0.5"><circle cx="3" cy="4" r="0.9"/><circle cx="12" cy="9" r="0.7"/><circle cx="20" cy="3" r="0.8"/><circle cx="7" cy="17" r="0.6"/><circle cx="16" cy="21" r="0.9"/><circle cx="23" cy="14" r="0.7"/></g>
      <g fill="#000" opacity="0.14"><circle cx="8" cy="7" r="0.6"/><circle cx="21" cy="19" r="0.7"/><circle cx="14" cy="24" r="0.5"/></g>
    </pattern>
    <filter id="mn-chalk-grain" x="-12%" y="-12%" width="124%" height="124%" color-interpolation-filters="sRGB">
      <feTurbulence type="fractalNoise" baseFrequency="0.045" numOctaves="2" seed="5" result="warp"/>
      <feDisplacementMap in="SourceGraphic" in2="warp" scale="1.8" xChannelSelector="R" yChannelSelector="G" result="bent"/>
      <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves="1" seed="13" result="speckle"/>
      <feColorMatrix in="speckle" type="matrix" values="0 0 0 0 1  0 0 0 0 1  0 0 0 0 1  0 0 0 2.2 -0.45" result="speckleAlpha"/>
      <feComposite in="bent" in2="speckleAlpha" operator="in" result="powder"/>
      <feGaussianBlur in="powder" stdDeviation="0.22"/>
    </filter>
  </defs>
  <g transform="translate({_r(CX)} {_r(CY)})">
    <path d="{disc}" fill="{BODY}"/>
    <g clip-path="url(#mn-disc)" fill="none" stroke-linecap="round">{''.join(streaks)}{''.join(bands)}{''.join(cross)}{''.join(shade)}{''.join(lights)}</g>
    <path d="{disc}" fill="url(#mn-tooth)" opacity="0.28"/>
    <path d="{disc}" fill="none" stroke="{SHADE}" stroke-width="1.5" opacity="0.4"/>
    {chalk_ring(R, rng)}
  </g>
</svg>"""


def glyph_memory(size: int = 18, cls: str = "") -> str:
    """Mnemora's glyph: an open ring holding a single dot (a memory, held)."""
    return (
        f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 24 24" aria-hidden="true">'
        '<g fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
        '<path d="M15.2 5.8 A7 7 0 1 0 18.4 14.6"/><circle cx="12.2" cy="12.1" r="1.25" fill="currentColor" stroke="none"/></g></svg>'
    )


def chalk_star(size: int = 14, cls: str = "") -> str:
    """A tiny hand-drawn cross star used as the archive's answer marker."""
    return (
        f'<svg class="{cls}" width="{size}" height="{size}" viewBox="0 0 16 16" aria-hidden="true">'
        '<g fill="none" stroke="currentColor" stroke-width="1.3" stroke-linecap="round">'
        '<path d="M2.2 8.4 q3 -1 5.6 -0.3 q3 0.6 6 0.1"/><path d="M8.3 2.1 q-0.9 3 -0.2 5.8 q0.5 3 0.1 6"/></g></svg>'
    )


ROOT = Path(__file__).resolve().parent
_ASSETS = ROOT / "assets"


@lru_cache(maxsize=1)
def _roro_markup() -> str:
    """The portfolio's own Roro (rendered by @bible-strong/avatar-react), saved as SVG."""
    return (_ASSETS / "roro.svg").read_text(encoding="utf-8")


def roro(size: int = 56, cls: str = "") -> str:
    """Roro, keeper of the archive: the same avatar the portfolio shows."""
    svg = _roro_markup()
    return svg.replace('class="bs-avatar__svg"', f'class="bs-avatar__svg {cls}" width="{size}" height="{size}"', 1)
