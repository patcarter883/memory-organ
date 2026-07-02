#!/usr/bin/env python3
"""Generate docs/assets/carry-vs-m-{light,dark}.svg — the README capacity figure.

Data is RESULTS.md §1 (held-out carry vs M). Two SVGs (GitHub <picture> light/dark);
palette = the repo figure palette, categorical slots in fixed order, validated for
CVD separation and surface contrast on GitHub's light (#ffffff) / dark (#0d1117)
surfaces. Chance (1/M) is a neutral dashed reference, not a categorical hue.

Run from the repo root:  python tools/make_carry_figure.py
"""
import math
import os

# ---- data (RESULTS.md §1) --------------------------------------------------
MS = [8, 16, 32, 64, 128]
RECORD = [(8, 0.948), (16, 0.926), (32, 0.894), (64, 0.921), (128, 0.929)]
NAIVE = [(8, 0.840), (16, 0.025), (32, 0.020)]          # cannot bind past M=32
NOSUP = [(8, 0.360), (16, 0.262), (32, 0.020)]          # the dropped-loss port bug
CHANCE = [(m, 1.0 / m) for m in MS]

# ---- geometry ----------------------------------------------------------------
W, H = 760, 420
PL, PR, PT, PB = 56, 150, 48, 60                        # plot padding (right holds direct labels)
PW, PH = W - PL - PR, H - PT - PB


def x(m):
    return PL + (math.log2(m) - 3) / 4 * PW


def y(v):
    return PT + (1 - v) * PH


def path(pts):
    return "M " + " L ".join(f"{x(m):.1f} {y(v):.1f}" for m, v in pts)


MODES = {
    "light": dict(surface="#ffffff", ink="#0b0b0b", sec="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7",
                  s1="#2a78d6", s2="#1baf7a", s3="#eda100"),
    "dark": dict(surface="#0d1117", ink="#ffffff", sec="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835",
                 s1="#3987e5", s2="#199e70", s3="#c98500"),
}

FONT = 'font-family="system-ui, -apple-system, &quot;Segoe UI&quot;, sans-serif"'


def series(pts, color, surface):
    out = [f'<path d="{path(pts)}" fill="none" stroke="{color}" stroke-width="2" '
           f'stroke-linejoin="round" stroke-linecap="round"/>']
    for m, v in pts:
        out.append(f'<circle cx="{x(m):.1f}" cy="{y(v):.1f}" r="4" fill="{color}" '
                   f'stroke="{surface}" stroke-width="2"/>')
    return "\n".join(out)


def label(px, py, text, color, anchor="start", size=12, weight="600"):
    return (f'<text x="{px:.1f}" y="{py:.1f}" {FONT} font-size="{size}" '
            f'font-weight="{weight}" fill="{color}" text-anchor="{anchor}">{text}</text>')


def build(mode):
    c = MODES[mode]
    e = []
    e.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
             f'viewBox="0 0 {W} {H}" role="img" '
             f'aria-label="Held-out carry vs difficulty M. The product-key store with '
             f'addressing supervision stays flat around 0.93 through M=128 while the naive '
             f'recurrent store and the no-supervision ablation collapse to chance past M=8.">')
    e.append(f'<rect width="{W}" height="{H}" fill="{c["surface"]}"/>')
    # title + subtitle
    e.append(label(16, 22, "Held-out carry vs difficulty M", c["ink"], size=14, weight="700"))
    e.append(label(16, 38, "recall from the frozen store; chance = 1/M · data: RESULTS.md §1",
                   c["sec"], size=11, weight="400"))
    # legend (top-right, one row)
    lx = W - 356
    for dx, col, name in ((0, c["s1"], "pk + addr-sup"), (110, c["s2"], "naive"),
                          (168, c["s3"], "pk, no sup")):
        e.append(f'<rect x="{lx + dx}" y="14" width="10" height="10" rx="2" fill="{col}"/>')
        e.append(label(lx + dx + 14, 23, name, c["sec"], size=11, weight="400"))
    e.append(f'<line x1="{lx + 250}" y1="19" x2="{lx + 260}" y2="19" stroke="{c["muted"]}" '
             f'stroke-width="1.5" stroke-dasharray="4 3"/>')
    e.append(label(lx + 264, 23, "chance", c["sec"], size=11, weight="400"))
    # gridlines + y ticks
    for v in (0, 0.25, 0.5, 0.75, 1.0):
        gy = y(v)
        e.append(f'<line x1="{PL}" y1="{gy:.1f}" x2="{W - PR}" y2="{gy:.1f}" '
                 f'stroke="{c["grid"]}" stroke-width="1"/>')
        e.append(f'<text x="{PL - 8}" y="{gy + 4:.1f}" {FONT} font-size="11" fill="{c["muted"]}" '
                 f'text-anchor="end" style="font-variant-numeric: tabular-nums">{v:g}</text>')
    # baseline + x ticks
    e.append(f'<line x1="{PL}" y1="{y(0):.1f}" x2="{W - PR}" y2="{y(0):.1f}" '
             f'stroke="{c["axis"]}" stroke-width="1"/>')
    for m in MS:
        e.append(f'<text x="{x(m):.1f}" y="{y(0) + 18:.1f}" {FONT} font-size="11" '
                 f'fill="{c["muted"]}" text-anchor="middle" '
                 f'style="font-variant-numeric: tabular-nums">{m}</text>')
    e.append(label(PL + PW / 2, H - 8, "M (bindings per document, log₂ spacing)",
                   c["muted"], anchor="middle", size=11, weight="400"))
    # chance reference (neutral, dashed — a reference, not a series)
    e.append(f'<path d="{path(CHANCE)}" fill="none" stroke="{c["muted"]}" stroke-width="1.5" '
             f'stroke-dasharray="4 3"/>')
    e.append(label(x(128) + 10, y(1 / 128) + 4, "chance = 1/M", c["muted"], size=11, weight="400"))
    # series (draw the headline last so it sits on top)
    e.append(series(NOSUP, c["s3"], c["surface"]))
    e.append(series(NAIVE, c["s2"], c["surface"]))
    e.append(series(RECORD, c["s1"], c["surface"]))
    # direct labels (identity is never color-alone)
    e.append(label(x(128) + 10, y(0.929) - 2, "pk + addr-sup", c["ink"]))
    e.append(label(x(128) + 10, y(0.929) + 12, "0.929 at M=128", c["sec"], size=11, weight="400"))
    e.append(label(x(8) + 10, y(0.840) - 10, "naive recurrent", c["ink"]))
    e.append(label(x(8) + 10, y(0.360) - 10, "product-key, no addr-sup", c["ink"]))
    e.append(label(x(16) + 8, y(0.025) - 18, "collapse to ≈ chance past M=8–12",
                   c["sec"], size=11, weight="400"))
    e.append("</svg>")
    return "\n".join(e) + "\n"


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    out = os.path.join(os.path.dirname(here), "docs", "assets")
    os.makedirs(out, exist_ok=True)
    for mode in MODES:
        p = os.path.join(out, f"carry-vs-m-{mode}.svg")
        with open(p, "w") as f:
            f.write(build(mode))
        print(f"wrote {p}")


if __name__ == "__main__":
    main()
