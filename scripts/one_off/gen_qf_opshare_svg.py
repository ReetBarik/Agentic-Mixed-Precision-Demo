#!/usr/bin/env python3
"""Render the op-share pie charts for QF_INTEGRATION_2026-08-13.md.

Reads the SAME source data the report's op-share table is derived from — per-region
``ops`` counters in the characterization report, summed per integral and grouped by
each strategy run's ``tu_routing`` — and emits a single self-contained SVG with one
pie per run (control, then the qf run).

Deliberately dependency-free: this box has no matplotlib/numpy/node, and a committed
SVG renders on GitHub and in any local markdown previewer, which a mermaid block does
not.  Re-run after any routing change; the numbers are never hand-edited into the SVG.

Usage:
    python scripts/one_off/gen_qf_opshare_svg.py [--check]

``--check`` recomputes and prints the shares without writing the file.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_QCD = _REPO / "runs" / "qcdloop"

CHARACTERIZATION = _QCD / "report_smoke.json"
RUNS = [
    ("control", "qf rung absent", _QCD / "strategy" / "20260813_185708_0d26ef7e" / "report.json"),
    ("qf run", "qf rung present", _QCD / "strategy" / "20260813_185337_237840a2" / "report.json"),
]
OUT = _QCD / "qf_op_share.svg"

# Ladder order, cheapest first.  Colors are categorical slots from the dataviz
# reference palette, used verbatim (the validator needs node, absent here) —
# double/ff/qf are slots 1/2/3, which the palette documents as all-pairs safe in
# both modes; dd takes slot 5 (magenta) rather than slot 4 (yellow), because
# yellow-beside-orange is the one documented failing pair.
LADDER = ["float", "ff", "double", "qf", "dd"]
FILL_LIGHT = {"float": "#4a3aa7", "ff": "#eb6834", "double": "#2a78d6",
              "qf": "#1baf7a", "dd": "#e87ba4"}
FILL_DARK = {"float": "#9085e9", "ff": "#d95926", "double": "#3987e5",
             "qf": "#199e70", "dd": "#d55181"}

# Draw order within a pie: biggest first from 12 o'clock, so the slice that changes
# identity between the two runs lands in the same wedge position in both.
DRAW_ORDER = ["double", "ff", "qf", "dd", "float"]

# Layout.  Generous margins are load-bearing, not taste: the first cut of this chart
# cleared the subtitle by 0.2px and the neighbouring pie's label by 1px, which no
# rasteriser would have honoured.  _assert_layout below re-checks on every run.
W, H = 1000, 444
CY, R, LAB_GAP = 248.0, 118.0, 26.0
CENTRES = [235.0, 735.0]
Y_TITLE, Y_SUB = 28, 52                          # 24px apart: clears MIN_GAP honestly
Y_LEG_RECT, Y_LEG_TEXT, Y_CAP = 396, 406, 432
MIN_GAP = 8.0                                    # px of clear space demanded anywhere


def op_totals_per_integral() -> dict[str, int]:
    doc = json.loads(CHARACTERIZATION.read_text())
    out: dict[str, int] = {}
    for name, body in doc["integrals"].items():
        regions = body["regions"]
        regions = list(regions.values()) if isinstance(regions, dict) else regions
        out[name] = sum(c for r in regions for c in (r.get("ops") or {}).values())
    return out


def shares(per_integral: dict[str, int], routing: dict[str, str]) -> dict[str, int]:
    by = {rung: 0 for rung in LADDER}
    for integral, total in per_integral.items():
        by[routing[integral]] += total
    return by


def _arc(cx: float, cy: float, r: float, a0: float, a1: float) -> str:
    """Pie wedge path from angle a0 to a1 (degrees, 0 = 12 o'clock, clockwise)."""
    if a1 - a0 >= 359.999:                       # full circle needs two arcs
        return (f"M {cx:.2f} {cy - r:.2f} "
                f"A {r} {r} 0 1 1 {cx:.2f} {cy + r:.2f} "
                f"A {r} {r} 0 1 1 {cx:.2f} {cy - r:.2f} Z")
    t0, t1 = math.radians(a0 - 90), math.radians(a1 - 90)
    x0, y0 = cx + r * math.cos(t0), cy + r * math.sin(t0)
    x1, y1 = cx + r * math.cos(t1), cy + r * math.sin(t1)
    large = 1 if (a1 - a0) > 180 else 0
    return (f"M {cx:.2f} {cy:.2f} L {x0:.2f} {y0:.2f} "
            f"A {r:.2f} {r:.2f} 0 {large} 1 {x1:.2f} {y1:.2f} Z")


def _box(text: str, x: float, y: float, size: float, anchor: str) -> tuple:
    """Conservative bounding box for a text run (advance ~0.58em, ascent ~0.78em)."""
    w = len(text) * size * 0.58
    x0 = x if anchor == "start" else (x - w if anchor == "end" else x - w / 2)
    return (text, x0, x0 + w, y - size * 0.78, y + size * 0.24)


def pie(cx: float, cy: float, by: dict[str, int], total: int,
        boxes: list) -> list[str]:
    out, angle = [], 0.0
    for rung in DRAW_ORDER:
        v = by.get(rung, 0)
        if v <= 0:
            continue                              # a 0% slice has no wedge to draw
        sweep = 360.0 * v / total
        # 2px surface-colored stroke = the mandated gap between adjacent fills.
        out.append(f'    <path d="{_arc(cx, cy, R, angle, angle + sweep)}" '
                   f'fill="var(--s-{rung})" stroke="var(--surface)" stroke-width="2"/>')
        mid = math.radians(angle + sweep / 2 - 90)
        lx, ly = cx + (R + LAB_GAP) * math.cos(mid), cy + (R + LAB_GAP) * math.sin(mid)
        anchor = "start" if lx >= cx else "end"
        label = f"{rung} {100.0 * v / total:.2f}%"
        boxes.append(_box(label, lx, ly, 13.0, anchor))
        # Labels wear text ink, never the series color.
        out.append(f'    <text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" '
                   f'class="lbl">{rung} <tspan class="pct">'
                   f'{100.0 * v / total:.2f}%</tspan></text>')
        angle += sweep
    return out


def _assert_layout(boxes: list) -> None:
    """Fail loudly on overflow or near-collision — the eyeball check this box can't do."""
    problems = []
    for t, x0, x1, y0, y1 in boxes:
        if x0 < 0 or x1 > W or y0 < 0 or y1 > H:
            problems.append(f"overflow: {t!r} x[{x0:.1f},{x1:.1f}] y[{y0:.1f},{y1:.1f}]")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            a, b = boxes[i], boxes[j]
            gx = max(a[1], b[1]) - min(a[2], b[2])   # >0 means separated on x
            gy = max(a[3], b[3]) - min(a[4], b[4])
            if gx < MIN_GAP and gy < MIN_GAP:
                problems.append(f"collision: {a[0]!r} vs {b[0]!r} "
                                f"(gap x={gx:.1f} y={gy:.1f}, need {MIN_GAP})")
    if problems:
        raise SystemExit("LAYOUT CHECK FAILED\n  " + "\n  ".join(problems))


def build_svg(per_integral: dict[str, int]) -> str:
    total = sum(per_integral.values())
    panels, boxes = [], []
    for (title, sub, path), cx in zip(RUNS, CENTRES):
        routing = json.loads(path.read_text())["tu_routing"]
        by = shares(per_integral, routing)
        panels.append(f'    <text x="{cx}" y="{Y_TITLE}" text-anchor="middle" '
                      f'class="ttl">{title}</text>')
        panels.append(f'    <text x="{cx}" y="{Y_SUB}" text-anchor="middle" '
                      f'class="sub">{sub}</text>')
        boxes.append(_box(title, cx, Y_TITLE, 17.0, "middle"))
        boxes.append(_box(sub, cx, Y_SUB, 12.5, "middle"))
        panels.extend(pie(cx, CY, by, total, boxes))

    # Legend carries all five rungs, including the two at 0% — the zeros are the
    # point, and a pie cannot draw them.
    routing_qf = json.loads(RUNS[1][2].read_text())["tu_routing"]
    by_qf = shares(per_integral, routing_qf)
    entries = [(r, f"{r} {100.0 * by_qf.get(r, 0) / total:.2f}%") for r in LADDER]
    widths = [19 + len(t) * 12.5 * 0.58 for _r, t in entries]
    span = sum(widths) + 26 * (len(entries) - 1)
    legend, x = [], (W - span) / 2
    for (rung, text), w in zip(entries, widths):
        pct = 100.0 * by_qf.get(rung, 0) / total
        legend.append(f'    <rect x="{x:.1f}" y="{Y_LEG_RECT}" width="12" height="12" '
                      f'rx="3" fill="var(--s-{rung})"/>')
        legend.append(f'    <text x="{x + 19:.1f}" y="{Y_LEG_TEXT}" class="leg">{rung} '
                      f'<tspan class="legp">{pct:.2f}%</tspan></text>')
        boxes.append(_box(text, x, Y_LEG_TEXT, 12.5, "start"))
        x += w + 26

    _assert_layout(boxes)

    css_light = "\n".join(f"      --s-{k}: {v};" for k, v in FILL_LIGHT.items())
    css_dark = "\n".join(f"        --s-{k}: {v};" for k, v in FILL_DARK.items())
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"
     width="{W}" height="{H}" role="img"
     aria-label="Math ops by precision rung. Control: double 51.53 percent, ff 38.36
     percent, dd 10.11 percent. QF run: double 51.53 percent, ff 38.36 percent, qf
     10.11 percent. Total {total:,} ops.">
  <style>
    :root {{
      --surface: #fcfcfb;
      --ink: #0b0b0b;
      --ink2: #52514e;
{css_light}
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --surface: #1a1a19;
        --ink: #ffffff;
        --ink2: #c3c2b7;
{css_dark}
      }}
    }}
    text {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif; }}
    .ttl  {{ font-size: 17px; font-weight: 600; fill: var(--ink); }}
    .sub  {{ font-size: 12.5px; fill: var(--ink2); }}
    .lbl  {{ font-size: 13px; font-weight: 600; fill: var(--ink); }}
    .pct  {{ font-weight: 400; fill: var(--ink2); }}
    .leg  {{ font-size: 12.5px; fill: var(--ink); }}
    .legp {{ fill: var(--ink2); }}
    .cap  {{ font-size: 12px; fill: var(--ink2); }}
  </style>
  <rect width="{W}" height="{H}" fill="var(--surface)"/>
{chr(10).join(panels)}
{chr(10).join(legend)}
  <text x="{W // 2}" y="{Y_CAP}" text-anchor="middle" class="cap">Share of {total:,} math ops — legend shows the qf run. Op counts, not cost; all 21 integrals weighted equally.</text>
</svg>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="print shares, write nothing")
    args = ap.parse_args()

    per_integral = op_totals_per_integral()
    total = sum(per_integral.values())
    for title, _sub, path in RUNS:
        by = shares(per_integral, json.loads(path.read_text())["tu_routing"])
        cells = "  ".join(f"{r}={100.0 * by[r] / total:5.2f}%" for r in LADDER)
        print(f"{title:8} {cells}")
    print(f"total ops: {total:,}")

    if not args.check:
        OUT.write_text(build_svg(per_integral))
        print(f"wrote {OUT.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
