#!/usr/bin/env python3
"""Render the precision-assignment charts for the qcdloop workload.

Three panels in one SVG, all describing the CURRENT routing (no before/after):

  top-left   share of INTEGRALS by rung          — 21 equal wedges, one per integral
  top-right  share of math OPS by precision rung — wedge size = op count
  bottom     expected per-integral SPEEDUP on an NVIDIA GB300

The two pies read together are the first point: the same three rungs, weighted two
different ways.  The five double-routed integrals are 23.8% of the integrals but
51.5% of the ops, and a rung-share pie alone cannot show that.

The bar panel is the second, sharper point: op share is not cost share either.  Bar
heights come from gb300_cost_model, which weights each op by what it actually costs
(counted from the vendored emulation headers) against GB300's ~62:1 FP32:FP64 vector
ratio.  Weighted that way the picture inverts — log and atan2 are ~3% of ops but
~69% of modelled time, so the transcendental-heavy integrals barely gain (B15 1.38x)
while the arithmetic-heavy ones do well (B6 3.61x).

Bars are quoted against an ISO-ACCURACY baseline: the cheapest rung clearing the same
7.0-digit tolerance gate.  That is double for 19 integrals, but dd for B12/B16, which
score 3.69 and 6.57 digits at double and so FAIL there — an all-double baseline for
those two would compare against a run you would have to throw away.  Because that
makes their bars the tallest in the chart while measuring something different, each
carries a dashed GHOST bar at its vs-double value (0.23x, 0.16x): the same flip read
as a pure cost, with the accuracy it buys not counted.  Both readings are true and
the pair is the honest way to show it.

The bottom panel is full-width rather than a third column: 21 bars across a third of
1140px is a ~18px pitch, which cannot carry readable category labels.

Op counts come from the per-region ``ops`` counters in the characterization report,
summed per integral and grouped by the run's ``tu_routing``.  Valid because tu_only
flips a whole TU, so every op in an integral executes at that integral's rung.

COLOR MUST STAY ON THE ELEMENTS.  Every fill/stroke is a literal hex *presentation
attribute*; the stylesheet only carries the dark-mode override.  An earlier version
put the palette in CSS custom properties (``fill="var(--s-double)"``) and rendered as
a SOLID BLACK BOX anywhere custom properties are unsupported — PowerPoint's importer,
and librsvg here — because an unresolvable fill falls back to the initial value, which
is black.  Presentation attributes sit at the bottom of the CSS cascade, so browsers
still apply the dark-mode rules, while importers that ignore CSS entirely (including
PowerPoint's "Convert to Shape") get correct light-theme colors.  Do not reintroduce
var() here.

Deliberately dependency-free: this box has no matplotlib/numpy/node, and a committed
SVG renders on GitHub and in any local markdown previewer, which a mermaid block does
not.  Numbers are read from source on every run and never hand-edited into the file.

Usage:
    python scripts/one_off/gen_op_share_svg.py [--check]

``--check`` recomputes and prints both distributions without writing the file.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gb300_cost_model as gb300           # noqa: E402  (needs the path above)

_REPO = Path(__file__).resolve().parents[2]
_QCD = _REPO / "runs" / "qcdloop"

CHARACTERIZATION = _QCD / "report_smoke.json"
RUN_REPORT = _QCD / "strategy" / "20260813_185337_237840a2" / "report.json"
OUT = _QCD / "op_share.svg"

# Ladder order, cheapest first.  Colors are categorical slots from the dataviz
# reference palette, used verbatim (its validator needs node, absent here) —
# double/ff/qf are slots 1/2/3, which the palette documents as all-pairs safe in
# both modes; dd takes slot 5 (magenta) rather than slot 4 (yellow), because
# yellow-beside-orange is the one documented failing pair.
LADDER = ["float", "ff", "double", "qf", "dd"]
FILL_LIGHT = {"float": "#4a3aa7", "ff": "#eb6834", "double": "#2a78d6",
              "qf": "#1baf7a", "dd": "#e87ba4"}
FILL_DARK = {"float": "#9085e9", "ff": "#d95926", "double": "#3987e5",
             "qf": "#199e70", "dd": "#d55181"}
SURFACE_L, INK_L, INK2_L = "#fcfcfb", "#0b0b0b", "#52514e"
SURFACE_D, INK_D, INK2_D = "#1a1a19", "#ffffff", "#c3c2b7"

# PowerPoint on Windows resolves Segoe UI; the -apple-system / BlinkMacSystemFont
# keywords are meaningless to it, so lead with a real family name.
FONT = "'Segoe UI', Helvetica, Arial, sans-serif"

# Wedge order in BOTH panels, so a rung starts at the same clock position in each
# and the reader can see its arc grow or shrink between them.
DRAW_ORDER = ["double", "ff", "qf", "dd", "float"]

# Layout.  Generous margins are load-bearing, not taste: an early cut of this chart
# cleared its subtitle by 0.2px and the neighbouring panel's label by 1px, which no
# rasteriser would have honoured.  _assert_layout re-checks on every run.
W, H = 1140, 848
CY, R, LAB_GAP = 252.0, 118.0, 20.0
CENTRES = [260.0, 880.0]
Y_TITLE, Y_SUB = 28, 52
Y_LEG_RECT, Y_LEG_TEXT, Y_CAP = 412, 422, 452
MIN_GAP = 8.0                                    # px of clear space demanded anywhere

# Bottom (speedup) panel.
Y_T3, Y_S3 = 512, 536                            # title / subtitle
BAR_TOP, BAR_BASE = 560, 770                     # plot box; BAR_BASE is y=0
Y_XLAB, Y_CAP3, Y_CAP4 = 788, 809, 831
PAD_L, PAD_R = 62.0, 24.0
Y_MAX = 8.0                                      # axis top, comfortably over B12@7.67
GRID = [0, 2, 4, 6, 8]
BAR_W, PAIR_GAP = 20.0, 6.0                      # a qf slot holds two BAR_W bars


def _natural(name: str) -> tuple:
    """B1 < B2 < B10 < B16 < BIN0 < BIN4 — deterministic wedge order."""
    m = re.match(r"^([A-Za-z]+)(\d+)$", name)
    return (m.group(1), int(m.group(2))) if m else (name, 0)


def op_totals_per_integral() -> dict[str, int]:
    doc = json.loads(CHARACTERIZATION.read_text())
    out: dict[str, int] = {}
    for name, body in doc["integrals"].items():
        regions = body["regions"]
        regions = list(regions.values()) if isinstance(regions, dict) else regions
        out[name] = sum(c for r in regions for c in (r.get("ops") or {}).values())
    return out


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


def _wedge(cx: float, a0: float, a1: float, rung: str) -> str:
    # 2px surface-colored stroke = the mandated gap between adjacent fills.  In the
    # integrals panel it is also what makes the 21 individual wedges countable.
    return (f'    <path d="{_arc(cx, CY, R, a0, a1)}" class="s-{rung} sep" '
            f'fill="{FILL_LIGHT[rung]}" stroke="{SURFACE_L}" stroke-width="2"/>')


def _arc_label(cx: float, a0: float, a1: float, text: str, boxes: list) -> str:
    mid = math.radians((a0 + a1) / 2 - 90)
    lx = cx + (R + LAB_GAP) * math.cos(mid)
    ly = CY + (R + LAB_GAP) * math.sin(mid)
    anchor = "start" if lx >= cx else "end"
    boxes.append(_box(text, lx, ly, 13.0, anchor))
    head, _, tail = text.partition(" ")
    # Labels wear text ink, never the series color.
    return (f'    <text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" class="ink" '
            f'font-size="13" font-weight="600" fill="{INK_L}">{head} '
            f'<tspan class="ink2" font-weight="400" fill="{INK2_L}">{tail}</tspan></text>')


def panel_ops(cx: float, by: dict[str, int], total: int, boxes: list) -> list[str]:
    """Wedge size proportional to op count."""
    out, angle = [], 0.0
    for rung in DRAW_ORDER:
        v = by.get(rung, 0)
        if v <= 0:
            continue                              # a 0% rung has no wedge to draw
        sweep = 360.0 * v / total
        out.append(_wedge(cx, angle, angle + sweep, rung))
        out.append(_arc_label(cx, angle, angle + sweep,
                              f"{rung} {100.0 * v / total:.2f}%", boxes))
        angle += sweep
    return out


def panel_integrals(cx: float, routing: dict[str, str], boxes: list) -> list[str]:
    """One equal wedge per integral, colored by its rung."""
    n = len(routing)
    step = 360.0 / n
    out, angle = [], 0.0
    for rung in DRAW_ORDER:
        members = sorted((k for k, v in routing.items() if v == rung), key=_natural)
        if not members:
            continue
        start = angle
        for _ in members:                         # one wedge each, all the same size
            out.append(_wedge(cx, angle, angle + step, rung))
            angle += step
        out.append(_arc_label(cx, start, angle,
                              f"{rung} ×{len(members)} "
                              f"{100.0 * len(members) / n:.2f}%", boxes))
    return out


def _bar_y(v: float) -> float:
    return BAR_BASE - (v / Y_MAX) * (BAR_BASE - BAR_TOP)


def _bar_path(x: float, w: float, v: float) -> str:
    """Bar with 4px rounded top corners, square where it meets the baseline.

    rx on a <rect> would round the BOTTOM corners too, which detaches the mark from
    its own axis.  The radius clamps on short bars so the two ghosts (0.16x, 0.23x)
    stay rectangles instead of collapsing to lozenges.
    """
    top = _bar_y(v)
    r = min(4.0, max(0.0, (BAR_BASE - top) / 2.0))
    return (f"M {x:.1f} {BAR_BASE:.1f} L {x:.1f} {top + r:.1f} "
            f"Q {x:.1f} {top:.1f} {x + r:.1f} {top:.1f} "
            f"L {x + w - r:.1f} {top:.1f} "
            f"Q {x + w:.1f} {top:.1f} {x + w:.1f} {top + r:.1f} "
            f"L {x + w:.1f} {BAR_BASE:.1f} Z")


def panel_speedup(speed: dict, boxes: list) -> list:
    """Per-integral iso-accuracy speedup; qf slots carry a dashed vs-double ghost."""
    names = sorted(speed, key=_natural)
    pitch = (W - PAD_L - PAD_R) / len(names)
    out = []

    # Grid first, so bars paint over it.  The 1.0x line is the break-even reference
    # and is drawn heavier than the rest — below it a flip costs more than it saves.
    for g in GRID:
        y = _bar_y(g)
        out.append(f'    <line x1="{PAD_L:.1f}" y1="{y:.1f}" x2="{W - PAD_R:.1f}" '
                   f'y2="{y:.1f}" class="grid" stroke="{INK2_L}" stroke-width="1" '
                   f'stroke-opacity="0.18"/>')
        out.append(f'    <text x="{PAD_L - 10:.1f}" y="{y + 4:.1f}" text-anchor="end" '
                   f'class="ink2" font-size="11.5" fill="{INK2_L}">{g}&#215;</text>')
        boxes.append(_box(f"{g}x", PAD_L - 10, y + 4, 11.5, "end"))
    y1 = _bar_y(1.0)
    out.append(f'    <line x1="{PAD_L:.1f}" y1="{y1:.1f}" x2="{W - PAD_R:.1f}" '
               f'y2="{y1:.1f}" class="grid" stroke="{INK_L}" stroke-width="1.25" '
               f'stroke-dasharray="5 4" stroke-opacity="0.55"/>')
    out.append(f'    <text x="{W - PAD_R:.1f}" y="{y1 - 7:.1f}" text-anchor="end" '
               f'class="ink2" font-size="11" fill="{INK2_L}">1&#215; break-even</text>')
    boxes.append(_box("1x break-even", W - PAD_R, y1 - 7, 11.0, "end"))

    ffs = {n: speed[n]["iso"] for n in names if speed[n]["rung"] == "ff"}
    # Selective direct labels only: both qf bars (they are the ones measuring
    # something different), plus the ff extremes that carry the panel's message.
    label = {max(ffs, key=ffs.get), min(ffs, key=ffs.get)}

    for i, n in enumerate(names):
        rec = speed[n]
        rung, iso = rec["rung"], rec["iso"]
        cx = PAD_L + (i + 0.5) * pitch
        paired = rung == "qf"
        x = cx - (BAR_W + PAIR_GAP / 2 if paired else BAR_W / 2)

        out.append(f'    <path d="{_bar_path(x, BAR_W, iso)}" class="s-{rung}" '
                   f'fill="{FILL_LIGHT[rung]}"/>')
        if paired or n in label:
            out.append(f'    <text x="{x + BAR_W / 2:.1f}" y="{_bar_y(iso) - 7:.1f}" '
                       f'text-anchor="middle" class="ink" font-size="11.5" '
                       f'font-weight="600" fill="{INK_L}">{iso:.2f}&#215;</text>')
            boxes.append(_box(f"{iso:.2f}x", x + BAR_W / 2, _bar_y(iso) - 7, 11.5,
                              "middle"))

        if paired:
            gx, gv = cx + PAIR_GAP / 2, rec["vs_double"]
            out.append(f'    <path d="{_bar_path(gx, BAR_W, gv)}" class="k-{rung}" '
                       f'fill="none" stroke="{FILL_LIGHT[rung]}" stroke-width="1.5" '
                       f'stroke-dasharray="3 2"/>')
            out.append(f'    <text x="{gx + BAR_W / 2:.1f}" y="{_bar_y(gv) - 7:.1f}" '
                       f'text-anchor="middle" class="ink2" font-size="10.5" '
                       f'fill="{INK2_L}">{gv:.2f}&#215;</text>')
            boxes.append(_box(f"{gv:.2f}x", gx + BAR_W / 2, _bar_y(gv) - 7, 10.5,
                              "middle"))

        out.append(f'    <text x="{cx:.1f}" y="{Y_XLAB}" text-anchor="middle" '
                   f'class="ink2" font-size="11" fill="{INK2_L}">{n}</text>')
        boxes.append(_box(n, cx, Y_XLAB, 11.0, "middle"))

    out.append(f'    <line x1="{PAD_L:.1f}" y1="{BAR_BASE:.1f}" x2="{W - PAD_R:.1f}" '
               f'y2="{BAR_BASE:.1f}" class="grid" stroke="{INK2_L}" '
               f'stroke-width="1.25" stroke-opacity="0.5"/>')
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


def build_svg(per_integral: dict[str, int], routing: dict[str, str],
              speed: dict) -> str:
    total = sum(per_integral.values())
    n = len(routing)
    by_ops: dict[str, int] = {r: 0 for r in LADDER}
    for integral, ops in per_integral.items():
        by_ops[routing[integral]] += ops
    n_double = sum(1 for v in routing.values() if v == "double")

    body, boxes = [], []
    panels = [
        ("integrals", f"{n} integrals · one equal wedge each",
         lambda cx: panel_integrals(cx, routing, boxes)),
        ("math operations", f"{total:,} ops · wedge = op count",
         lambda cx: panel_ops(cx, by_ops, total, boxes)),
    ]
    for (title, sub, draw), cx in zip(panels, CENTRES):
        body.append(f'    <text x="{cx}" y="{Y_TITLE}" text-anchor="middle" class="ink" '
                    f'font-size="17" font-weight="600" fill="{INK_L}">{title}</text>')
        body.append(f'    <text x="{cx}" y="{Y_SUB}" text-anchor="middle" class="ink2" '
                    f'font-size="12.5" fill="{INK2_L}">{sub}</text>')
        boxes.append(_box(title, cx, Y_TITLE, 17.0, "middle"))
        boxes.append(_box(sub, cx, Y_SUB, 12.5, "middle"))
        body.extend(draw(cx))

    whole = (sum(v["t_base"] for v in speed.values())
             / sum(v["t_now"] for v in speed.values()))
    t3 = "expected speedup on NVIDIA GB300"
    s3 = (f"iso-accuracy baseline · whole app {whole:.2f}× · "
          f"dashed ghost = qf measured against double instead of dd")
    body.append(f'    <text x="{W / 2}" y="{Y_T3}" text-anchor="middle" class="ink" '
                f'font-size="17" font-weight="600" fill="{INK_L}">{t3}</text>')
    body.append(f'    <text x="{W / 2}" y="{Y_S3}" text-anchor="middle" class="ink2" '
                f'font-size="12.5" fill="{INK2_L}">{s3}</text>')
    boxes.append(_box(t3, W / 2, Y_T3, 17.0, "middle"))
    boxes.append(_box(s3, W / 2, Y_S3, 12.5, "middle"))
    body.extend(panel_speedup(speed, boxes))

    # Legend spans the full ladder.  float and dd carry an explicit 0.00% — they are
    # zero in BOTH panels, so the number is unambiguous; the three rungs in use are
    # labeled on their own arcs, where the two panels disagree.
    entries = [(r, r if by_ops[r] else f"{r} 0.00%") for r in LADDER]
    widths = [19 + len(t) * 12.5 * 0.58 for _r, t in entries]
    span = sum(widths) + 30 * (len(entries) - 1)
    legend, x = [], (W - span) / 2
    for (rung, text), w in zip(entries, widths):
        legend.append(f'    <rect x="{x:.1f}" y="{Y_LEG_RECT}" width="12" height="12" '
                      f'rx="3" class="s-{rung}" fill="{FILL_LIGHT[rung]}"/>')
        tail = ("" if by_ops[rung] else
                f' <tspan class="ink2" fill="{INK2_L}">0.00%</tspan>')
        legend.append(f'    <text x="{x + 19:.1f}" y="{Y_LEG_TEXT}" class="ink" '
                      f'font-size="12.5" fill="{INK_L}">{rung}{tail}</text>')
        boxes.append(_box(text, x, Y_LEG_TEXT, 12.5, "start"))
        x += w + 30

    cap = (f"The {n_double} double-routed integrals are "
           f"{100.0 * n_double / n:.1f}% of the integrals but "
           f"{100.0 * by_ops['double'] / total:.1f}% of the ops. "
           f"Op counts, not cost; every integral sampled equally.")
    boxes.append(_box(cap, W / 2, Y_CAP, 12.0, "middle"))

    # Two lines: one 12px run of this length overflows 1140px, and _assert_layout
    # rightly refuses it.
    cap3 = ("log and atan2 are 3% of ops but 69% of modelled time, so the "
            "transcendental-heavy integrals gain least.")
    cap4 = ("Throughput model only — no occupancy or register pressure, which would "
            "cost the 4-limb qf integrals most.")
    boxes.append(_box(cap3, W / 2, Y_CAP3, 12.0, "middle"))
    boxes.append(_box(cap4, W / 2, Y_CAP4, 12.0, "middle"))

    # Screen readers get every bar; the visual panel direct-labels only four.
    aria_bars = ", ".join(
        f"{k} {speed[k]['iso']:.2f}"
        + (f" against dd, {speed[k]['vs_double']:.2f} against double"
           if speed[k]["rung"] == "qf" else "")
        for k in sorted(speed, key=_natural))

    _assert_layout(boxes)
    dark = "\n".join(f"      .s-{k} {{ fill: {v}; }}" for k, v in FILL_DARK.items())
    # Ghost bars carry the rung color as a STROKE, so the fill override above cannot
    # reach them; they need their own rule or they stay light-theme orange on dark.
    dark += "\n" + "\n".join(f"      .k-{k} {{ stroke: {v}; }}"
                             for k, v in FILL_DARK.items())
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"
     width="{W}" height="{H}" role="img"
     aria-label="Precision assignment and expected GB300 speedup for the qcdloop
     workload. Two pie charts and a bar chart. By integral count: ff 14 of {n},
     double 5 of {n}, qf 2 of {n}. By math operations: double 51.53 percent, ff 38.36
     percent, qf 10.11 percent of {total:,} ops. float and dd are unused. {cap}
     Per-integral iso-accuracy speedup on GB300, whole app {whole:.2f} times:
     {aria_bars}. {cap3}">
  <style>
    /* Light theme lives on the elements as presentation attributes, so renderers that
       ignore CSS (PowerPoint) still get real colors.  This block only overrides for
       dark mode; presentation attributes lose to any selector, so browsers apply it. */
    @media (prefers-color-scheme: dark) {{
      .bg   {{ fill: {SURFACE_D}; }}
      .sep  {{ stroke: {SURFACE_D}; }}
      .ink  {{ fill: {INK_D}; }}
      .ink2 {{ fill: {INK2_D}; }}
      .grid {{ stroke: {INK2_D}; }}
{dark}
    }}
  </style>
  <rect class="bg" width="{W}" height="{H}" fill="{SURFACE_L}"/>
  <g font-family="{FONT}">
{chr(10).join(body)}
{chr(10).join(legend)}
    <text x="{W // 2}" y="{Y_CAP}" text-anchor="middle" class="ink2" font-size="12"
          fill="{INK2_L}">{cap}</text>
    <text x="{W // 2}" y="{Y_CAP3}" text-anchor="middle" class="ink2" font-size="12"
          fill="{INK2_L}">{cap3}</text>
    <text x="{W // 2}" y="{Y_CAP4}" text-anchor="middle" class="ink2" font-size="12"
          fill="{INK2_L}">{cap4}</text>
  </g>
</svg>
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="print shares, write nothing")
    args = ap.parse_args()

    per_integral = op_totals_per_integral()
    routing = json.loads(RUN_REPORT.read_text())["tu_routing"]
    total, n = sum(per_integral.values()), len(routing)

    by_ops = {r: 0 for r in LADDER}
    by_cnt = {r: 0 for r in LADDER}
    for integral, ops in per_integral.items():
        by_ops[routing[integral]] += ops
        by_cnt[routing[integral]] += 1
    print(f"{'rung':8} {'ints':>5} {'int share':>10} {'ops':>12} {'op share':>9}")
    for rung in LADDER:
        print(f"{rung:8} {by_cnt[rung]:>5} {100.0 * by_cnt[rung] / n:9.2f}% "
              f"{by_ops[rung]:>12,} {100.0 * by_ops[rung] / total:8.2f}%")
    print(f"{'TOTAL':8} {n:>5} {'100.00%':>10} {total:>12,} {'100.00%':>9}")

    speed = gb300.per_integral_speedup()
    whole = (sum(v["t_base"] for v in speed.values())
             / sum(v["t_now"] for v in speed.values()))
    print(f"\n{'integral':9} {'rung':7} {'iso base':>9} {'iso':>8} {'vs double':>10}")
    for name in sorted(speed, key=_natural):
        rec = speed[name]
        print(f"{name:9} {rec['rung']:7} {rec['iso_base']:>9} "
              f"{rec['iso']:7.2f}x {rec['vs_double']:9.2f}x")
    print(f"{'WHOLE APP':9} {'':7} {'':>9} {whole:7.2f}x")

    if not args.check:
        OUT.write_text(build_svg(per_integral, routing, speed))
        print(f"wrote {OUT.relative_to(_REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
