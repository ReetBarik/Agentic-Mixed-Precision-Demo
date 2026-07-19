"""Output writers — ``report.json`` (full) + ``report.md`` (projection, Q4).

The JSON is the authoritative artifact (consumed by downstream agents); the
markdown is a strict projection for Reet, never carrying data the JSON lacks.
Ceiling regions get top billing — both ``dd_ceiling`` (physics limits) and
``dd_untested`` (P6a: DD never honestly tried).
"""

from __future__ import annotations

import json
from pathlib import Path

from agents.strategy.models import LADDER


def write_reports(run_dir: str | Path, report: dict) -> tuple[Path, Path]:
    """Write ``report.json`` + ``report.md`` under ``run_dir``; return both paths."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    json_path = run_dir / "report.json"
    md_path = run_dir / "report.md"
    json_path.write_text(json.dumps(report, indent=2, sort_keys=False) + "\n")
    md_path.write_text(_render_markdown(report))
    return json_path, md_path


def _render_markdown(r: dict) -> str:
    cs = r.get("correctness_summary", {})
    ceilings = cs.get("ceiling_regions", [])
    dist = r.get("precision_distribution", {})
    rewrites = r.get("algorithmic_rewrites", [])
    lines: list[str] = []

    # -- header --
    lines += [
        f"# Strategy run `{r.get('run_id', '?')}`",
        "",
        f"- **status:** {r.get('status', '?')}",
        f"- **tolerance:** {r.get('tolerance', '?')} precise digits",
        f"- **duration:** {r.get('duration_sec', '?')} s",
        f"- **starting SHA:** `{r.get('starting_sha', '?')}`",
        f"- **final branch:** `{r.get('final_branch', '?')}`",
        f"- **iterations:** {r.get('iterations', '?')} "
        f"(see `{r.get('iteration_log_path', 'iterations.jsonl')}`)",
        "",
    ]

    # -- ceiling regions (top billing) --
    lines += ["## Ceiling regions", ""]
    if not ceilings:
        lines += ["_None — every worked region reached threshold._", ""]
    else:
        lines += [
            "Regions that could not reach tolerance. `dd_ceiling` = physics limit "
            "(DD tried + rejected, rewrites exhausted); `dd_untested` = DD never "
            "honestly built (Patcher failure, P6a) — investigate, not a true limit.",
            "",
            "| location | kind | final digits | signal class | rewrites / reason |",
            "|---|---|---|---|---|",
        ]
        for c in ceilings:
            detail = (", ".join(c.get("attempted_rewrites") or [])
                      or c.get("reason", ""))
            digits = c.get("final_min_digits")
            digits_s = "—" if digits is None else f"{digits}"
            lines.append(
                f"| {c.get('location', '?')} | {c.get('ceiling_kind', '?')} | "
                f"{digits_s} | {c.get('signal_class', '?')} | {detail} |")
        lines.append("")

    # -- precision distribution --
    lines += ["## Precision distribution", "", "| precision | regions |", "|---|---|"]
    for p in LADDER:
        lines.append(f"| {p} | {dist.get(p, 0)} |")
    lines += [f"| **total** | {sum(dist.get(p, 0) for p in LADDER)} |", ""]

    # -- algorithmic rewrites --
    lines += ["## Algorithmic rewrites accepted", ""]
    accepted_rw = [w for w in rewrites if w.get("accepted")]
    if not accepted_rw:
        lines += ["_None accepted._", ""]
    else:
        lines += ["| location | kind | identity | rationale |", "|---|---|---|---|"]
        for w in accepted_rw:
            loc = _loc(w)
            lines.append(
                f"| {loc} | {w.get('kind', '?')} | {w.get('identity') or '—'} | "
                f"{w.get('rationale_id', '?')} |")
        lines.append("")

    # -- two-phase walk summary (accepts grouped by phase) --
    ps = r.get("phase_summary", {})
    if ps:
        assigns = r.get("precision_assignment", [])
        corr_acc = sum(1 for a in assigns if a.get("phase") == "correctness")
        spd_acc = sum(1 for a in assigns if a.get("phase") == "speedup")
        c, s = ps.get("correctness", {}), ps.get("speedup", {})
        lines += [
            "## Two-phase walk", "",
            "| phase | iterations | iter cap | accepts (walk) | precision assigns |",
            "|---|---|---|---|---|",
            f"| correctness | {c.get('iterations', 0)} | {c.get('iter_cap', '?')} | "
            f"{c.get('accepts', 0)} | {corr_acc} |",
            f"| speedup | {s.get('iterations', 0)} | {s.get('iter_cap', '?')} | "
            f"{s.get('accepts', 0)} | {spd_acc} |",
            "",
        ]
        skipped = s.get("skipped_dd_promoted")
        if skipped:
            lines += [f"_{skipped} region(s) promoted to dd in phase 1 were skipped "
                      f"in the speedup phase._", ""]

    # -- iteration summary --
    lines += [
        "## Iteration summary",
        "",
        f"- regions at threshold: {cs.get('regions_at_threshold', '?')}",
        f"- regions at DD ceiling: {cs.get('regions_at_dd_ceiling', '?')}",
        f"- regions DD-untested: {cs.get('regions_dd_untested', '?')}",
        f"- regions unresolved (budget/stop): {cs.get('regions_unresolved', '?')}",
        f"- precision assignments: {len(r.get('precision_assignment', []))}",
        f"- accepted rewrites: {len(accepted_rw)}",
        "",
        f"Full per-iteration log: `{r.get('iteration_log_path', 'iterations.jsonl')}`",
        "",
    ]
    return "\n".join(lines)


def _loc(record: dict) -> str:
    f = record.get("file", "?")
    a, b = record.get("line_start"), record.get("line_end")
    return f"{f}:{a}" if a == b else f"{f}:{a}-{b}"
