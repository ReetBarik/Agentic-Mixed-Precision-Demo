"""Build a self-contained per-integral pass manifest.

The manifest is the forensic record of one ``run_per_integral_pass``: everything
needed to reconstruct *what the pass did* — which intents were attempted, which
were accepted vs rejected, the settled precision per region, and which files were
touched — **without re-running** anything.

Sources (all already produced by the untouched Strategy agent):

* ``iterations.jsonl`` (``iteration_log_path``) — the authoritative append-only
  per-iteration trail (one record per attempt, accepted or rejected), carrying the
  target span, kind, intent, phase and validator verdict.  This is the accept/reject
  list.
* the fat strategy ``report.json`` (``report_json_path``) — ``precision_assignment``
  (settled precision per retained region), ``precision_distribution``,
  ``correctness_summary`` (ceiling regions) and the run status.
* the cumulative candidate diff (``cumulative_diff_path``) — the set of modified
  files (``+++ b/<path>`` headers).
"""

from __future__ import annotations

import json
from pathlib import Path


def _read_jsonl(path: str | Path) -> list[dict]:
    p = Path(path)
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def _modified_files(diff_path: str | Path | None) -> list[str]:
    """Paths from a unified diff's ``+++ b/<path>`` headers (deduped, ordered)."""
    if not diff_path:
        return []
    p = Path(diff_path)
    if not p.is_file():
        return []
    files: list[str] = []
    seen: set[str] = set()
    for line in p.read_text().splitlines():
        if line.startswith("+++ "):
            path = line[4:].strip()
            if path.startswith("b/"):
                path = path[2:]
            if path and path != "/dev/null" and path not in seen:
                seen.add(path)
                files.append(path)
    return files


def _decision_from_iter(rec: dict, integral: str,
                        settled: dict[tuple, str]) -> dict:
    """One manifest decision row from an ``iterations.jsonl`` record."""
    tgt = rec.get("target", {}) or {}
    file = tgt.get("file")
    line_start = tgt.get("line_start")
    line_end = tgt.get("line_end", line_start)
    accepted = bool(rec.get("accepted"))
    if accepted:
        verdict = "accept"
    else:
        # a genuine dd-ceiling retain is logged as not-accepted but keeps its
        # precision; surface the validator verdict so a reject is distinguishable
        # from a ceiling retain downstream.
        verdict = rec.get("validator_verdict") or "reject"
    key = (file, line_start, line_end)
    return {
        "file": file,
        "line": line_start,
        "line_end": line_end,
        "precision": settled.get(key, rec.get("intent") or rec.get("kind")),
        "intent": rec.get("intent"),
        "kind": rec.get("kind"),
        "verdict": verdict,
        "accepted": accepted,
        "phase": rec.get("phase"),
        "iter_id": rec.get("iter_id"),
        "integral": integral,
    }


def build_manifest(integral: str, strategy_result: dict, filter_meta: dict,
                   *, timing: dict | None = None,
                   tree_path: str | None = None) -> dict:
    """Assemble the per-integral manifest dict.

    ``strategy_result`` is the ``strategy_result`` bundle returned by
    ``agents.strategy.agent.run`` (pointers to the fat report + cumulative diff).
    ``filter_meta`` is :func:`filter_report`'s return.  ``timing`` folds in
    wall/iteration bookkeeping the orchestrator measures.
    """
    report_json_path = strategy_result.get("report_json_path")
    fat: dict = {}
    if report_json_path and Path(report_json_path).is_file():
        fat = json.loads(Path(report_json_path).read_text())

    # settled precision per retained region (final, post-Validator)
    settled: dict[tuple, str] = {}
    for a in fat.get("precision_assignment", []) or []:
        key = (a.get("file"), a.get("line_start"), a.get("line_end"))
        settled[key] = a.get("precision")

    iter_log_path = fat.get("iteration_log_path")
    iters = _read_jsonl(iter_log_path) if iter_log_path else []
    decisions = [_decision_from_iter(r, integral, settled) for r in iters]

    n_accept = sum(1 for d in decisions if d["accepted"])
    n_reject = len(decisions) - n_accept

    return {
        "kind": "per_integral_manifest",
        "integral": integral,
        "status": strategy_result.get("status") or fat.get("status"),
        "run_id": strategy_result.get("run_id") or fat.get("run_id"),
        "final_branch": strategy_result.get("final_branch"),
        "filter": filter_meta,
        "counts": {
            "intents_attempted": len(decisions),
            "accepted": n_accept,
            "rejected": n_reject,
        },
        "precision_distribution": fat.get("precision_distribution", {}),
        "correctness_summary": fat.get("correctness_summary", {}),
        "speedup_summary": fat.get("speedup_summary", {}),
        "iterations": fat.get("iterations"),
        "duration_sec": fat.get("duration_sec"),
        "modified_files": _modified_files(
            strategy_result.get("cumulative_diff_path")),
        "decisions": decisions,
        "artifacts": {
            "tree_path": tree_path,
            "report_json_path": report_json_path,
            "report_md_path": strategy_result.get("report_md_path"),
            "cumulative_diff_path": strategy_result.get("cumulative_diff_path"),
            "iteration_log_path": iter_log_path,
        },
        "timing": timing or {},
    }
