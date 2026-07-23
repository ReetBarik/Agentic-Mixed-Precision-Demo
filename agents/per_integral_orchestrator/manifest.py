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

# Phase-2a fan-out failure-mode prefixes the Patcher stamps into an intent's error
# detail (persisted to ``errors/iter_<id>.txt``).  The manifest classifies a
# rejected decision to the finest mode it can recover — these first, else the coarse
# patcher_status — so a run's forensic record distinguishes a genuine build/compile
# failure from a fan-out wiring bug without re-running anything.
_FANOUT_FAILURE_PREFIXES = (
    "call_graph_build_failed",
    "variant_name_collision",
    "rename_cascade_incomplete",
    "silent_bypass",
    # Phase 2c: an empty promotion payload (region body byte-identical to original).
    "promotion_no_op",
)


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


def _classify_failure_mode(rec: dict, errors_dir: Path | None) -> str | None:
    """Finest failure mode for a rejected/failed decision, or ``None`` if accepted.

    Reads the Patcher's per-iter error excerpt (``errors/iter_<id>.txt``) when
    reachable and scans it for a Phase-2a fan-out prefix
    (:data:`_FANOUT_FAILURE_PREFIXES`); falls back to the coarse ``patcher_status``.
    Strategy is untouched, so the fine-grained detail lives only in that excerpt —
    this recovers it for the manifest without any Strategy-side logging change.
    """
    if bool(rec.get("accepted")):
        return None
    status = rec.get("patcher_status")
    iter_id = rec.get("iter_id")
    if errors_dir is not None and iter_id is not None:
        excerpt = errors_dir / f"iter_{iter_id}.txt"
        if excerpt.is_file():
            try:
                text = excerpt.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                text = ""
            for prefix in _FANOUT_FAILURE_PREFIXES:
                if prefix in text:
                    return prefix
    return status


def _decision_from_iter(rec: dict, integral: str,
                        settled: dict[tuple, str],
                        errors_dir: Path | None = None) -> dict:
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
        # Phase-2 followup (BIN4 addendum): preserve the Patcher status + Validator
        # reason verbatim, plus the finest fan-out failure mode we can recover, so
        # the new modes (call_graph_build_failed / variant_name_collision /
        # rename_cascade_incomplete / silent_bypass) surface in the manifest.
        "patcher_status": rec.get("patcher_status"),
        "verdict_reason": rec.get("verdict_reason"),
        "failure_mode": _classify_failure_mode(rec, errors_dir),
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
    # The Patcher writes per-iter error excerpts to ``errors/`` beside the iteration
    # log; use them to recover the fine-grained fan-out failure mode.
    errors_dir = (Path(iter_log_path).parent / "errors") if iter_log_path else None
    decisions = [_decision_from_iter(r, integral, settled, errors_dir) for r in iters]

    n_accept = sum(1 for d in decisions if d["accepted"])
    n_reject = len(decisions) - n_accept

    # Tally the finest failure mode across rejected decisions (fan-out modes first,
    # else the coarse patcher_status) — the at-a-glance forensic summary of a pass.
    failure_modes: dict[str, int] = {}
    for d in decisions:
        if not d["accepted"] and d.get("failure_mode"):
            fm = d["failure_mode"]
            failure_modes[fm] = failure_modes.get(fm, 0) + 1

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
        "failure_modes": failure_modes,
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
