"""Append-only per-iteration log — ``iterations.jsonl``.

One record per iteration (accepted OR rejected), written the moment the
iteration resolves so a crashed run still leaves a forensic trail.  ``iter_id``
is a monotonic counter over every record and doubles as the ``rationale_id``
(``iter_23``) that cross-references commit messages and the final report (Q3).
"""

from __future__ import annotations

import json
from pathlib import Path


class IterationLogger:
    """Streams one JSON object per line to ``<run_dir>/iterations.jsonl``."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "iterations.jsonl"
        self._next_id = 0
        # truncate any stale file from a re-used run_id
        self._fh = open(self.path, "w", buffering=1)

    def next_iter_id(self) -> int:
        """Reserve and return the next monotonic iteration id."""
        i = self._next_id
        self._next_id += 1
        return i

    def rationale_id(self, iter_id: int) -> str:
        return f"iter_{iter_id}"

    def write(self, *, iter_id: int, target: dict, kind: str, intent: str,
              current_precision: str, patcher_status: str,
              validator_verdict: str | None, accepted: bool, log_tag: str,
              rationale: str, phase: str | None = None,
              strategy_bug: bool = False, extra: dict | None = None) -> dict:
        """Append one iteration record and return it.

        ``phase`` is the two-phase walk phase (``correctness`` | ``speedup``) the
        iteration ran under. ``strategy_bug`` is set true only on
        ``patch_apply_failed`` (P6). ``extra`` folds in optional fields (digit
        deltas, candidate_sha, identity, …).
        """
        record = {
            "iter_id": iter_id,
            "phase": phase,
            "target": target,
            "kind": kind,
            "intent": intent,
            "current_precision": current_precision,
            "patcher_status": patcher_status,
            "validator_verdict": validator_verdict,
            "accepted": accepted,
            "log_tag": log_tag,
            "rationale": rationale,
        }
        if strategy_bug:
            record["strategy_bug"] = True
        if extra:
            record.update(extra)
        self._fh.write(json.dumps(record) + "\n")
        return record

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    def __enter__(self) -> "IterationLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
