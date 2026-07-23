"""Validator agent — LangGraph node stub + the SHA↔patch adapter for Strategy.

Two things live here:

* :func:`run` — the (now vestigial) LangGraph node.  Strategy drives the Validator
  as a callable (Q5), so the graph no longer routes through this node.
* :func:`make_validator_fn` — the adapter that reconciles the two contracts
  (Strategy HANDOFF item 9).  Strategy calls ``validator_fn(candidate_sha, ctx)``
  with a **SHA on the strategy branch**; the real
  :func:`agents.validator.validate.validate` takes a **candidate patch** (a
  unified diff vs the pristine vanilla tree).  The adapter turns the SHA into that
  diff — ``git diff <starting_sha>..<candidate_sha>`` — so the candidate patch
  encodes the full accumulated tree (all accepted patches + this candidate) as one
  diff against ``vanilla_headers``.  This matches ``validate``'s v1 requirement
  that ``base_state.accepted_patches == []`` (the cumulative diff carries them).

The mirror direction — Patcher returning a SHA rather than a diff — is handled by
the Patcher committing on the branch (design §P2); together the two adapters mean
neither Strategy nor the Validator ever handles a raw diff/SHA mismatch.
"""

from __future__ import annotations

import subprocess
from typing import Callable

from agents.state import PipelineState
from agents.validator.validate import validate as _validate


def run(state: PipelineState) -> dict:
    """No-op graph node.  Strategy drives the Validator as a callable (Q5)."""
    return {}


def make_validator_fn(base_state: dict, starting_sha: str, repo_path: str,
                      *, tolerance: float = 8.0,
                      validate_fn: Callable | None = None,
                      scorer_manifest_path: str | None = None,
                      iteration_id: int = 0,
                      baseline_spec: dict | None = None):
    """Build ``validator_fn(candidate_sha, ctx) -> verdict`` for Strategy's state.

    ``base_state`` / ``tolerance`` are the validate() inputs; ``starting_sha`` is
    the branch base the cumulative candidate diff is taken against; ``repo_path``
    is the strategy working tree.  ``validate_fn`` is injectable for tests
    (defaults to the real Validator).

    Phase 2b — when ``scorer_manifest_path`` is set, each call additionally reduces
    the app-level delta attributable to the intent's region (the ``scorer_cell`` on
    ``ctx``, supplied by Strategy) into a ``(region_id, rung)`` cell appended to that
    manifest.  ``iteration_id`` tags every cell (working-tree baseline policy: cells
    are not cross-iteration-comparable); ``baseline_spec`` overrides the ground-truth
    reference (defaults to qcdloop's ``dd`` instantiation inside ``validate``).  With
    no ``scorer_manifest_path`` the Validator behaves exactly as pre-2b.
    """
    vfn = validate_fn if validate_fn is not None else _validate

    def validator_fn(candidate_sha: str, ctx: dict) -> dict:
        patch = _diff(repo_path, starting_sha, candidate_sha)
        snapshot = ctx.get("snapshot")
        tol = ctx.get("tolerance", tolerance)
        # Build-fuse: reuse the Patcher's gate binary for the candidate run when the
        # tree it was built against matches the candidate tree (CALIBRATION.md §Bug 5).
        return vfn(base_state, patch, tol, snapshot,
                   reuse_binary=ctx.get("gate_binary"),
                   reuse_tree_hash=ctx.get("gate_tree_hash"),
                   cell=ctx.get("scorer_cell"),
                   scorer_manifest_path=scorer_manifest_path,
                   iteration_id=iteration_id,
                   baseline_spec=baseline_spec)

    return validator_fn


def _diff(repo_path: str, base_sha: str, candidate_sha: str) -> str:
    """Cumulative unified diff ``base_sha..candidate_sha`` (git apply -p1 form)."""
    r = subprocess.run(
        ["git", "-C", str(repo_path), "diff", f"{base_sha}..{candidate_sha}"],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git diff {base_sha[:8]}..{candidate_sha[:8]} failed:\n{r.stderr}")
    return r.stdout
