"""P2 return contract — the 9-value status enum + response builders.

Every Patcher call returns the dict shape locked in design §P2::

    {
      "status": <one of STATUSES>,
      "candidate_sha": <40-hex> | None,
      "parent_sha":    <40-hex> | None,
      "artifacts": {"shim_paths", "boundary_patch_path",
                    "build_log_path", "runtime_log_path"},
      "error":     {"kind", "detail", "excerpt_path"} | None,
      "llm_tokens": int,          # cumulative LLM tokens this intent spent
    }

``ok`` carries the freshly-committed ``candidate_sha`` on ``strategy/<run_id>``;
every non-``ok`` status carries no commit (``candidate_sha`` is None) and leaves
the working tree reset to ``parent_sha``.  ``llm_tokens`` is a Patcher extension
Strategy already reads for its budget accounting (see ``StrategyRun._drive_walk``).
"""

from __future__ import annotations

from pathlib import Path

# The exhaustive statuses (design §P2).
OK = "ok"
LLM_GEN_FAILED = "llm_gen_failed"
PATCH_APPLY_FAILED = "patch_apply_failed"
COMMIT_FAILED = "commit_failed"
BUILD_FAILED = "build_failed"
RUNTIME_CRASHED = "runtime_crashed"
RUNTIME_NAN = "runtime_nan"
TIMEOUT = "timeout"
# A gen+build that succeeded but produced NO net tree change vs the parent — the
# candidate is byte-identical to the baseline, so there is nothing to commit.
# Distinct from COMMIT_FAILED (a genuine git commit failure, Q3-fatal): an empty
# candidate is benign — the remediation produced no distinct change — so Strategy
# advances the walk instead of aborting the run.
EMPTY_CANDIDATE = "empty_candidate"
# A plain-type-edit rung (``double-to-float`` / the ``ff-to-float`` composite tail)
# that cannot apply because the region source carries no bare ``double`` token to
# rewrite — the region is template-typed (``T``, not a literal ``double``), so the
# transition is *inapplicable to this code*, not a malformed Strategy intent.
# Distinct from PATCH_APPLY_FAILED (a genuine strategy_bug: bad intent / missing
# revert commit): an inapplicable rung is benign — Strategy advances the walk
# (settling at the current rung) instead of flagging a bug.
PATCH_INAPPLICABLE = "patch_inapplicable"
# A fan-out variant / in-place promotion whose rendered region body is byte-identical
# to the original — an *empty promotion payload* (no reads/writes retyped), so the
# candidate would be a bit-for-bit clone of the baseline at plain double.  Terminal
# and deterministic (retrying the LLM shim cannot change the source-derived reads):
# distinct from EMPTY_CANDIDATE (a whole-tree no-diff after a *real* edit) — this is
# specifically "the promotion transform did nothing".  Phase 2c defense-in-depth so
# an empty payload can never again masquerade as a silent ``measured`` scorer cell.
PROMOTION_NO_OP = "promotion_no_op"
# An UPCAST (ff/dd) promotion that DID retype the region body, but whose every landing
# is a store back to caller precision (a Case-B write, or a region-local decl typed at
# the caller scalar/complex) with no wider persistent sink — so the extended value is
# truncated at the region boundary and the candidate is numerically inert (delta ==
# baseline).  Terminal + deterministic (source-derived, rung-fixed), detected at gen
# time upstream of any build.  Phase 2d-B — the upcast analogue of PROMOTION_NO_OP:
# distinct in that the body IS retyped (not byte-identical), it just cannot survive the
# demotion at the boundary.  NEVER raised for a native ``float`` downcast (truncating to
# a narrower target is real precision loss — same two_limb discipline as the 2d-A guard).
WRITE_TRUNCATION = "write_truncation"

STATUSES = frozenset({
    OK, LLM_GEN_FAILED, PATCH_APPLY_FAILED, COMMIT_FAILED,
    BUILD_FAILED, RUNTIME_CRASHED, RUNTIME_NAN, TIMEOUT, EMPTY_CANDIDATE,
    PATCH_INAPPLICABLE, PROMOTION_NO_OP, WRITE_TRUNCATION,
})

# error.kind vocabulary (design §P2).
ERR_COMPILE = "compile"
ERR_NAN = "nan"
ERR_CRASH = "crash"
ERR_INTEGRATOR = "integrator"
ERR_LLM = "llm"
ERR_APPLY = "apply"
ERR_COMMIT = "commit"
ERR_TIMEOUT = "timeout"
ERR_EMPTY = "empty"
ERR_TRUNCATION = "truncation"   # Phase 2d-B write-boundary truncation


def _artifacts(shim_paths=None, boundary_patch_path=None,
               build_log_path=None, runtime_log_path=None,
               gate_binary=None, gate_tree_hash=None) -> dict:
    def _s(p):
        return str(p) if p is not None else None
    return {
        "shim_paths": [str(p) for p in shim_paths] if shim_paths else None,
        "boundary_patch_path": _s(boundary_patch_path),
        "build_log_path": _s(build_log_path),
        "runtime_log_path": _s(runtime_log_path),
        # Build-fuse handoff: the gate's built binary + the content hash of the tree
        # it was built against (CALIBRATION.md §Bug 5).  The Validator reuses the
        # binary for its candidate run when this hash matches the tree it would
        # build, halving the per-accept build cost.
        "gate_binary": _s(gate_binary),
        "gate_tree_hash": gate_tree_hash,
    }


def ok(candidate_sha: str, parent_sha: str, *, shim_paths=None,
       boundary_patch_path=None, build_log_path=None, runtime_log_path=None,
       gate_binary=None, gate_tree_hash=None, llm_tokens: int = 0) -> dict:
    return {
        "status": OK,
        "candidate_sha": candidate_sha,
        "parent_sha": parent_sha,
        "artifacts": _artifacts(shim_paths, boundary_patch_path,
                                build_log_path, runtime_log_path,
                                gate_binary, gate_tree_hash),
        "error": None,
        "llm_tokens": llm_tokens,
    }


def failure(status: str, parent_sha: str | None, *, err_kind: str | None = None,
            detail: str | None = None, excerpt_path: str | Path | None = None,
            shim_paths=None, boundary_patch_path=None, build_log_path=None,
            runtime_log_path=None, llm_tokens: int = 0) -> dict:
    if status not in STATUSES or status == OK:
        raise ValueError(f"not a failure status: {status!r}")
    return {
        "status": status,
        "candidate_sha": None,
        "parent_sha": parent_sha,
        "artifacts": _artifacts(shim_paths, boundary_patch_path,
                                build_log_path, runtime_log_path),
        "error": {
            "kind": err_kind,
            "detail": detail,
            "excerpt_path": str(excerpt_path) if excerpt_path is not None else None,
        },
        "llm_tokens": llm_tokens,
    }
