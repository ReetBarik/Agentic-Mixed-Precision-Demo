"""Validator — precision-loss acceptance test for one candidate patch.

Stateless entry point :func:`validate`.  Given the current working tree, a
candidate patch (unified diff against that tree), a precision tolerance, and an
input snapshot, it runs three builds on bit-identical inputs and reports how many
correct decimal digits the candidate retains versus a double-double ground truth:

1. **DD ground truth** — the working tree in double-double (~31 digits), built
   from a qcdloop@ddfun_enabled tree via the dd_integrator stub.  Cached per
   (dd-tree-hash, snapshot).
2. **current baseline** — the working tree as-is, vanilla double.  Cached per
   (working-tree-hash, snapshot).
3. **candidate** — the working tree + candidate patch, vanilla double.

Per output component (coeff0/1/2, real+imag independent), across all samples and
all 21 integrals, it computes precise-digits vs the DD reference (see
:mod:`agents.validator.precise_digits`), takes the min for both candidate and
current, and emits the verdict contract.  Components that are effectively zero
against their sample's characteristic magnitude (numeric/physics zeros — e.g. the
imaginary part of a purely-real integral) are reported at the cap rather than as
spurious 0-digit noise, and counted in ``zeroed_components``.  The verdict has
two gates: a **regression guard** (reject if the candidate loses more than
``max_regression`` digits of global-min vs current — a precision-preserving patch
always passes this, since double is inherently ~9-digit on the ill-conditioned
integrals) and an **absolute threshold** (``tolerance``, default 8 digits; a
non-regressing candidate below it is ``insufficient_fix``).  The Validator is
stateless — Strategy owns the accumulated accepted-patch state (passed in via
``base_state``).

``base_state`` schema (v1)::

    {
      "vanilla_headers": <path>,       # working tree, patchable (headers_full)
      "dd_source_repo":  <path>,       # external qcdloop checkout
      "dd_ref":          <git ref>,    # ddfun_enabled branch or SHA (archived)
      "accepted_patches": [<diff>, …], # unified diffs vs vanilla_headers (v1: [])
      "kokkos_root":     <path>,       # optional; defaults to ~/kokkos-install
      "tail_samples":    {<B>: {...}}, # optional; per-integral adversarial-offset
                                       # battery from the characterizer (see
                                       # agents.validator.tail).  Absent → no tail
                                       # battery (fail-open, random-only verdict).
    }

``snapshot`` schema::  {"seed": 12345, "sample_count": 100000}

Tail battery
------------
When ``base_state["tail_samples"]`` is present, in addition to the n random samples
the Validator re-tests the specific per-integral input offsets the characterizer
flagged as adversarial (worst rel-err / cancellation-conditioning / magnitude
extremes on the output components).  It first verifies each integral's
``determinism_hash`` against the candidate binary (a mismatch raises
:class:`agents.validator.tail.DeterminismMismatch` loudly — the offsets are only
meaningful if the input generator is unchanged), then dispatches the offsets via
``--sample-list`` and folds the worst tail precise-digits into the candidate's
gating minimum.  A tail failure is therefore a hard reject, never a warning.
"""

from __future__ import annotations

import hashlib
import json
import pickle
import shutil
import subprocess
import tempfile
import time
from array import array
from pathlib import Path

from agents.integrator_base import cache as _hashcache
from agents.dd_integrator import agent as dd_integrator
from agents.validator import runner
from agents.validator import scorer as _scorer
from agents.validator import tail as _tail
from agents.validator.coeffs import COMPONENT_LABELS, N_COMPONENTS
from agents.validator.precise_digits import (
    MAX_DIGITS_F, effectively_zero, precise_digits_fast,
)

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_VANILLA = _REPO / "runs" / "qcdloop_headers_full"
_DEFAULT_DD_REPO = Path.home() / "qcdloop"
_DEFAULT_KOKKOS = Path.home() / "kokkos-install"
_VALIDATOR_ROOT = _REPO / "runs" / "qcdloop" / "validator"
_CACHE_DIR = _VALIDATOR_ROOT / "cache"

# The regression-guard margin: a candidate may lose at most this many digits of
# global-min vs the baseline before it is rejected as a regression.  Exposed as a
# named constant (default for ``validate``'s ``max_regression``) so downstream
# consumers that apply the same regression-relative rule — e.g. the Phase-2e solver
# gate — reuse the one figure instead of re-inventing it.
DEFAULT_MAX_REGRESSION = 0.5


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate(base_state: dict, candidate_patch: str | None, tolerance: float = 8.0,
             snapshot: dict | None = None, *, max_regression: float = DEFAULT_MAX_REGRESSION,
             chunk: int = 0, workers: int = 1,
             run_id: str | None = None, persist: bool = True,
             reuse_binary: str | None = None,
             reuse_tree_hash: str | None = None,
             cell: dict | None = None,
             scorer_manifest_path: str | None = None,
             iteration_id: int = 0,
             baseline_spec: dict | None = None) -> dict:
    """Precision-loss acceptance test for one ``candidate_patch``.

    Returns the verdict object (see module docstring / the task contract).
    Stateless: no memory of prior calls.

    Phase 2b — the scorer (measurement) is now split from the verdict (decision).
    When ``cell`` is supplied (``{region_id, rung, intent_id, integrals}``), the
    already-computed candidate + DD-reference coeff arrays are additionally reduced
    into a ``(region_id, rung) -> delta`` manifest cell (see
    :mod:`agents.validator.scorer`) — the app-level relative error *attributable to
    that region at that rung*, un-buried from the whole-app min the verdict gates on.
    The cell is appended to ``scorer_manifest_path`` (a documented side-effect
    artifact) and echoed back under the verdict's additive ``"scorer"`` key.  The
    verdict itself (returned keys, gate logic) is unchanged: existing callers that
    pass no ``cell`` see exactly the pre-2b behavior.  This reduction is pure over
    arrays the Validator already builds, so it adds no wall-clock.

    Two gates, evaluated in order:

    * **regression guard** (delta) — reject with reason ``regression`` if the
      candidate loses more than ``max_regression`` digits of global-min vs the
      current baseline (``cand_min - curr_min < -max_regression``).  Robust to
      double precision being inherently limited on the ill-conditioned integrals:
      a patch that merely preserves the existing precision never trips this.
    * **absolute threshold** — ``tolerance`` is the precise-digit bar the
      candidate must clear (default 8).  If it does not regress but still falls
      below ``tolerance``, the reason is ``insufficient_fix``.

    ``chunk``/``workers`` tune the chunked run (0 chunk → single [0, total) run).
    ``run_id`` names the persisted per-sample precise-digits artifact;
    auto-derived if omitted.
    """
    snapshot = snapshot or {"seed": 12345, "sample_count": 100000}
    seed = int(snapshot.get("seed", 12345))
    total = int(snapshot["sample_count"])
    snap_key = f"seed{seed}_n{total}"

    vanilla_headers = Path(base_state.get("vanilla_headers", _DEFAULT_VANILLA)).resolve()
    dd_repo = Path(base_state.get("dd_source_repo", _DEFAULT_DD_REPO)).resolve()
    dd_ref = base_state.get("dd_ref", "ddfun_enabled")
    accepted = list(base_state.get("accepted_patches", []))
    kokkos_root = Path(base_state.get("kokkos_root", _DEFAULT_KOKKOS)).resolve()

    if accepted:
        # v1 limitation: accepted patches are diffs vs the vanilla (master) tree;
        # mapping them onto the ddfun_enabled DD tree needs the 3-file line-map we
        # deferred. Surface loudly rather than silently produce a mismatched ref.
        raise NotImplementedError(
            "validate(): non-empty accepted_patches not supported in v1 — the DD "
            "ground-truth tree (ddfun_enabled) would need the deferred master->"
            "ddfun line-map to apply vanilla-tree patches. accepted_patches must "
            "be [] for now."
        )

    work_tree_hash = _working_tree_hash(vanilla_headers, accepted)
    dd_tree_hash = _dd_tree_hash(dd_repo, dd_ref)

    # Tail battery spec (characterization-derived): {integral: {determinism_hash,
    # max_rel_err:[...], ...}}.  Fail-open — an old report predating the schema (or
    # a caller that does not pass it) simply skips the tail battery and the verdict
    # reverts to the random-only behavior (see _tail_battery).
    tail_spec = base_state.get("tail_samples") or {}
    tail_offsets = _tail.all_offsets(tail_spec) if tail_spec else []

    t0 = time.monotonic()

    # ---- 1. DD ground truth (cached per dd-tree-hash + snapshot) ----
    dd_ref_coeffs = _cached_or_run(
        role="dd", key=f"{dd_tree_hash}_{snap_key}",
        build_and_run=lambda scratch: _run_dd(
            dd_repo, dd_ref, kokkos_root, scratch, total, chunk, workers),
    )

    # ---- 2. current baseline (cached per work-tree-hash + snapshot) ----
    current_coeffs = _cached_or_run(
        role="current", key=f"{work_tree_hash}_{snap_key}",
        build_and_run=lambda scratch: _run_vanilla(
            vanilla_headers, accepted, None, kokkos_root, scratch, total, chunk, workers),
    )

    # ---- 3. candidate (never cached — the patch is the variable) + tail battery ----
    # The tail battery reuses the candidate's own binary: determinism check
    # (--dump-inputs) and adversarial-offset dispatch (--sample-list) run before
    # the scratch tree is torn down.
    cand_tail = None
    with tempfile.TemporaryDirectory(prefix="qcdloop_cand_") as scratch:
        candidate_coeffs, cand_binary = _run_vanilla(
            vanilla_headers, accepted, candidate_patch, kokkos_root,
            Path(scratch), total, chunk, workers,
            reuse_binary=reuse_binary, reuse_tree_hash=reuse_tree_hash,
            return_binary=True)
        if tail_spec:
            # Determinism check first (raises DeterminismMismatch loudly on drift).
            tested = [b for b in tail_spec if tail_spec[b].get("determinism_hash")]
            _tail.verify_determinism(
                cand_binary,
                {b: tail_spec[b]["determinism_hash"] for b in tested},
                tested)
            cand_tail = _tail.run_offsets(cand_binary, tail_offsets)

    # ---- 4. precise-digits vs DD, min + hotspot, persist ----
    if run_id is None:
        run_id = f"{work_tree_hash[:12]}_{snap_key}"
    out_dir = _VALIDATOR_ROOT / run_id
    cand_stats = _score(candidate_coeffs, dd_ref_coeffs, "candidate",
                        out_dir if persist else None)
    curr_stats = _score(current_coeffs, dd_ref_coeffs, "current",
                        out_dir if persist else None)

    # ---- 5. tail battery (regression-relative): score candidate AND current at
    #         the adversarial offsets; the tail feeds the REGRESSION guard (a
    #         candidate-induced tail loss vs the baseline is a hard reject), while
    #         the absolute floor stays on the random battery — adversarial offsets
    #         include workload physics ceilings where even the baseline is < tol,
    #         so an absolute tail floor would reject every candidate for a workload
    #         property (see PIPELINE_v1.md §tail-testing design finding). ----
    tail_stats = _tail_battery(
        tail_spec, cand_tail, tail_offsets,
        dd_repo, dd_ref, kokkos_root, dd_tree_hash,
        vanilla_headers, accepted, work_tree_hash)

    rand_cand_min = cand_stats["min_precise_digits"]
    rand_curr_min = curr_stats["min_precise_digits"]
    # combined (random + tail) minima drive the regression delta; None tail (fail-
    # open) leaves them at the random values -> exactly the pre-tail behavior.
    comb_cand_min = rand_cand_min
    comb_curr_min = rand_curr_min
    tcand = tail_stats["tail_cand_min_precise_digits"]
    tcurr = tail_stats["tail_curr_min_precise_digits"]
    if tcand is not None:
        comb_cand_min = min(comb_cand_min, tcand)
    if tcurr is not None:
        comb_curr_min = min(comb_curr_min, tcurr)

    verdict, reason = _decide_tail(rand_cand_min, comb_cand_min, comb_curr_min,
                                   max_regression, floor=tolerance)

    # ---- 6. scorer (Phase 2b): reduce the SAME candidate + DD arrays into a
    #         per-(region_id, rung) manifest cell, split from this verdict.  Pure
    #         reduction — no extra build/run; skipped entirely when no cell given. ----
    scorer_out = None
    if cell:
        scorer_out = _emit_scorer_cell(
            cell, candidate_coeffs, current_coeffs, dd_ref_coeffs,
            cand_tail, tail_offsets,
            dd_repo, dd_ref, kokkos_root, dd_tree_hash,
            work_tree_hash, snapshot, baseline_spec, iteration_id,
            scorer_manifest_path)

    return {
        "verdict": verdict,
        "threshold": float(tolerance),          # absolute precise-digit accept bar (random)
        "max_regression": float(max_regression),  # delta guard vs current baseline
        "candidate": cand_stats,
        "current": curr_stats,
        "cand_min_precise_digits": round(comb_cand_min, 4),  # combined random+tail min
        "curr_min_precise_digits": round(comb_curr_min, 4),
        "tail": tail_stats,
        "delta": round(comb_cand_min - comb_curr_min, 6),
        "verdict_reason": reason,
        "snapshot": {"seed": seed, "sample_count": total},
        "run_id": run_id,
        "wall_seconds": round(time.monotonic() - t0, 1),
        # Phase 2b additive field: the measured manifest cell + its artifact path.
        # None when no cell was requested (unchanged behavior for pre-2b callers).
        "scorer": scorer_out,
    }


def _emit_scorer_cell(cell: dict, candidate_coeffs, current_coeffs, dd_ref_coeffs,
                      cand_tail, tail_offsets, dd_repo, dd_ref, kokkos_root,
                      dd_tree_hash, work_tree_hash, snapshot,
                      baseline_spec, iteration_id,
                      manifest_path: str | None) -> dict:
    """Build the measured manifest cell for ``cell`` and append it to the manifest.

    ``cell`` = ``{region_id, rung, intent_id, integrals}``.  The reduction is over
    the candidate + DD arrays already computed for the verdict (random battery) and,
    when present, the sparse adversarial tail (``None`` -> ``delta_adversarial``
    null, the 2b stub).  The unpatched ``current_coeffs`` (also already computed)
    are reduced at the same scope into ``baseline_delta_*`` so inertness is visible.
    Returns ``{row, manifest_path}`` for the verdict echo.
    """
    spec = baseline_spec or _scorer.qcdloop_baseline_spec()
    bid = _scorer.baseline_id(spec, work_tree_hash, dd_tree_hash)
    battery = _scorer.snapshot_battery_spec(snapshot, tail_offsets)
    bver = battery["version"]

    # DD reference at the adversarial offsets (cached; returns {} when no offsets —
    # the 2b report_5k case, so this is free).
    dd_ref_tail = _dd_tail_coeffs(dd_repo, dd_ref, kokkos_root,
                                  tail_offsets, dd_tree_hash) if tail_offsets else None

    row = _scorer.score_cell(
        region_id=cell["region_id"], rung=cell["rung"],
        iteration_id=int(iteration_id),
        candidate_coeffs=candidate_coeffs, dd_ref_coeffs=dd_ref_coeffs,
        integrals_scope=cell.get("integrals"),
        baseline_id=bid, battery_version=bver,
        candidate_tail=cand_tail, dd_ref_tail=dd_ref_tail,
        baseline_coeffs=current_coeffs,
        intent_id=cell.get("intent_id"),
        patcher_metadata=cell.get("patcher_metadata"))

    if manifest_path:
        _scorer.append_row(manifest_path, row)
    return {"row": row.to_dict(), "manifest_path": manifest_path}


# ---------------------------------------------------------------------------
# Verdict logic
# ---------------------------------------------------------------------------

def _decide(cand_min: float, curr_min: float, max_regression: float,
            floor: float | None = None) -> tuple[str, str]:
    """Verdict + reason from the two gates, regression first then floor.

    1. **regression guard** — the candidate must not lose more than
       ``max_regression`` digits of global-min vs the *current* baseline::

           reject/regression  ⇔  (cand_min - curr_min) < -max_regression

       Double precision is inherently limited on the ill-conditioned integrals
       (~9 digits on the BIN cancellation cascade), so this delta guard lets a
       precision-preserving patch through regardless of the absolute bar.
    2. **absolute threshold** — ``floor`` is the precise-digit bar the candidate
       must clear.  A candidate that does not regress but still falls below
       ``floor`` is ``reject/insufficient_fix``.  ``None`` disables this gate
       (pure regression mode, used in some unit tests).
    """
    delta = cand_min - curr_min
    if delta < -max_regression:
        return "reject", "regression"
    if floor is not None and cand_min < floor:
        return "reject", "insufficient_fix"
    return "accept", "accept"


def _decide_tail(rand_cand_min: float, comb_cand_min: float, comb_curr_min: float,
                 max_regression: float, floor: float | None) -> tuple[str, str]:
    """Tail-aware verdict: regression on the combined (random+tail) minima, floor
    on the random-only candidate minimum.

    Two gates, in order:

    * **regression guard** — ``(comb_cand_min - comb_curr_min) < -max_regression``
      → ``reject/regression``.  Because both minima fold in the tail battery, a
      candidate that does materially worse than the baseline at an adversarial tail
      offset (an overflow on a float demotion, a broken cancellation) is a HARD
      reject here; a shared workload ceiling (candidate ≈ baseline at that offset)
      cancels in the delta and does not.
    * **absolute floor** — ``rand_cand_min < floor`` → ``reject/insufficient_fix``.
      Deliberately on the RANDOM battery, not the combined min: adversarial offsets
      (``min_abs`` / ``max_cond`` near-zero components) include inherent workload
      physics ceilings below ``tol`` that no demotion decision owns, so gating the
      absolute bar on them would reject every candidate for a workload property.

    Reduces to the pre-tail :func:`_decide` when the tail is absent (fail-open):
    then ``comb_cand_min == rand_cand_min`` and ``comb_curr_min == rand_curr_min``.
    """
    if (comb_cand_min - comb_curr_min) < -max_regression:
        return "reject", "regression"
    if floor is not None and rand_cand_min < floor:
        return "reject", "insufficient_fix"
    return "accept", "accept"


# ---------------------------------------------------------------------------
# Scoring: precise-digits over every component, min + hotspot, persistence
# ---------------------------------------------------------------------------

def _score(cand: runner.CoeffArrays, ref: runner.CoeffArrays, label: str,
           out_dir: Path | None) -> dict:
    """Min precise-digits of ``cand`` vs DD ``ref`` + its hotspot; persist array.

    Iterates every (integral, sample, component); tracks the global minimum and
    the component that realized it.  Each sample's ``ref_scale`` is the max
    |DD reference component| across that sample's six coeffs — a component whose
    DD reference is an analytic zero against it (see
    :func:`~agents.validator.precise_digits.effectively_zero`) reports at the cap
    rather than as spurious 0-digit noise.  ``zeroed_components`` counts the
    analytic zeros that carried a nonzero double roundoff (the ones the band
    actually rescued) so the min is never silently inflated.  When ``out_dir`` is
    given, writes a JSONL row per (integral, sample) with the six component
    digits to ``<out_dir>/<label>_precise_digits.jsonl``.
    """
    integrals = sorted(ref.keys())
    best_min = MAX_DIGITS_F
    hot = None
    zeroed = 0
    total_components = 0
    # Phase 2f kernel-scope: the per-integral min_precise_digits (a candidate targeting
    # kernel K is gated against K's own floor, not the whole-app min pinned by whichever
    # kernel is worst).  A pure by-product of the same per-component sweep — no extra run.
    per_integral_min: dict[str, float] = {}

    writer = None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        writer = open(out_dir / f"{label}_precise_digits.jsonl", "w")

    try:
        for integ in integrals:
            r_hi, r_lo = ref[integ]
            if integ not in cand:
                raise ValueError(f"{label}: integral {integ} missing (present in DD ref)")
            c_hi, c_lo = cand[integ]
            if len(c_hi) != len(r_hi):
                raise ValueError(
                    f"{label}: {integ} length {len(c_hi)} != DD ref {len(r_hi)} "
                    "(sample-count mismatch)")
            integ_min = MAX_DIGITS_F
            n_samples = len(r_hi) // N_COMPONENTS
            for s in range(n_samples):
                base = s * N_COMPONENTS
                # Per-sample characteristic magnitude: the largest |DD coeff|.
                ref_scale = 0.0
                for c in range(N_COMPONENTS):
                    j = base + c
                    m = abs(r_hi[j] + r_lo[j])
                    if m > ref_scale:
                        ref_scale = m
                row_digits = []
                for c in range(N_COMPONENTS):
                    j = base + c
                    d = precise_digits_fast(c_hi[j], c_lo[j], r_hi[j], r_lo[j],
                                            ref_scale=ref_scale)
                    row_digits.append(d)
                    total_components += 1
                    err = abs((c_hi[j] - r_hi[j]) + (c_lo[j] - r_lo[j]))
                    true = abs(r_hi[j] + r_lo[j])
                    if err != 0.0 and effectively_zero(true, ref_scale):
                        zeroed += 1
                    if d < integ_min:
                        integ_min = d
                    if d < best_min:
                        best_min = d
                        hot = {
                            "integral": integ,
                            "sample_idx": s,
                            "component": COMPONENT_LABELS[c],
                            "reference_dd": r_hi[j] + r_lo[j],
                            label: c_hi[j] + c_lo[j],
                            "precise_digits": d,
                        }
                if writer is not None:
                    writer.write(json.dumps({
                        "integral": integ, "sample_idx": s,
                        "digits": [round(x, 4) for x in row_digits],
                    }) + "\n")
            per_integral_min[integ] = round(integ_min, 4)
    finally:
        if writer is not None:
            writer.close()

    if hot is not None:
        hot["precise_digits"] = round(hot["precise_digits"], 4)
    return {
        "min_precise_digits": round(best_min, 4),
        "per_integral_min_precise_digits": per_integral_min,   # Phase 2f kernel-scope
        "hotspot": hot,
        "zeroed_components": zeroed,
        "total_components": total_components,
    }


# ---------------------------------------------------------------------------
# Run helpers (build a tree + driver, run, aggregate)
# ---------------------------------------------------------------------------

def _run_vanilla(vanilla_headers: Path, accepted: list, candidate_patch: str | None,
                 kokkos_root: Path, scratch: Path, total: int, chunk: int,
                 workers: int, *, reuse_binary: str | None = None,
                 reuse_tree_hash: str | None = None,
                 return_binary: bool = False):
    """Copy the working tree, apply patches, build+run the vanilla driver.

    Build-fuse (CALIBRATION.md §Bug 5): when ``reuse_binary`` is given and
    ``reuse_tree_hash`` matches the content hash of the (patched) candidate tree,
    reuse that binary — the Patcher already built this exact translation unit at its
    gate, so a second cmake build of the byte-identical tree is redundant.  Falls
    back to a fresh build on any mismatch (defensive; a hash mismatch should never
    happen in normal flow but guards against a race / stale artifact).
    """
    tree = scratch / "tree"
    shutil.copytree(vanilla_headers, tree)
    for patch in accepted:
        _git_apply(tree, patch)
    if candidate_patch:
        _git_apply(tree, candidate_patch)

    if (reuse_binary and reuse_tree_hash
            and Path(reuse_binary).is_file()
            and _hashcache.hash_header_dir(tree) == reuse_tree_hash):
        binary = Path(reuse_binary)
    else:
        binary = runner.build_driver(tree, "vanilla", scratch / "build", kokkos_root)
    coeffs = runner.run_and_aggregate(binary, total, chunk=chunk, workers=workers)
    if return_binary:
        return coeffs, binary
    return coeffs


def _build_dd_binary(dd_repo: Path, dd_ref: str, kokkos_root: Path,
                     scratch: Path) -> Path:
    """Archive the ddfun_enabled DD tree, verify via the stub, build the DD driver."""
    tree = scratch / "ddtree"
    dd_headers = runner.materialize_dd_headers(dd_repo, dd_ref, tree)
    # dd_integrator stub: verify the DD triple is present (raises loudly if not).
    dd_integrator.integrate(dd_headers, dd_headers / "boxGPU.h")
    return runner.build_driver(dd_headers, "dd", scratch / "build", kokkos_root)


def _run_dd(dd_repo: Path, dd_ref: str, kokkos_root: Path, scratch: Path,
            total: int, chunk: int, workers: int) -> runner.CoeffArrays:
    """Archive the ddfun_enabled DD tree, verify via the stub, build+run DD."""
    binary = _build_dd_binary(dd_repo, dd_ref, kokkos_root, scratch)
    return runner.run_and_aggregate(binary, total, chunk=chunk, workers=workers)


# ---------------------------------------------------------------------------
# Tail battery: adversarial per-integral offsets, scored against the DD oracle
# ---------------------------------------------------------------------------

def _offset_key(offsets: list[int]) -> str:
    return hashlib.sha256(",".join(str(o) for o in offsets).encode()).hexdigest()[:16]


def _dd_tail_coeffs(dd_repo: Path, dd_ref: str, kokkos_root: Path,
                    offsets: list[int], dd_tree_hash: str) -> dict:
    """DD reference coeffs at the tail ``offsets`` (``{integral: {offset: [(hi,lo)×6]}}``).

    Cached on ``(dd_tree_hash, offset-set)`` — the DD tree is pinned and the tail
    offsets are fixed for a run, so this builds+runs the DD driver at most once per
    Strategy run regardless of how many candidates are validated.
    """
    offsets = sorted({int(o) for o in offsets if int(o) >= 0})
    if not offsets:
        return {}

    def build_and_run(scratch: Path) -> dict:
        binary = _build_dd_binary(dd_repo, dd_ref, kokkos_root, scratch)
        return _tail.run_offsets(binary, offsets)

    return _cached_or_run(role="dd_tail",
                          key=f"{dd_tree_hash}_{_offset_key(offsets)}",
                          build_and_run=build_and_run)


def _current_tail_coeffs(vanilla_headers: Path, accepted: list, kokkos_root: Path,
                         offsets: list[int], work_tree_hash: str) -> dict:
    """Current-baseline coeffs at the tail ``offsets`` (pristine working tree).

    The tail battery is *regression-relative*: many adversarial offsets (esp.
    ``min_abs`` / ``max_cond`` near-zero components) are workload physics ceilings
    where even the double baseline sits below ``tol`` — an absolute tail floor
    would reject every candidate for a workload property.  So the current baseline
    is scored at the SAME offsets and the tail contributes only to the regression
    delta.  Cached on ``(work_tree_hash, offset-set)`` — the current tree is fixed
    for a run (v1: accepted == []), so this builds+runs at most once per run.
    """
    offsets = sorted({int(o) for o in offsets if int(o) >= 0})
    if not offsets:
        return {}

    def build_and_run(scratch: Path) -> dict:
        _coeffs, binary = _run_vanilla(vanilla_headers, accepted, None, kokkos_root,
                                       scratch, 1, 0, 1, return_binary=True)
        return _tail.run_offsets(binary, offsets)

    return _cached_or_run(role="current_tail",
                          key=f"{work_tree_hash}_{_offset_key(offsets)}",
                          build_and_run=build_and_run)


def _score_tail(cand_tail: dict, dd_tail: dict, tail_spec: dict) -> dict:
    """Min precise-digits of candidate vs DD over each integral's tail offsets.

    Mirrors :func:`_score`'s per-component metric (per-sample ``ref_scale`` = the
    max |DD coeff| across the six components; analytic zeros report at the cap) but
    over the sparse ``{integral: {offset: [(hi,lo)×6]}}`` tail structure.  Returns
    the global tail minimum, its hotspot, the number of (integral, offset) samples
    tested, and the integrals whose tail spec had to be skipped (fail-open).
    """
    best_min = MAX_DIGITS_F
    hot = None
    tested_samples = 0
    covered: list[str] = []
    skipped: list[str] = []

    for integral in sorted(tail_spec):
        offs = _tail.integral_offsets(tail_spec[integral])
        c_by_off = cand_tail.get(integral, {})
        d_by_off = dd_tail.get(integral, {})
        if not offs or not c_by_off or not d_by_off:
            skipped.append(integral)
            continue
        covered.append(integral)
        for off in offs:
            c_comps = c_by_off.get(off)
            d_comps = d_by_off.get(off)
            if c_comps is None or d_comps is None:
                continue
            ref_scale = 0.0
            for (dh, dl) in d_comps:
                m = abs(dh + dl)
                if m > ref_scale:
                    ref_scale = m
            for c in range(N_COMPONENTS):
                ch, cl = c_comps[c]
                dh, dl = d_comps[c]
                d = precise_digits_fast(ch, cl, dh, dl, ref_scale=ref_scale)
                if d < best_min:
                    best_min = d
                    hot = {
                        "integral": integral,
                        "offset": off,
                        "component": COMPONENT_LABELS[c],
                        "reference_dd": dh + dl,
                        "candidate": ch + cl,
                        "precise_digits": round(d, 4),
                    }
            tested_samples += 1

    return {
        "tail_min_precise_digits": (round(best_min, 4) if tested_samples else None),
        "tail_hotspot": hot,
        "tail_samples_tested": tested_samples,
        "integrals_covered": covered,
        "integrals_skipped": skipped,
    }


_TAIL_WARNED: set[str] = set()


def _tail_battery(tail_spec: dict, cand_tail: dict | None, offsets: list[int],
                  dd_repo: Path, dd_ref: str, kokkos_root: Path, dd_tree_hash: str,
                  vanilla_headers: Path, accepted: list, work_tree_hash: str) -> dict:
    """Run + score the tail battery (candidate AND current) for the verdict.

    Regression-relative by design: scores both the candidate and the current
    baseline at the same adversarial offsets against the DD oracle, so a candidate
    that merely inherits a workload physics ceiling at a tail point is not
    penalized — only a candidate that does materially *worse* than the baseline
    there (a candidate-induced tail failure) trips the regression guard.

    Fail-open: with no ``tail_spec`` (old report / caller opted out) or no offsets,
    returns ``tail_cand_min = tail_curr_min = None`` so the verdict is exactly the
    random-only behavior.  A one-time per-integral warning is emitted for any
    integral whose tail spec is present but unusable.
    """
    empty = {
        "tail_batteries_run": 0,
        "tail_hash_mismatches": 0,   # a mismatch raises DeterminismMismatch upstream
        "tail_samples_tested": 0,
        "tail_offsets": 0,
        "tail_cand_min_precise_digits": None,
        "tail_curr_min_precise_digits": None,
        "tail_hotspot": None,
        "integrals_covered": [],
        "integrals_skipped": [],
    }
    if not tail_spec or not offsets or cand_tail is None:
        return empty

    dd_tail = _dd_tail_coeffs(dd_repo, dd_ref, kokkos_root, offsets, dd_tree_hash)
    curr_tail = _current_tail_coeffs(vanilla_headers, accepted, kokkos_root,
                                     offsets, work_tree_hash)
    cand_scored = _score_tail(cand_tail, dd_tail, tail_spec)
    curr_scored = _score_tail(curr_tail, dd_tail, tail_spec)
    for integral in cand_scored["integrals_skipped"]:
        if integral not in _TAIL_WARNED:
            _TAIL_WARNED.add(integral)
            print(f"[validator] tail battery: skipping {integral} "
                  f"(no usable tail spec / offsets)", flush=True)
    return {
        "tail_batteries_run": len(cand_scored["integrals_covered"]),
        "tail_hash_mismatches": 0,
        "tail_samples_tested": cand_scored["tail_samples_tested"],
        "tail_offsets": len(offsets),
        "tail_cand_min_precise_digits": cand_scored["tail_min_precise_digits"],
        "tail_curr_min_precise_digits": curr_scored["tail_min_precise_digits"],
        "tail_hotspot": cand_scored["tail_hotspot"],
        "integrals_covered": cand_scored["integrals_covered"],
        "integrals_skipped": cand_scored["integrals_skipped"],
    }


def _cached_or_run(role: str, key: str, build_and_run) -> runner.CoeffArrays:
    """Return cached coeff arrays for (role, key) or compute + cache them."""
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = _CACHE_DIR / f"{role}_{key}.pkl"
    if cache_file.is_file():
        with open(cache_file, "rb") as fh:
            return pickle.load(fh)
    with tempfile.TemporaryDirectory(prefix=f"qcdloop_{role}_") as scratch:
        coeffs = build_and_run(Path(scratch))
    tmp = cache_file.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump(coeffs, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(cache_file)
    return coeffs


# ---------------------------------------------------------------------------
# Tree + patch + hashing helpers
# ---------------------------------------------------------------------------

def _git_apply(tree: Path, patch_text: str) -> None:
    """Apply a unified diff (a/… b/… , -p1) to files under ``tree``."""
    r = subprocess.run(["git", "apply", "-p1", "--whitespace=nowarn", "-"],
                       cwd=str(tree), input=patch_text, text=True,
                       capture_output=True)
    if r.returncode != 0:
        raise RuntimeError(f"git apply failed in {tree}:\n{r.stderr}")


def _working_tree_hash(vanilla_headers: Path, accepted: list) -> str:
    """Hash the working tree: header bytes ⊕ accepted-patch text."""
    h = hashlib.sha256()
    h.update(_hashcache.hash_header_dir(vanilla_headers).encode())
    for patch in accepted:
        h.update(b"\0")
        h.update(patch.encode("utf-8"))
    return h.hexdigest()


def _dd_tree_hash(dd_repo: Path, dd_ref: str) -> str:
    """Hash the DD tree by its pinned git object id (ddfun_enabled commit)."""
    r = subprocess.run(["git", "-C", str(dd_repo), "rev-parse", dd_ref],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"git rev-parse {dd_ref} failed:\n{r.stderr}")
    return r.stdout.strip()
