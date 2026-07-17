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
    }

``snapshot`` schema::  {"seed": 12345, "sample_count": 100000}
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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate(base_state: dict, candidate_patch: str | None, tolerance: float = 8.0,
             snapshot: dict | None = None, *, max_regression: float = 0.5,
             chunk: int = 0, workers: int = 1,
             run_id: str | None = None, persist: bool = True) -> dict:
    """Precision-loss acceptance test for one ``candidate_patch``.

    Returns the verdict object (see module docstring / the task contract).
    Stateless: no memory of prior calls.

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

    # ---- 3. candidate (never cached — the patch is the variable) ----
    with tempfile.TemporaryDirectory(prefix="qcdloop_cand_") as scratch:
        candidate_coeffs = _run_vanilla(
            vanilla_headers, accepted, candidate_patch, kokkos_root,
            Path(scratch), total, chunk, workers)

    # ---- 4. precise-digits vs DD, min + hotspot, persist ----
    if run_id is None:
        run_id = f"{work_tree_hash[:12]}_{snap_key}"
    out_dir = _VALIDATOR_ROOT / run_id
    cand_stats = _score(candidate_coeffs, dd_ref_coeffs, "candidate",
                        out_dir if persist else None)
    curr_stats = _score(current_coeffs, dd_ref_coeffs, "current",
                        out_dir if persist else None)

    verdict, reason = _decide(cand_stats["min_precise_digits"],
                              curr_stats["min_precise_digits"],
                              max_regression, floor=tolerance)

    return {
        "verdict": verdict,
        "threshold": float(tolerance),          # absolute precise-digit accept bar
        "max_regression": float(max_regression),  # delta guard vs current baseline
        "candidate": cand_stats,
        "current": curr_stats,
        "delta": round(cand_stats["min_precise_digits"] - curr_stats["min_precise_digits"], 6),
        "verdict_reason": reason,
        "snapshot": {"seed": seed, "sample_count": total},
        "run_id": run_id,
        "wall_seconds": round(time.monotonic() - t0, 1),
    }


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
    finally:
        if writer is not None:
            writer.close()

    if hot is not None:
        hot["precise_digits"] = round(hot["precise_digits"], 4)
    return {
        "min_precise_digits": round(best_min, 4),
        "hotspot": hot,
        "zeroed_components": zeroed,
        "total_components": total_components,
    }


# ---------------------------------------------------------------------------
# Run helpers (build a tree + driver, run, aggregate)
# ---------------------------------------------------------------------------

def _run_vanilla(vanilla_headers: Path, accepted: list, candidate_patch: str | None,
                 kokkos_root: Path, scratch: Path, total: int, chunk: int,
                 workers: int) -> runner.CoeffArrays:
    """Copy the working tree, apply patches, build+run the vanilla driver."""
    tree = scratch / "tree"
    shutil.copytree(vanilla_headers, tree)
    for patch in accepted:
        _git_apply(tree, patch)
    if candidate_patch:
        _git_apply(tree, candidate_patch)
    binary = runner.build_driver(tree, "vanilla", scratch / "build", kokkos_root)
    return runner.run_and_aggregate(binary, total, chunk=chunk, workers=workers)


def _run_dd(dd_repo: Path, dd_ref: str, kokkos_root: Path, scratch: Path,
            total: int, chunk: int, workers: int) -> runner.CoeffArrays:
    """Archive the ddfun_enabled DD tree, verify via the stub, build+run DD."""
    tree = scratch / "ddtree"
    tree.mkdir(parents=True, exist_ok=True)
    _git_archive(dd_repo, dd_ref, "src/qcdloop", tree)
    dd_headers = tree / "src" / "qcdloop"
    # dd_integrator stub: verify the DD triple is present (raises loudly if not).
    dd_integrator.integrate(dd_headers, dd_headers / "boxGPU.h")
    binary = runner.build_driver(dd_headers, "dd", scratch / "build", kokkos_root)
    return runner.run_and_aggregate(binary, total, chunk=chunk, workers=workers)


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


def _git_archive(repo: Path, ref: str, subpath: str, dest: Path) -> None:
    """Extract ``repo@ref:subpath`` into ``dest`` (repo stays on its branch)."""
    proc = subprocess.run(["git", "-C", str(repo), "archive", ref, subpath],
                          capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git archive {ref}:{subpath} failed:\n{proc.stderr.decode()[-1500:]}")
    tar = subprocess.run(["tar", "-x", "-C", str(dest)], input=proc.stdout,
                         capture_output=True)
    if tar.returncode != 0:
        raise RuntimeError(f"tar extract failed:\n{tar.stderr.decode()[-1500:]}")


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
