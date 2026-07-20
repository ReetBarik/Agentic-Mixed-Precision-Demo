#!/usr/bin/env python3
"""Augment a characterization report with per-integral *tail sample* offsets.

Purpose
-------
The Validator's n=1000 random battery can miss failure modes at untested inputs
(``FLOAT_RETRO_PROBE``).  This emitter records, per integral, the specific input
*offsets* that stress the computation hardest so the Validator can re-test those
exact points on every candidate (see ``agents/validator/tail.py``).

Why a driver re-run and not a journal walk
------------------------------------------
The four tail criteria are measured on the integral's **output components**
(``coeff0.imag`` etc.) — worst relative error, worst cancellation-conditioning,
and magnitude extremes.  The characterizer's report (and its transient ~80 GB
journal) is REGION-keyed (per source ``file:line``, aggregated over samples): it
carries neither per-sample identity nor per-output-component scoring.  That
granularity lives only in the Validator/driver path (the app drivers' ``RES``
output vs the DD oracle).  So the honest realization of "characterize with
tail-preservation" is to regenerate exactly the per-sample output signal the
criteria need — run the vanilla + DD drivers over ``[0,total)`` and compare
per component — rather than walk a journal that never held this data (and, for
the CALIBRATION_v2 report, no longer exists on disk).  This is also far cheaper
than a full re-characterization: RES output is ~tens of MB, no journal.

The offsets are per-integral ``mt19937(12345)`` stream indices, bit-identical
across the tracked (characterizer) and app (validator) drivers, so they transfer
directly.  A per-integral ``determinism_hash`` (SHA-256 of the first-100 inputs)
is frozen alongside them; the Validator verifies it against the candidate binary
before trusting the offsets.

Schema written under ``integrals.<B>.tail_samples``::

    {
      "determinism_hash": "sha256:...",
      "max_rel_err":   [ {"offset": 4217, "criterion_value": 1.4e-05,
                          "output_component": "coeff0.imag"}, ...K ],
      "max_cond":      [ ... ],   # cancellation proxy: ref_scale / |component|
      "max_abs_value": [ ... ],   # largest |output component|
      "min_abs_value": [ ... ]    # smallest nonzero |output component|
    }

The original report is preserved as ``<report>.pre_tail.json`` before augmenting.

Usage (under the venv + module env)::

    python runs/qcdloop/emit_tail_offsets.py \
        --report runs/qcdloop/report_10k.json \
        --total 10000 --k 10 \
        --dd-repo ~/qcdloop --dd-ref ddfun_enabled \
        --kokkos-root ~/kokkos-install --workers 16 --chunk 500
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

import importlib  # noqa: E402

from agents.validator import runner, tail  # noqa: E402
# The agents.validator package re-exports the ``validate`` *function*, which
# shadows the submodule of the same name even for ``import ... as`` — so reach the
# submodule (for ``_build_dd_binary``) via import_module.
_validate = importlib.import_module("agents.validator.validate")  # noqa: E402
from agents.validator.coeffs import COMPONENT_LABELS, N_COMPONENTS  # noqa: E402
from agents.validator.precise_digits import effectively_zero  # noqa: E402


def _top_offsets(per_offset: dict[int, tuple[float, str]], k: int,
                 largest: bool) -> list[dict]:
    """Rank offsets by their per-offset criterion value; return the top-K entries.

    ``per_offset[offset] = (value, output_component)``.  ``largest`` picks the K
    highest values (rel-err / cond / max-abs); ``largest=False`` the K lowest
    (min-abs).  Distinct offsets — one entry per offset — for input diversity.
    """
    items = sorted(per_offset.items(), key=lambda kv: kv[1][0], reverse=largest)
    out = []
    for offset, (value, comp) in items[:k]:
        out.append({
            "offset": int(offset),
            "criterion_value": float(value),
            "output_component": comp,
        })
    return out


def _select_tail(van, dd, total: int, k: int) -> dict:
    """Compute the four tail-criteria offset lists for one integral.

    ``van`` / ``dd`` are ``(hi, lo)`` ``array('d')`` buffers of length
    ``total*N_COMPONENTS`` (vanilla ``lo == 0``).  Per sample the characteristic
    magnitude ``ref_scale`` is the max |DD component|; analytic zeros (DD ref below
    the noise floor for that scale) are excluded from rel-err / cond / min-abs
    (they carry no meaningful signal), but never win max-abs anyway.
    """
    v_hi, v_lo = van
    d_hi, d_lo = dd

    best_relerr: dict[int, tuple[float, str]] = {}
    best_cond: dict[int, tuple[float, str]] = {}
    best_maxabs: dict[int, tuple[float, str]] = {}
    best_minabs: dict[int, tuple[float, str]] = {}

    for s in range(total):
        base = s * N_COMPONENTS
        # per-sample characteristic magnitude (max |DD component|)
        ref_scale = 0.0
        dd_vals = []
        for c in range(N_COMPONENTS):
            j = base + c
            val = d_hi[j] + d_lo[j]
            dd_vals.append(val)
            m = abs(val)
            if m > ref_scale:
                ref_scale = m

        for c in range(N_COMPONENTS):
            j = base + c
            true = dd_vals[c]
            abs_true = abs(true)
            comp = COMPONENT_LABELS[c]
            is_zero = (abs_true == 0.0)
            is_analytic_zero = effectively_zero(abs_true, ref_scale)

            # (c) max abs value — largest |output component| (over/underflow-large)
            if not is_zero:
                cur = best_maxabs.get(s)
                if cur is None or abs_true > cur[0]:
                    best_maxabs[s] = (abs_true, comp)

            if is_zero or is_analytic_zero:
                continue  # analytic/exact zeros carry no rel-err / cond / min signal

            # (a) worst rel-err on the output component
            van_val = v_hi[j] + v_lo[j]
            rel = abs(van_val - true) / abs_true
            cur = best_relerr.get(s)
            if cur is None or rel > cur[0]:
                best_relerr[s] = (rel, comp)

            # (b) cancellation-conditioning proxy: ref_scale / |component|.
            # Large ⇒ the component is a small residue of large quantities
            # (catastrophic cancellation) ⇒ ill-conditioned.  Distinct from (a):
            # a small component may still be computed accurately.
            cond = (ref_scale / abs_true) if abs_true > 0.0 else 0.0
            cur = best_cond.get(s)
            if cur is None or cond > cur[0]:
                best_cond[s] = (cond, comp)

            # (d) min nonzero abs value — smallest genuine |output component|
            cur = best_minabs.get(s)
            if cur is None or abs_true < cur[0]:
                best_minabs[s] = (abs_true, comp)

    return {
        "max_rel_err":   _top_offsets(best_relerr, k, largest=True),
        "max_cond":      _top_offsets(best_cond, k, largest=True),
        "max_abs_value": _top_offsets(best_maxabs, k, largest=True),
        "min_abs_value": _top_offsets(best_minabs, k, largest=False),
    }


def emit(report_path: Path, total: int, k: int, dd_repo: Path, dd_ref: str,
         kokkos_root: Path, vanilla_headers: Path, chunk: int, workers: int) -> dict:
    """Build drivers, compute tail offsets for every integral, augment the report.

    Returns a small telemetry summary (also useful for tests / the pipeline doc).
    """
    t0 = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="qcdloop_tailemit_") as scratch_s:
        scratch = Path(scratch_s)

        # --- build vanilla + DD drivers ---
        van_tree = scratch / "vtree"
        shutil.copytree(vanilla_headers, van_tree)
        van_binary = runner.build_driver(van_tree, "vanilla", scratch / "vbuild",
                                         kokkos_root)
        dd_binary = _validate._build_dd_binary(dd_repo, dd_ref, kokkos_root,
                                               scratch / "dd")

        # --- run both over [0,total) ---
        print(f"[emit] running vanilla over [0,{total})...", flush=True)
        van = runner.run_and_aggregate(van_binary, total, chunk=chunk, workers=workers)
        print(f"[emit] running DD over [0,{total})...", flush=True)
        dd = runner.run_and_aggregate(dd_binary, total, chunk=chunk, workers=workers)

        # --- determinism hash (first-100 inputs per integral) ---
        determ = tail.determinism_hash(van_binary, tail.DETERMINISM_N)

    integrals = sorted(set(van) & set(dd))
    tail_by_integral: dict[str, dict] = {}
    for integral in integrals:
        ts = _select_tail(van[integral], dd[integral], total, k)
        ts["determinism_hash"] = determ.get(integral)
        tail_by_integral[integral] = ts
        n_off = len(tail.integral_offsets(ts))
        print(f"[emit] {integral}: {n_off} distinct tail offsets, "
              f"hash={ts['determinism_hash']}", flush=True)

    # --- augment report in place, preserving the original ---
    pre = report_path.with_suffix(".pre_tail.json")
    if not pre.exists():
        print(f"[emit] preserving original -> {pre.name}", flush=True)
        shutil.copy2(report_path, pre)
    else:
        print(f"[emit] {pre.name} already exists; not overwriting the preserved copy",
              flush=True)

    print(f"[emit] loading report {report_path.name} ...", flush=True)
    with open(report_path) as fh:
        report = json.load(fh)
    ri = report.setdefault("integrals", {})
    augmented = 0
    for integral, ts in tail_by_integral.items():
        if integral in ri:
            ri[integral]["tail_samples"] = ts
            augmented += 1
    report["tail_schema_version"] = 1

    tmp = report_path.with_suffix(".json.tmp")
    with open(tmp, "w") as fh:
        json.dump(report, fh, separators=(",", ":"))
    tmp.replace(report_path)

    summary = {
        "integrals_augmented": augmented,
        "total_samples_scanned": total,
        "k_per_criterion": k,
        "distinct_offsets": len(tail.all_offsets(tail_by_integral)),
        "hash_present": sum(1 for ts in tail_by_integral.values()
                            if ts.get("determinism_hash")),
        "wall_seconds": round(time.monotonic() - t0, 1),
    }
    print(f"[emit] done: {summary}", flush=True)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--report", default=str(HERE / "report_10k.json"))
    ap.add_argument("--total", type=int, default=10000,
                    help="samples per integral to scan (must match the report)")
    ap.add_argument("--k", type=int, default=10, help="offsets per criterion")
    ap.add_argument("--dd-repo", default=str(Path.home() / "qcdloop"))
    ap.add_argument("--dd-ref", default="ddfun_enabled")
    ap.add_argument("--kokkos-root", default=str(Path.home() / "kokkos-install"))
    ap.add_argument("--vanilla-headers",
                    default=str(REPO / "runs" / "qcdloop_headers_full"))
    ap.add_argument("--chunk", type=int, default=500)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(argv)

    report = Path(args.report).resolve()
    if not report.is_file():
        raise SystemExit(f"report not found: {report}")

    emit(report, args.total, args.k, Path(args.dd_repo).resolve(), args.dd_ref,
         Path(args.kokkos_root).resolve(), Path(args.vanilla_headers).resolve(),
         args.chunk, args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
