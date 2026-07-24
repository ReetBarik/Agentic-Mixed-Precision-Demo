#!/usr/bin/env python3
"""Item 6-revised: B12 COMPUTED-vs-ALGORITHMIC via tracked-bound error decomposition.

Read-only over the existing 5k characterizer shards
(``/vast/projects/pepper_hep/qcdloop_shards_5k/``).  No runs, no builds, no LLM,
no >dd reference.  Reuses ``stability_reducer.merge_reports`` / ``finalize_report``
(the ``merge`` CLI path) verbatim — this script only trims each shard to the
integrals of interest before merging, and derives the two rungs the shard schema
doesn't ship (``double`` and ``dd``) as ``U_rung * max_sensitivity``.

The tracked-bound framework (see stability_reducer module header):
  * ``max_sensitivity = cond * amp`` — the first-order forward-cone amplification
    of a machine-eps roundoff injected at a region/chain to the observable output.
    cond and amp are properties of the *math function* → precision-invariant.
  * ``predicted_rel_err_if_<rung> = U_<rung> * max_sensitivity`` where
    U_float=2^-24, U_ff=2^-46 (reducer's empirical ff floor), U_double=2^-53,
    U_dd=2^-106.  (The prompt cited U_ff=2^-44 / U_double=2^-53; we use the
    reducer's own U_FF=2^-46 as the source of truth and note it.)

Discriminator: whether that first-order model *explains* the measured error.
  * predicted_if_double ~= measured  -> bound tight -> error IS machine-eps
    amplification -> dd (U_dd = U_double * 2^-53) scales it down ~16 orders ->
    COMPUTED (dd would lift the floor).
  * predicted_if_double << measured  -> the region/chain's OWN sensitivity does
    not capture the error (it is injected upstream and amplified through here);
    the region-level bound is loose and cannot by itself quantify the dd lift.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agents.shared.stability_reducer import merge_reports, finalize_report  # noqa: E402

SHARD_DIR = Path("/vast/projects/pepper_hep/qcdloop_shards_5k")
INTEGRALS = ["B12", "B1", "B8", "B9", "B10"]   # B12 target + Item-5 sanity set

U_FLOAT = 2.0 ** -24     # ~5.96e-8
U_FF = 2.0 ** -46        # ~1.42e-14  (reducer's empirical ff floor)
U_DOUBLE = 2.0 ** -53    # ~1.11e-16
U_DD = 2.0 ** -106       # ~1.23e-32

# Item-5 measured single-region dd baseline deltas (whole-app rel-err) for the
# 7 measured-INERT cells, from DD_TRIAGE_2026-07-25.md (for the sanity check).
INERT_CELLS = [
    ("B10", "kokkosUtils.h:212", 1.324e-10, "physics_ceiling / documented floor"),
    ("B8",  "kokkosUtils.h:212", 7.266e-11, "benign_no_op"),
    ("B8",  "kokkosUtils.h:206", 7.266e-11, "benign_no_op"),
    ("B8",  "kokkosUtils.h:199", 7.266e-11, "benign_no_op"),
    ("B8",  "kokkosUtils.h:174", 7.266e-11, "benign_no_op"),
    ("B9",  "kokkosUtils.h:212", 2.950e-12, "benign_no_op"),
    ("B12", "kokkosUtils.h:212", 2.039e-4,  "benign_no_op (not B12 floor region)"),
]

# Whole-app floor for B12 from Item 3 (double vs dd, sample 3868, coeff0.imag).
B12_FLOOR_REL_ERR = 2.039e-4      # = 10**-3.6906
B12_FLOOR_DIGITS = 3.6906


def _predict(sens: float) -> dict:
    return {
        "predicted_rel_err_if_float": U_FLOAT * sens,
        "predicted_rel_err_if_ff": U_FF * sens,
        "predicted_rel_err_if_double": U_DOUBLE * sens,
        "predicted_rel_err_if_dd": U_DD * sens,
    }


def _digits(rel_err: float) -> float:
    if rel_err is None or rel_err <= 0.0:
        return 31.9
    return min(31.9, -math.log10(rel_err))


def _trim_shard(shard: dict) -> dict:
    """Keep only INTEGRALS from a shard report (structure-preserving)."""
    return {
        "schema_version": shard.get("schema_version", 2),
        "kind": shard.get("kind", "stability_shard_report"),
        "no_id_records": shard.get("no_id_records", 0),
        "samples_seen": {k: v for k, v in shard.get("samples_seen", {}).items()
                         if k in INTEGRALS},
        "integrals": {k: v for k, v in shard.get("integrals", {}).items()
                      if k in INTEGRALS},
    }


def load_and_merge() -> dict:
    shards = sorted(SHARD_DIR.glob("shard_*.json"))
    if not shards:
        raise SystemExit(f"no shards under {SHARD_DIR}")
    trimmed = []
    for sp in shards:
        t0 = time.time()
        with open(sp) as f:
            full = json.load(f)
        trimmed.append(_trim_shard(full))
        del full
        print(f"  loaded {sp.name} ({time.time()-t0:.1f}s)", flush=True)
    print(f"merging {len(trimmed)} shards (reusing merge_reports/finalize_report)...",
          flush=True)
    return finalize_report(merge_reports(trimmed))


def region_row(loc: str, reg: dict) -> dict:
    sens = reg.get("max_sensitivity", 0.0)
    measured = reg.get("max_rel_err", 0.0)
    pred = _predict(sens)
    pd = pred["predicted_rel_err_if_double"]
    tightness = (pd / measured) if measured > 0 else None
    return {
        "location": loc,
        "signal_class": reg.get("signal_class"),
        "max_cond": reg.get("max_cond", 0.0),
        "max_amp": reg.get("max_amp", 0.0),
        "max_sensitivity": sens,
        "measured_max_rel_err": measured,
        **pred,
        # tightness of the first-order bound: predicted_if_double / measured.
        # ~1 => tight; <<1 => model under-predicts (error injected upstream / not
        # captured by this region's own cone); >>1 => conservative over-estimate.
        "tightness_double_over_measured": tightness,
        "n": reg.get("n", 0),
    }


def chain_row(ch: dict) -> dict:
    sens = ch.get("max_sensitivity", 0.0)
    measured = ch.get("max_rel_err", 0.0)
    pred = _predict(sens)
    pd = pred["predicted_rel_err_if_double"]
    return {
        "chain_id": ch.get("chain_id"),
        "signal_class": ch.get("signal_class"),
        "n_contributors": ch.get("n", 0),
        "chain_lines": [f"{s.get('file')}:{s.get('line_start')}"
                        for s in ch.get("chain", [])],
        "max_cond": ch.get("max_cond", 0.0),
        "max_sensitivity": sens,
        "measured_max_rel_err": measured,
        **pred,
        "tightness_double_over_measured": (pd / measured) if measured > 0 else None,
    }


def main() -> int:
    print("Loading + merging 5k shards for", INTEGRALS, flush=True)
    report = load_and_merge()
    ints = report["integrals"]

    out: dict = {
        "meta": {
            "shard_dir": str(SHARD_DIR),
            "n_shards": 10,
            "integrals": INTEGRALS,
            "samples_seen": report.get("samples_seen"),
            "U": {"float": U_FLOAT, "ff": U_FF, "double": U_DOUBLE, "dd": U_DD},
            "b12_whole_app_floor": {"rel_err": B12_FLOOR_REL_ERR,
                                    "digits": B12_FLOOR_DIGITS},
        }
    }

    b12 = ints["B12"]
    regions = b12["regions"]
    rows = [region_row(loc, reg) for loc, reg in regions.items()]

    # hotspot region B2m.h:241 (confirm from data)
    hot_loc = "B2m.h:241"
    out["b12_hotspot_region"] = (region_row(hot_loc, regions[hot_loc])
                                 if hot_loc in regions else None)

    # top-10 by max_sensitivity and by max_rel_err
    out["b12_top10_by_sensitivity"] = sorted(
        rows, key=lambda r: -r["max_sensitivity"])[:10]
    out["b12_top10_by_measured_rel_err"] = sorted(
        rows, key=lambda r: -r["measured_max_rel_err"])[:10]

    # cascade chains (the accumulated-cancellation object the per-region bound
    # misses): top by sensitivity and by measured rel_err
    chains = b12.get("cascade_chains", [])
    crows = [chain_row(c) for c in chains]
    out["b12_n_cascade_chains"] = len(chains)
    out["b12_top10_chains_by_sensitivity"] = sorted(
        crows, key=lambda r: -r["max_sensitivity"])[:10]
    out["b12_top10_chains_by_measured_rel_err"] = sorted(
        crows, key=lambda r: -r["measured_max_rel_err"])[:10]

    # sanity check: 7 measured-INERT dd cells -> predicted_if_dd vs baseline delta
    sanity = []
    for integ, loc, baseline_delta, note in INERT_CELLS:
        reg = ints.get(integ, {}).get("regions", {}).get(loc)
        if reg is None:
            sanity.append({"integral": integ, "location": loc,
                           "found": False, "note": note})
            continue
        row = region_row(loc, reg)
        pdd = row["predicted_rel_err_if_dd"]
        sanity.append({
            "integral": integ, "location": loc, "found": True,
            "signal_class": row["signal_class"],
            "max_sensitivity": row["max_sensitivity"],
            "measured_max_rel_err": row["measured_max_rel_err"],
            "predicted_rel_err_if_double": row["predicted_rel_err_if_double"],
            "predicted_rel_err_if_dd": pdd,
            "baseline_dd_delta_measured": baseline_delta,
            # INERT is consistent if the region's dd prediction is at/below the
            # integral's achievable floor (widening it can't beat the delta):
            "dd_pred_below_baseline_delta": pdd <= baseline_delta,
            "note": note,
        })
    out["sanity_inert_cells"] = sanity

    outpath = Path(__file__).resolve().parent / "b12_bound_decomposition.json"
    outpath.write_text(json.dumps(out, indent=2, sort_keys=False))
    print(f"\nwrote {outpath}\n", flush=True)

    # ---- console summary ----
    def fmt(x):
        return f"{x:.3e}" if isinstance(x, (int, float)) and x else str(x)

    print("=== B12 hotspot region B2m.h:241 ===")
    h = out["b12_hotspot_region"]
    if h:
        for k in ("signal_class", "max_cond", "max_amp", "max_sensitivity",
                  "measured_max_rel_err", "predicted_rel_err_if_float",
                  "predicted_rel_err_if_ff", "predicted_rel_err_if_double",
                  "predicted_rel_err_if_dd", "tightness_double_over_measured"):
            print(f"  {k:34s}= {fmt(h[k])}")

    print("\n=== B12 top-5 regions by max_sensitivity ===")
    for r in out["b12_top10_by_sensitivity"][:5]:
        print(f"  {r['location']:20s} sens={fmt(r['max_sensitivity'])} "
              f"meas_re={fmt(r['measured_max_rel_err'])} "
              f"pred_dbl={fmt(r['predicted_rel_err_if_double'])} "
              f"pred_dd={fmt(r['predicted_rel_err_if_dd'])} "
              f"tight={fmt(r['tightness_double_over_measured'])} "
              f"[{r['signal_class']}]")

    print("\n=== B12 top-5 cascade chains by max_sensitivity ===")
    for r in out["b12_top10_chains_by_sensitivity"][:5]:
        print(f"  sens={fmt(r['max_sensitivity'])} "
              f"meas_re={fmt(r['measured_max_rel_err'])} "
              f"pred_dbl={fmt(r['predicted_rel_err_if_double'])} "
              f"pred_dd={fmt(r['predicted_rel_err_if_dd'])} "
              f"tight={fmt(r['tightness_double_over_measured'])} "
              f"nlines={len(r['chain_lines'])}")

    print("\n=== B12 top-5 cascade chains by measured rel_err ===")
    for r in out["b12_top10_chains_by_measured_rel_err"][:5]:
        print(f"  meas_re={fmt(r['measured_max_rel_err'])} "
              f"sens={fmt(r['max_sensitivity'])} "
              f"pred_dbl={fmt(r['predicted_rel_err_if_double'])} "
              f"pred_dd={fmt(r['predicted_rel_err_if_dd'])} "
              f"nlines={len(r['chain_lines'])} lines={r['chain_lines'][:4]}")

    print("\n=== Sanity: 7 measured-INERT dd cells ===")
    for s in sanity:
        if not s.get("found"):
            print(f"  {s['integral']:4s} {s['location']:20s} NOT FOUND")
            continue
        print(f"  {s['integral']:4s} {s['location']:20s} "
              f"sens={fmt(s['max_sensitivity'])} "
              f"meas_re={fmt(s['measured_max_rel_err'])} "
              f"pred_dd={fmt(s['predicted_rel_err_if_dd'])} "
              f"baseline_delta={fmt(s['baseline_dd_delta_measured'])} "
              f"dd_pred<=delta={s['dd_pred_below_baseline_delta']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
