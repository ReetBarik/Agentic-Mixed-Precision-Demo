#!/usr/bin/env python3
"""Item 7: bound decomposition across ALL 21 in-scope integrals.

Read-only over the existing 5k characterizer shards
(``/vast/projects/pepper_hep/qcdloop_shards_5k/``).  No runs, no builds, no LLM,
no >dd reference.  Reuses ``stability_reducer.merge_reports`` / ``finalize_report``
verbatim (the ``merge`` CLI path); this driver only loops the Item-6-revised
decomposition arithmetic (``b12_bound_decomposition.py``) over all 21 integrals
and adds a floor derivation + verdict + Tier-B target ranking.

Framework (identical to Item 6-revised — see that report / the reducer header):
  * ``max_sensitivity = cond * amp`` — first-order forward-cone amplification of a
    machine-eps roundoff to the observable output.  cond and amp are properties of
    the *math function* -> precision-invariant; only the injected U changes per rung.
  * ``predicted_rel_err_if_<rung> = U_<rung> * max_sensitivity``.
  * Chain tightness = ``predicted_if_double / measured`` at the dominant cascade
    chain.  ~1 (within ~2 orders) => the first-order model EXPLAINS the measured
    error => the error is roundoff amplified by cancellation => COMPUTED (dd, which
    changes U by 2^-53, recovers it).  <<1 even at chain scope => the chain cone
    cannot capture the error => analytic ill-conditioning => ALGORITHMIC.

Floor handling (per the Item-7 prompt):
  * 5 integrals have MEASURED per-integral solver floors on disk (B1/B8/B9/B10/B12,
    from the Phase-2b per-integral scorer manifests + B12 whole-app min).  Used
    verbatim; source tagged ``measured``.
  * The other 16 have no solver floor on disk -> a DERIVED estimate from the
    dominant-chain measured rel-err, using a monotone model CALIBRATED on the 5
    measured points (see ``_derive_floor``).  Source tagged ``derived`` (or
    ``derived_nochain`` for integrals with no cascade chains).  The derivation is
    coarse (the shard rel-err normalization blows up near cancellation
    zero-crossings, so it does not linearly track the whole-app floor) and is
    flagged as such; the MECHANISM verdict does not depend on the exact digit count.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from agents.shared.stability_reducer import merge_reports, finalize_report  # noqa: E402
# Phase 2f: the first-order bound arithmetic + constants now live in a shared module
# so the pipeline (solver.queue / strategy.ranking) uses the SAME math this script does.
from agents.shared.bound_decomposition import (  # noqa: E402
    U_FLOAT, U_FF, U_DOUBLE, U_DD,
    DOUBLE_DIGITS, DD_DIGITS, DD_LIFT_ORDERS,
    STABLE_FLOOR, BENIGN_FLOOR, TIGHT_LO, TIGHT_HI,
    _predict, chain_row,
)

SHARD_DIR = Path("/vast/projects/pepper_hep/qcdloop_shards_5k")
CACHE = Path(__file__).resolve().parent / ".merged_all_21.cache.json"

# All 21 in-scope integrals.
INTEGRALS = [f"B{i}" for i in range(1, 17)] + [f"BIN{i}" for i in range(5)]

# ---- MEASURED per-integral solver floors (source of truth for these 5) -------
# digits, whole-integral rel-err delta, provenance.  From the Phase-2b per-integral
# scorer manifests (max baseline_delta_effective -> digits) and, for B12, the
# whole-app min (sample 3868, coeff0.imag) used throughout Items 3/5/6.
MEASURED_FLOORS = {
    "B1":  (12.158, 6.942e-13, "per-integral scorer (Phase 2b): B0m.h:118/124/125 baseline delta"),
    "B8":  (10.139, 7.266e-11, "per-integral scorer: kokkosUtils.h dilog branches (DD_TRIAGE benign_no_op)"),
    "B9":  (11.530, 2.950e-12, "per-integral scorer: kokkosUtils.h:212 (DD_TRIAGE benign_no_op)"),
    "B10": (9.878,  1.324e-10, "per-integral scorer: kokkosUtils.h:212 (DD_TRIAGE physics_ceiling)"),
    "B12": (3.691,  2.039e-4,  "solver Stage-1 / whole-app min (sample 3868, coeff0.imag)"),
}

# Verdict thresholds (STABLE_FLOOR / BENIGN_FLOOR / TIGHT_LO / TIGHT_HI) and the
# bound arithmetic (_predict / chain_row) are imported from
# agents.shared.bound_decomposition (Phase 2f — one source of truth).


def _derive_floor(dom_chain_re: float, max_reg_re: float, has_chains: bool):
    """Coarse whole-app floor estimate for an integral with no measured floor.

    Calibrated on the 5 measured points (see report).  The dominant-chain measured
    rel-err ``l = log10(chain_re)`` is a MONOTONE predictor of the measured floor
    across those 5 points; the shape is: flat (~benign ceiling) for mild
    cancellation, then ~1 digit lost per decade of chain rel-err once cancellation
    is catastrophic (l>1).

        floor(l) = 10.0 - 1.05 * (l - 1)              for l > 1
        floor(l) = min(11.5, 10.0 + 0.4 * (1 - l))    for l <= 1

    Calibration check (measured floors in parens): B12 l=7.21 -> 3.48 (3.69);
    B10 l=1.29 -> 9.70 (9.88); B8 l=-2.22 -> 11.3 (10.14); B9 l=-3.37 -> 11.5 (11.53).

    For integrals with NO cascade chains, cancellation is absent; the floor is set
    by benign forward-amplified roundoff.  ``max_reg_re`` under-states the true
    floor (calibrated on B1: -log10(1.77e-9)=8.75 vs measured 12.16, a +3.4 gap),
    so we report ``-log10(max_reg_re) + 3.4`` as a conservative lower estimate.

    Returns (floor_digits, uncapped_loss_digits).  ``uncapped_loss`` = the
    cancellation digit-loss WITHOUT flooring the double result at 0, so a caller
    can tell whether the loss exceeds dd's capacity (whole-chain dd insufficient).
    """
    if not has_chains:
        base = (-math.log10(max_reg_re) + 3.4) if max_reg_re and max_reg_re > 0 else DOUBLE_DIGITS
        return min(DD_DIGITS, max(0.0, base)), 0.0
    if dom_chain_re is None or dom_chain_re <= 0:
        return None, None
    l = math.log10(dom_chain_re)
    if l > 1.0:
        uncapped_loss = 6.0 + 1.05 * (l - 1.0)      # digits lost to cancellation
        floor = DOUBLE_DIGITS - uncapped_loss
    else:
        floor = min(11.5, 10.0 + 0.4 * (1.0 - l))
        uncapped_loss = DOUBLE_DIGITS - floor
    return max(0.0, floor), uncapped_loss


def _verdict(floor: float, floor_source: str, has_chains: bool,
             tightness, dom_chain_re: float):
    """COMPUTED / ALGORITHMIC / STABLE_ALREADY / INCONCLUSIVE + a one-line reason."""
    if floor is None:
        return "INCONCLUSIVE", "no floor could be placed (no chains, no region rel-err)"
    if not has_chains:
        return "STABLE_ALREADY", (f"no cascade chains; well-conditioned "
                                  f"(floor ~{floor:.1f} digits, no cancellation lever)")
    # Floor-first: a high floor needs no lift regardless of the bound shape.  Only
    # integrals whose floor is genuinely low (< BENIGN_FLOOR) are candidates for a
    # COMPUTED/ALGORITHMIC split; a loose bound on a *tiny* measured error just means
    # the chain barely cancels (benign), not that the ill-conditioning is analytic.
    if floor >= STABLE_FLOOR:
        return "STABLE_ALREADY", f"floor {floor:.1f} >= {STABLE_FLOOR:.0f} digits; no lift needed"
    if floor >= BENIGN_FLOOR:
        return "STABLE_ALREADY", (f"floor {floor:.1f} >= {BENIGN_FLOOR:.0f} digits (benign cancellation; "
                                  f"DD_TRIAGE-class dd-inert), no lift warranted")
    # floor < BENIGN_FLOOR => the integral genuinely needs a lift; place the mechanism.
    if tightness is None:
        return "INCONCLUSIVE", "low floor but dominant chain has no measured rel-err"
    if tightness > TIGHT_HI:
        return "INCONCLUSIVE", (f"low floor but dominant-chain bound over-predicts "
                                f"(tightness {tightness:.1e}); amp cone loose-high, not a clean driver")
    if tightness < TIGHT_LO:
        return "ALGORITHMIC", (f"low floor and dominant-chain bound loose (tightness {tightness:.1e} < "
                               f"{TIGHT_LO:.0e}); error not captured by any chain cone -> analytic")
    return "COMPUTED", (f"tight dominant-chain bound (tightness {tightness:.3f}); "
                        f"catastrophic cancellation IS roundoff amplification -> dd lever exists")


def load_and_merge() -> dict:
    if CACHE.exists():
        print(f"loading cached merged report {CACHE.name}", flush=True)
        return json.loads(CACHE.read_text())
    shards = sorted(SHARD_DIR.glob("shard_*.json"))
    if not shards:
        raise SystemExit(f"no shards under {SHARD_DIR}")
    reps = []
    for sp in shards:
        t0 = time.time()
        reps.append(json.load(open(sp)))
        print(f"  loaded {sp.name} ({time.time()-t0:.1f}s)", flush=True)
    print(f"merging {len(reps)} shards (reusing merge_reports/finalize_report)...", flush=True)
    rep = finalize_report(merge_reports(reps))
    CACHE.write_text(json.dumps(rep))
    return rep


def main() -> int:
    report = load_and_merge()
    ints = report["integrals"]
    samples = report.get("samples_seen", {})

    per_integral = {}
    for I in INTEGRALS:
        d = ints.get(I)
        if d is None:
            per_integral[I] = {"present": False}
            continue
        regs = d["regions"]
        chains = d.get("cascade_chains", [])
        max_reg_re = max([r.get("max_rel_err", 0.0) for r in regs.values()] + [0.0])
        casc = [c for c in chains if c.get("signal_class") == "cancellation_cascade"]

        # dominant floor-driving chain = cascade chain with max measured rel-err.
        dom = max(chains, key=lambda c: c.get("max_rel_err", 0.0)) if chains else None
        dom_row = chain_row(dom) if dom else None
        dom_re = dom.get("max_rel_err", 0.0) if dom else None
        tight = dom_row["tightness_double_over_measured"] if dom_row else None

        # top-5 chains by measured rel-err (distinct floor-driving objects).
        top5 = sorted((chain_row(c) for c in chains),
                      key=lambda r: -r["measured_max_rel_err"])[:5]

        has_chains = len(chains) > 0
        # floor: measured if on disk, else derived.
        if I in MEASURED_FLOORS:
            fl, delta, prov = MEASURED_FLOORS[I]
            floor, floor_source, floor_note = fl, "measured", prov
            _, uncapped_loss = _derive_floor(dom_re, max_reg_re, has_chains)
            # use MEASURED floor to set the loss (more reliable than the model here)
            uncapped_loss = DOUBLE_DIGITS - fl
        else:
            floor, uncapped_loss = _derive_floor(dom_re, max_reg_re, has_chains)
            floor_source = "derived" if has_chains else "derived_nochain"
            floor_note = ("model on dominant-chain rel-err (coarse; calibrated on 5 measured pts)"
                          if has_chains else "no cascade chains; -log10(max_region_rel_err)+3.4 lower est")

        verdict, reason = _verdict(floor, floor_source, has_chains, tight, dom_re)

        # Confidence in the FLOOR number.  The derivation model is calibrated on 5
        # measured points spanning log10(chain_re) in [-3.4, 7.2]; a dominant chain
        # rel-err beyond that upper edge is extrapolated (the extreme integrals),
        # so the digit count is low-confidence even though the *mechanism* verdict
        # (tight bound => COMPUTED) is robust.
        CALIB_MAX_LOG = 7.3
        if floor_source == "measured":
            confidence = "measured"
        elif floor_source == "derived_nochain":
            confidence = "derived_nochain"
        elif dom_re and math.log10(dom_re) <= CALIB_MAX_LOG:
            confidence = "derived_in_calibration_range"
        else:
            confidence = "derived_extrapolated"

        # predicted whole-chain-dd floor + whether dd suffices.
        pred_dd_floor = None
        dd_sufficient = None
        if uncapped_loss is not None:
            pdf = DD_DIGITS - uncapped_loss
            pred_dd_floor = pdf
            dd_sufficient = pdf >= BENIGN_FLOOR

        per_integral[I] = {
            "present": True,
            "samples": samples.get(I, 0),
            "n_regions": len(regs),
            "n_cascade_chains": len(chains),
            "n_cancellation_cascade_chains": len(casc),
            "max_region_rel_err": max_reg_re,
            "floor_digits": round(floor, 3) if floor is not None else None,
            "floor_source": floor_source,
            "floor_confidence": confidence,
            "floor_note": floor_note,
            "dominant_chain": dom_row,
            "dominant_chain_tightness": tight,
            "cancellation_loss_digits": round(uncapped_loss, 2) if uncapped_loss is not None else None,
            "predicted_dd_floor_digits": round(pred_dd_floor, 2) if pred_dd_floor is not None else None,
            "dd_sufficient": dd_sufficient,
            "predicted_dd_floor_lift": (round(pred_dd_floor - floor, 2)
                                        if (pred_dd_floor is not None and floor is not None) else None),
            "verdict": verdict,
            "verdict_reason": reason,
            "top5_chains_by_measured_rel_err": top5,
        }

    # ---- Tier-B target list: COMPUTED integrals ranked by predicted floor lift --
    tierb = []
    for I, d in per_integral.items():
        if not d.get("present"):
            continue
        if d["verdict"] != "COMPUTED":
            continue
        tierb.append({
            "integral": I,
            "floor_digits": d["floor_digits"],
            "floor_source": d["floor_source"],
            "floor_confidence": d["floor_confidence"],
            "predicted_dd_floor_digits": d["predicted_dd_floor_digits"],
            "predicted_dd_floor_lift": d["predicted_dd_floor_lift"],
            "dd_sufficient": d["dd_sufficient"],
            "dominant_chain_lines": d["dominant_chain"]["chain_lines"] if d["dominant_chain"] else [],
            "dominant_chain_tightness": d["dominant_chain_tightness"],
        })
    tierb.sort(key=lambda t: -(t["predicted_dd_floor_lift"] or 0.0))

    out = {
        "meta": {
            "item": "Phase 2e Item 7 — bound decomposition across all 21 integrals",
            "shard_dir": str(SHARD_DIR),
            "n_shards": 10,
            "samples_seen": samples,
            "U": {"float": U_FLOAT, "ff": U_FF, "double": U_DOUBLE, "dd": U_DD},
            "double_digits": DOUBLE_DIGITS, "dd_digits": DD_DIGITS,
            "dd_lift_orders": DD_LIFT_ORDERS,
            "measured_floors": {k: {"digits": v[0], "rel_err": v[1], "source": v[2]}
                                for k, v in MEASURED_FLOORS.items()},
            "verdict_thresholds": {"stable_floor": STABLE_FLOOR, "benign_floor": BENIGN_FLOOR,
                                   "tight_lo": TIGHT_LO, "tight_hi": TIGHT_HI},
        },
        "per_integral": per_integral,
        "tierb_targets_ranked": tierb,
    }
    outpath = Path(__file__).resolve().parent / "bound_decomposition_all_21.json"
    outpath.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outpath}\n", flush=True)

    # ---- console table ----
    def f(x):
        return f"{x:.2e}" if isinstance(x, (int, float)) and x else ("0" if x == 0 else str(x))
    print(f"{'I':5s} {'floor':>6s} {'src':>14s} {'domRE':>9s} {'tight':>7s} "
          f"{'loss':>6s} {'ddFloor':>7s} {'ddOK':>5s} {'verdict':>14s}")
    for I in INTEGRALS:
        d = per_integral[I]
        if not d.get("present"):
            print(f"{I:5s}  NOT PRESENT"); continue
        dc = d["dominant_chain"]
        print(f"{I:5s} {str(d['floor_digits']):>6s} {d['floor_source']:>14s} "
              f"{f(dc['measured_max_rel_err']) if dc else '-':>9s} "
              f"{(round(d['dominant_chain_tightness'],3) if d['dominant_chain_tightness'] else '-'):>7} "
              f"{str(d['cancellation_loss_digits']):>6s} "
              f"{str(d['predicted_dd_floor_digits']):>7s} "
              f"{str(d['dd_sufficient']):>5s} {d['verdict']:>14s}")

    print("\n=== Tier-B targets (COMPUTED, ranked by predicted dd floor lift) ===")
    for t in tierb:
        print(f"  {t['integral']:5s} {t['floor_digits']}->{t['predicted_dd_floor_digits']} "
              f"(lift +{t['predicted_dd_floor_lift']}, dd_ok={t['dd_sufficient']}) "
              f"lines={t['dominant_chain_lines']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
