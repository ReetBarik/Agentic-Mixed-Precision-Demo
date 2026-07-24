"""Bound-decomposition arithmetic — the Item 6/7 first-order forward-cone model.

Extracted (Phase 2f) from ``runs/qcdloop/bound_decomposition_all_21.py`` so the
SAME arithmetic drives both the read-only analysis script AND the live chain-dd
candidate selection in the pipeline (``solver.queue`` / ``strategy.ranking``).

Framework (Item 6-revised — see the reducer header / bound_decomposition_all_21.py):
  * ``max_sensitivity = cond * amp`` — first-order forward-cone amplification of a
    machine-eps roundoff to the observable output.  cond and amp are properties of
    the *math function* -> precision-invariant; only the injected U changes per rung.
  * ``predicted_rel_err_if_<rung> = U_<rung> * max_sensitivity``.
  * Chain **tightness** = ``predicted_if_double / measured``.  Within [TIGHT_LO,
    TIGHT_HI] the first-order model EXPLAINS the measured error => the error is
    roundoff amplified by cancellation => **COMPUTED** (dd, which changes U by
    2^-53, recovers it).  Outside the band the chain cone cannot capture the error
    (analytic ill-conditioning, or a loose over-predicting bound).

This module holds ONLY the pure, precision-invariant arithmetic — no floor
derivation, no per-integral verdict, no I/O.  Those stay in the analysis script
(they are calibration heuristics, not the mechanism).
"""

from __future__ import annotations

import math

# Unit roundoff per rung.
U_FLOAT = 2.0 ** -24     # ~5.96e-8
U_FF = 2.0 ** -46        # ~1.42e-14  (reducer's empirical ff floor)
U_DOUBLE = 2.0 ** -53    # ~1.11e-16
U_DD = 2.0 ** -106       # ~1.23e-32

DOUBLE_DIGITS = 15.95    # ~ -log10(U_double); double's working precision in digits
DD_DIGITS = 31.9         # dd's working precision in digits
DD_LIFT_ORDERS = 15.95   # U_double / U_dd = 2^53 -> whole-chain dd drops the
                         # amplified error by this many orders (Item 6 §3).

# Chain-tightness band: the range in which the first-order bound explains the
# measured error, i.e. a genuine dd lever exists ("COMPUTED").
TIGHT_LO = 1e-3
TIGHT_HI = 1e1

# Integral-level floor thresholds (used by the analysis script's verdict; kept here
# so both consumers reference one source of truth).
STABLE_FLOOR = 12.0
BENIGN_FLOOR = 10.0


def predict(sens: float) -> dict:
    """The four ``predicted_rel_err_if_<rung>`` from a single ``max_sensitivity``."""
    return {
        "predicted_rel_err_if_float": U_FLOAT * sens,
        "predicted_rel_err_if_ff": U_FF * sens,
        "predicted_rel_err_if_double": U_DOUBLE * sens,
        "predicted_rel_err_if_dd": U_DD * sens,
    }


# Back-compat alias for the original private name used inside the analysis script.
_predict = predict


def chain_row(ch: dict) -> dict:
    """Decompose one raw cascade-chain dict (reducer output) into a bound row.

    ``ch`` is a ``cascade_chains[*]`` entry as emitted by
    ``stability_reducer.finalize_report`` — carries ``max_sensitivity``,
    ``max_rel_err``, ``chain`` (list of source spans), ``ops``, ``max_cond``.
    """
    sens = ch.get("max_sensitivity", 0.0)
    measured = ch.get("max_rel_err", 0.0)
    pred = predict(sens)
    pd = pred["predicted_rel_err_if_double"]
    return {
        "chain_id": ch.get("chain_id"),
        "signal_class": ch.get("signal_class"),
        "n_contributors": ch.get("n", 0),
        "chain_lines": [f"{s.get('file')}:{s.get('line_start')}"
                        for s in ch.get("chain", [])],
        "ops": ch.get("ops", {}),
        "max_cond": ch.get("max_cond", 0.0),
        "max_sensitivity": sens,
        "measured_max_rel_err": measured,
        **pred,
        "tightness_double_over_measured": (pd / measured) if measured > 0 else None,
    }


def chain_tightness(row: dict) -> float | None:
    """The tightness ratio ``predicted_if_double / measured`` for a chain row.

    ``None`` when the chain has no measured rel-err (nothing to explain)."""
    return row.get("tightness_double_over_measured")


def chain_is_computed(row: dict) -> bool:
    """True iff the chain's bound tightness is in the COMPUTED band [TIGHT_LO, TIGHT_HI].

    This is the Phase-2f chain-dd enqueue predicate (Reet 2026-07-24): qualify a
    chain by tightness ALONE — no integral-level ``floor < 10`` gate — and let the
    solver's positive-lift acceptance gate decide per candidate.
    """
    t = chain_tightness(row)
    return t is not None and TIGHT_LO <= t <= TIGHT_HI


def chain_predicted_lift(row: dict) -> float:
    """Predicted digits recovered by widening this chain from double to dd.

    Used ONLY to rank the chain-dd tier (biggest wins first); the real lift is
    measured by the Validator at accept time.  Derivation (Item 6 §3):

        digits_now      = -log10(measured_rel_err)            (current chain accuracy)
        digits_after_dd = min(DD_DIGITS, -log10(U_dd * sens)) (bound-predicted dd accuracy)
        lift            = max(0, digits_after_dd - digits_now)

    For a tight (COMPUTED) chain, ``measured ≈ U_double * sens`` so this collapses to
    ``min(DD_LIFT_ORDERS, DD_DIGITS - digits_now)`` — the whole-chain dd lift the
    Item-6/7 analysis reports.  Monotone in the chain's measured error, so it orders
    the tier the way we want (more cancellation -> promoted first).  Returns 0.0 when
    the chain has no measured error (nothing to recover).
    """
    measured = row.get("measured_max_rel_err", 0.0) or 0.0
    sens = row.get("max_sensitivity", 0.0) or 0.0
    if measured <= 0.0:
        return 0.0
    digits_now = -math.log10(measured)
    pred_dd = row.get("predicted_rel_err_if_dd")
    if pred_dd is None:
        pred_dd = U_DD * sens
    if pred_dd and pred_dd > 0.0:
        digits_after_dd = min(DD_DIGITS, -math.log10(pred_dd))
    else:
        digits_after_dd = DD_DIGITS
    return max(0.0, digits_after_dd - digits_now)
