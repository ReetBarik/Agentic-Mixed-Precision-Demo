"""Thin AMP wrapper over ``tracked_tools.reduce`` (the librarized reducer).

The generic core — JSONL streaming, id/scope-grammar parsing, per-sample DAG +
backward amplification, cascade-chain localization, LogHist, associative
merge, policy-parameterized finalize — lives in the Tracked library
(``third_party/tracked/tools``, installed as ``tracked_tools``).  This module
re-exports it with AMP's policy baked in, so every existing call site keeps
its historical signature and byte-identical output (differential-parity-tested
upstream against a frozen snapshot of the old monolith):

* :class:`ReducerConfig` defaults to AMP's prediction formats — IEEE single
  (``U_FLOAT``) plus the float-float emulation floor (``U_FF``) — and the
  float value-range guard.
* :func:`reduce_journal` / :func:`report_from_journals` sniff the journal
  version during the v0.3→v1 migration: a first record carrying a ``schema``
  key is read under the v1 rules (library ``docs/SCHEMA.md`` — header
  required, non-finite sentinels clamped to the alarm direction, additive
  ``nonfinite_records`` counter), anything else in legacy mode.  The strict
  library API requires an explicit choice; the sniff is AMP's transitional
  convenience and disappears once all corpora are v1.  Pass ``legacy=`` to
  pin a mode (contamination checks then hard-fail on a mismatch).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import tracked_tools.reduce.core as _core

# ---- generic core re-exports (behavior unchanged) ---------------------------

from tracked_tools.reduce.core import (  # noqa: F401
    DBL_MAX_CLAMP,
    FLT_MAX,
    FLT_MIN_NORMAL,
    LogHist,
    SCHEMA_VERSION,
    U_DOUBLE,
    U_FLOAT,
    merge_reports,
    read_journal,
    _ADD_SUB,
    _analyze_sample,
    _ancestors,
    _cancellation_ratio,
    _cascade_victims,
    _classify_region,
    _classify_variable,
    _cond,
    _cond_eff,
    _extract_cascade_chains,
    _integral_to_json,
    _is_cascade_contributor,
    _is_gate_a,
    _iter_samples,
    _max_opt,
    _merge_region,
    _merge_variable,
    _min_opt,
    _new_region,
    _new_region_json,
    _new_variable_json,
    _parse_region_span,
    _parse_scope,
    _prov_all,
    _prov_vars,
    _read_jsonl,
    _region_key,
    _region_local_reads,
    _rel_err,
    _scope_str,
    _short_hash,
    _signal_class,
    _topo_order,
    _update_region,
)

# Historical name for the library's 1/u saturation cap (gate-a).
ATAN2_SATURATION = _core.SATURATION_CAP

# float-float (double-single) unit roundoff.  ff carries ~2x float's mantissa
# (48 nominal bits), but the error-free-transformation emulation loses ~2 bits
# to the residual terms, so the *empirical* precision floor is ~2**-46
# (~1.42e-14, the ~14 documented digits) rather than the nominal 2**-48.  An
# AMP policy constant — the library core takes it through the predictions seam.
U_FF = 2.0 ** -46


def _amp_predictions() -> dict[str, float]:
    return {"float": U_FLOAT, "ff": U_FF}


@dataclass
class ReducerConfig(_core.ReducerConfig):
    """The library config with AMP's prediction formats as the default."""

    predictions: dict[str, float] = field(default_factory=_amp_predictions)


# ---- version sniffing (transitional, see module docstring) -------------------

def _sniff_legacy(path) -> bool:
    """True iff the journal's first non-empty record carries no v1 header."""
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                return "schema" not in json.loads(line)
            except json.JSONDecodeError:
                return True     # let the real reader raise its better error
    return True                 # empty file: reduces to empty either way


def reduce_journal(path, cfg: ReducerConfig | None = None, *,
                   legacy: bool | None = None) -> dict:
    """Reduce one journal file to a mergeable shard report (streaming)."""
    cfg = cfg or ReducerConfig()
    if legacy is None:
        legacy = _sniff_legacy(path)
    return _core.reduce_journal(path, cfg, legacy=legacy)


def finalize_report(merged: dict, cfg: ReducerConfig | None = None) -> dict:
    """Merged report -> consolidated (policy-neutral) report, AMP defaults."""
    return _core.finalize_report(merged, cfg or ReducerConfig())


def report_from_journals(paths: list, cfg: ReducerConfig | None = None, *,
                         legacy: bool | None = None) -> dict:
    """Convenience: reduce N shard journals, merge, finalize."""
    cfg = cfg or ReducerConfig()
    return finalize_report(
        merge_reports([reduce_journal(p, cfg, legacy=legacy) for p in paths]),
        cfg)


# ---- historical fixed-policy helpers -----------------------------------------

def _range_ok_for_float(reg: dict) -> bool:
    """Measured fact: do all recorded |val| at this region fit float's range?"""
    return _core._range_ok(reg, ReducerConfig())


def chain_range_ok_for_float(chain: dict, classified_regions: dict) -> bool:
    """A cascade chain is float-range-safe iff *every* contributor line is."""
    return _core.chain_range_ok(chain, classified_regions, ReducerConfig())


# ---------------------------------------------------------------------------
# CLI (unchanged surface)
# ---------------------------------------------------------------------------

def _write_json(obj: dict, path) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Characterizer stability reducer.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_red = sub.add_parser("reduce", help="reduce one journal -> shard report")
    p_red.add_argument("journal")
    p_red.add_argument("-o", "--out", required=True)

    p_rep = sub.add_parser("report", help="reduce+merge+finalize N journals -> report")
    p_rep.add_argument("journals", nargs="+")
    p_rep.add_argument("-o", "--out", required=True)

    p_mrg = sub.add_parser("merge", help="merge+finalize N shard reports -> report")
    p_mrg.add_argument("shards", nargs="+")
    p_mrg.add_argument("-o", "--out", required=True)

    args = ap.parse_args(argv)

    if args.cmd == "reduce":
        _write_json(reduce_journal(args.journal), args.out)
    elif args.cmd == "report":
        _write_json(report_from_journals(args.journals), args.out)
    elif args.cmd == "merge":
        shards = [json.loads(Path(s).read_text(encoding="utf-8")) for s in args.shards]
        _write_json(finalize_report(merge_reports(shards)), args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
