"""Journal -> mergeable code-region stability report (the characterizer reducer).

This is the *map* step of the sharded characterizer: each shard runs the tracked
whole-app binary over a disjoint slice of the input space, then reduces its own
(possibly enormous, transient) ``journal.jsonl`` **in-process** to a small,
mergeable report.  A separate *merge* step combines the per-shard reports into a
single consolidated report handed to the Strategy Agent.  The journal itself is
never moved or concatenated — only the reductions are (see PLAN_implementation.md
"Execution model": "per-chunk metadata ... reduces cleanly across chunks").

Two things this computes that a flat per-line rollup cannot:

* **Forward cone / amplification.**  Downcast safety is a *forward* dataflow
  property: raising a value's error floor to float's ``u`` (2**-24) is safe only
  if every path from it to an observable output attenuates the injected error
  below the acceptance margin.  We build the per-sample DAG from each record's
  ``in`` operand edges, invert them, and run one backward pass computing, for
  every node at once, ``amp(v) = max over consumers c of cond(c) * amp(v)``
  (``amp = 1`` at output sinks).  ``amp`` is a conservative upper bound (it
  ignores the ``max``-gating in the real error recurrence, so it can over-flag
  danger — the safe direction for a downcast guard).

* **Value-range guard.**  float has a narrower exponent range than double; a
  well-conditioned value that underflows/overflows float is unsafe to downcast
  for a reason the error model doesn't see.  We track min/max ``|val|`` per
  region from the recorded ``val``.

Signal classes follow the Stage-2 taxonomy (three mechanistically distinct
failure modes + a stable class + the documented atan2 saturation cap).  The
class is a *mechanistic* description of the error phenomenon, not a remediation:
the reducer is policy-neutral and emits measurements only (``max_amp``,
``predicted_rel_err_if_float``, ``value_range_ok_for_float``, class).  Applying
the acceptance margin and choosing the downcast/keep/upgrade action is the
Strategy Agent's job, so one characterization run serves any acceptance policy.

The reducer targets the v0.3 journal schema (records carry ``id``, ``in``,
``prov_vars``/``prov_consts``, and the ``@integral=<name>/sample=<i>`` scope
suffix appended after the ``#<callsite-counter>``).  Older journals without
``id`` cannot supply the DAG; they degrade to per-location aggregation with no
forward-cone signal (and a note).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

# ---------------------------------------------------------------------------
# Numeric constants
# ---------------------------------------------------------------------------

U_DOUBLE = 2.0 ** -53          # ~1.11e-16
U_FLOAT = 2.0 ** -24           # ~5.96e-8
FLT_MIN_NORMAL = 1.1754943508222875e-38
FLT_MAX = 3.4028234663852886e38
ATAN2_SATURATION = 2.0 ** 53   # documented gate-(a) cap (ops.hpp atan2 at 1/u)


@dataclass
class ReducerConfig:
    """Thresholds for the *mechanistic* signal classification.

    These boundaries describe the KIND of numerical phenomenon (ill-conditioned
    op, cancellation cascade, local cancellation) — they are properties of the
    error mechanism, not of any acceptance policy.  The reducer is deliberately
    policy-neutral: it emits measured quantities (``max_amp``,
    ``predicted_rel_err_if_float``, ``value_range_ok_for_float``) and this
    mechanistic class, and leaves the acceptance margin (required digits) and the
    downcast/keep/upgrade *decision* to the Strategy Agent.  So one (expensive,
    sharded) characterization run serves any acceptance policy without re-running.
    """

    local_cancel_cond: float = 1e15      # cond>this (post gate-a) => local cancellation
    high_cond: float = 1e6               # cond in [high_cond, local_cancel) => log-near-root
    cascade_rel_err: float = 1e-6        # rel_err>this with low cond => cancellation cascade
    cascade_cond_ceiling: float = 1e3    # "low per-op cond" ceiling for cascade detection
    gate_a_rel_tol: float = 1e-9         # |cond - 2**53| / 2**53 tolerance for gate-(a)


SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Mergeable log10 histogram (approximate, exactly-additive percentiles)
# ---------------------------------------------------------------------------

class LogHist:
    """Sparse base-10 log histogram over positive values.

    Bucketed by ``floor(log10(x))`` (one decade per bucket).  Bucket counts are
    exactly additive, so a percentile read from the merged histogram equals the
    percentile of the concatenated sample set (to one-decade resolution).  This
    is the mergeable quantile sketch the p99 acceptance metric needs — you
    cannot average shard percentiles.
    """

    __slots__ = ("buckets", "total")

    def __init__(self) -> None:
        self.buckets: dict[int, int] = {}
        self.total: int = 0

    def add(self, x: float) -> None:
        if x is None or not math.isfinite(x) or x <= 0.0:
            return
        b = math.floor(math.log10(x))
        self.buckets[b] = self.buckets.get(b, 0) + 1
        self.total += 1

    def merge(self, other: "LogHist") -> None:
        for b, c in other.buckets.items():
            self.buckets[b] = self.buckets.get(b, 0) + c
        self.total += other.total

    def quantile(self, q: float) -> float | None:
        """Approximate q-quantile as the lower edge of the crossing decade."""
        if self.total == 0:
            return None
        target = q * self.total
        cum = 0
        for b in sorted(self.buckets):
            cum += self.buckets[b]
            if cum >= target:
                return 10.0 ** b
        return 10.0 ** max(self.buckets)

    def to_dict(self) -> dict:
        return {"buckets": {str(k): v for k, v in self.buckets.items()},
                "total": self.total}

    @classmethod
    def from_dict(cls, d: dict) -> "LogHist":
        h = cls()
        h.buckets = {int(k): v for k, v in d.get("buckets", {}).items()}
        h.total = d.get("total", sum(h.buckets.values()))
        return h


# ---------------------------------------------------------------------------
# Record / scope parsing
# ---------------------------------------------------------------------------

def _read_jsonl(path: Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Bad JSON on line {lineno} of {path}: {exc}") from exc


def _scope_str(node_id: str) -> str:
    """Extract the scope suffix from an op id, or "" if unscoped.

    Ids look like ``<op>@<file>:<line>#<counter>[@<scope>]``.  The scope is
    appended after the ``#<counter>``, so we split on the first ``#`` (the file
    part never contains one) and take whatever follows the next ``@``.
    """
    if not node_id:
        return ""
    hash_idx = node_id.find("#")
    if hash_idx == -1:
        return ""
    rest = node_id[hash_idx + 1:]
    at = rest.find("@")
    return rest[at + 1:] if at != -1 else ""


def _parse_scope(scope: str) -> dict[str, str]:
    """``"integral=B15/sample=42"`` -> ``{"integral": "B15", "sample": "42"}``."""
    out: dict[str, str] = {}
    for part in scope.split("/"):
        if "=" in part:
            k, _, v = part.partition("=")
            out[k] = v
    return out


def _prov_vars(rec: dict) -> list[str]:
    """Source-variable provenance, tolerant of schema drift.

    v0.3 splits provenance into ``prov_vars`` (source roots) + ``prov_consts``
    (named constants).  Older journals used a single flat ``prov``.  The old
    log_parser read only ``prov`` and so came up empty on v0.3 journals — this is
    the documented fix.
    """
    if "prov_vars" in rec:
        return list(rec.get("prov_vars") or [])
    return list(rec.get("prov") or rec.get("provenance") or [])


def _prov_all(rec: dict) -> list[str]:
    if "prov_vars" in rec or "prov_consts" in rec:
        return list(rec.get("prov_vars") or []) + list(rec.get("prov_consts") or [])
    return list(rec.get("prov") or rec.get("provenance") or [])


def _cond(rec: dict) -> float:
    try:
        c = float(rec.get("cond", 0.0))
    except (TypeError, ValueError):
        return 0.0
    return c if math.isfinite(c) and c > 0.0 else 0.0


def _is_gate_a(cond: float, cfg: ReducerConfig) -> bool:
    """True for the documented atan2 saturation cap at 1/u = 2**53."""
    if cond <= 0.0:
        return False
    return abs(cond - ATAN2_SATURATION) <= cfg.gate_a_rel_tol * ATAN2_SATURATION


def _sample_key(scope_str: str) -> str:
    """The sample identity: the scope MINUS any ``line=`` component.

    A ``line=<file:line>`` sub-scope (pushed around a source statement to make
    it a code region — see the module header) changes the id suffix op-to-op
    *within* one sample.  Sample grouping and the whole-sample DAG must ignore
    it, or the accumulation's operands (computed on earlier lines, outside the
    line scope) would be split into a different batch and mis-read as sources.
    """
    d = _parse_scope(scope_str)
    return "/".join(f"{k}={v}" for k, v in d.items() if k != "line")


def _region_key(rec: dict) -> str:
    """The code-region a record belongs to: ``line=`` scope, else ``at``.

    ``line=`` (injected as a scope, so it lands on operator ops too) is the
    primary code-region signal for operator-heavy libraries; ``at`` (a real
    ``file:fn:line`` from a located named call) is used when present; ""
    otherwise (unattributed — the operator arithmetic with no line scope).
    """
    line = _parse_scope(_scope_str(rec.get("id", ""))).get("line")
    return line or rec.get("at", "") or ""


def _iter_samples(records: Iterable[dict]) -> Iterator[tuple[str, list[dict]]]:
    """Group a stream of records into contiguous per-sample batches.

    The consolidated driver emits each ``(integral, sample)`` fully before the
    next (RAII scope, Serial backend, append-ordered flush), so grouping runs of
    equal *sample key* recovers per-sample batches without loading the whole
    journal.  Grouping is on the sample key (scope minus ``line=``) so a line
    sub-scope pushed mid-sample does not fragment the batch.
    """
    cur_key: str | None = None
    batch: list[dict] = []
    for rec in records:
        key = _sample_key(_scope_str(rec.get("id", "")))
        if cur_key is None:
            cur_key = key
        if key != cur_key:
            yield cur_key, batch
            batch = []
            cur_key = key
        batch.append(rec)
    if batch:
        yield cur_key or "", batch


# ---------------------------------------------------------------------------
# Per-sample DAG + forward-cone amplification
# ---------------------------------------------------------------------------

def _topo_order(nodes: dict[str, dict]) -> list[str]:
    """Dependency-topological order (operands before the ops that consume them).

    Iterative DFS post-order over the internal ``in`` edges; robust to the
    per-sample DAG being a forest and to deep cascade chains (no recursion).
    """
    visited: set[str] = set()
    order: list[str] = []
    for root in nodes:
        if root in visited:
            continue
        stack: list[tuple[str, bool]] = [(root, False)]
        while stack:
            node, processed = stack.pop()
            if processed:
                order.append(node)
                continue
            if node in visited:
                continue
            visited.add(node)
            stack.append((node, True))
            for o in nodes[node].get("in", []):
                if o in nodes and o not in visited:
                    stack.append((o, False))
    return order


def _analyze_sample(records: list[dict], cfg: ReducerConfig):
    """Build the sample DAG and compute amplification for every node.

    Returns ``(nodes, amp, node_sens, source_sens, source_ids)`` where:
      * ``amp[v]``        forward amplification of an error at v to any output
      * ``node_sens[v]``  downcast impact factor = cond(v) * amp(v) for op v
      * ``source_sens[s]``downcast impact factor of a record-less source id s
      * ``source_ids``    operand ids with no record (track vars / consts / lits)

    Nodes without an ``id`` (v0.2 journals) yield an empty analysis — the caller
    falls back to per-location aggregation only.
    """
    nodes: dict[str, dict] = {}
    for r in records:
        rid = r.get("id")
        if rid is not None:
            nodes[rid] = r
    if not nodes:
        return {}, {}, {}, {}, set()

    children: dict[str, list[str]] = {rid: [] for rid in nodes}
    source_ids: set[str] = set()
    for rid, r in nodes.items():
        for o in r.get("in", []):
            if o in nodes:
                children[o].append(rid)
            else:
                source_ids.add(o)

    order = _topo_order(nodes)

    # Backward amplification pass: consumers before the node (reversed topo).
    amp: dict[str, float] = {}
    for v in reversed(order):
        ch = children[v]
        if not ch:
            amp[v] = 1.0                     # output sink (no internal consumer)
        else:
            best = 0.0
            for c in ch:
                cand = _cond_eff(nodes[c]) * amp[c]
                if cand > best:
                    best = cand
            amp[v] = best if best > 0.0 else 1.0

    node_sens: dict[str, float] = {}
    source_sens: dict[str, float] = {}
    for rid, r in nodes.items():
        impact = _cond_eff(r) * amp[rid]
        node_sens[rid] = impact
        for o in r.get("in", []):
            if o in source_ids:
                # A float floor at source o reaches r as cond(r)*u, then amp(r).
                if impact > source_sens.get(o, 0.0):
                    source_sens[o] = impact
    return nodes, amp, node_sens, source_sens, source_ids


def _cond_eff(rec: dict) -> float:
    """Effective local cond for amplification: a real cond, or 1 as a floor.

    Ops with cond <= 0 recorded (mul/div use cond=1; some emit 0) still pass
    error through, so a unit floor keeps them in the amplification chain rather
    than zeroing a downstream cone.  gate-(a) saturation is left as-is here (it
    genuinely amplifies) — it is only excluded from the *reported* max_cond.
    """
    c = _cond(rec)
    return c if c > 0.0 else 1.0


# ---------------------------------------------------------------------------
# Aggregation (the mergeable shard report)
# ---------------------------------------------------------------------------

def _new_region() -> dict:
    return {
        "ops": {},
        "n": 0,
        "max_cond": 0.0,
        "gate_a_count": 0,
        "max_rel_err": 0.0,
        "rel_err_hist": LogHist(),
        "max_sensitivity": 0.0,   # max cond*amp over ops at this location
        "max_amp": 0.0,
        "abs_val_min": None,
        "abs_val_max": None,
        "prov_vars": set(),
    }


def _update_region(reg: dict, rec: dict, amp_v: float, sens_v: float,
                   cfg: ReducerConfig) -> None:
    reg["n"] += 1
    op = rec.get("op", "unknown")
    reg["ops"][op] = reg["ops"].get(op, 0) + 1

    cond = _cond(rec)
    if _is_gate_a(cond, cfg):
        reg["gate_a_count"] += 1
    elif cond > reg["max_cond"]:
        reg["max_cond"] = cond

    try:
        rel = float(rec.get("rel_err", 0.0))
    except (TypeError, ValueError):
        rel = 0.0
    if math.isfinite(rel) and rel > 0.0:
        if rel > reg["max_rel_err"]:
            reg["max_rel_err"] = rel
        reg["rel_err_hist"].add(rel)

    if sens_v > reg["max_sensitivity"]:
        reg["max_sensitivity"] = sens_v
    if amp_v > reg["max_amp"]:
        reg["max_amp"] = amp_v

    try:
        val = abs(float(rec.get("val", 0.0)))
    except (TypeError, ValueError):
        val = 0.0
    if math.isfinite(val) and val > 0.0:
        if reg["abs_val_min"] is None or val < reg["abs_val_min"]:
            reg["abs_val_min"] = val
        if reg["abs_val_max"] is None or val > reg["abs_val_max"]:
            reg["abs_val_max"] = val

    reg["prov_vars"].update(_prov_vars(rec))


def reduce_journal(path, cfg: ReducerConfig | None = None) -> dict:
    """Reduce one journal file to a mergeable shard report (streaming)."""
    cfg = cfg or ReducerConfig()
    integrals: dict[str, dict] = {}
    samples_seen: dict[str, int] = {}
    no_id_records = 0

    for scope, batch in _iter_samples(_read_jsonl(Path(path))):
        integral = _parse_scope(scope).get("integral", "")
        nodes, amp, node_sens, source_sens, source_ids = _analyze_sample(batch, cfg)
        if not nodes:
            no_id_records += len(batch)
            continue

        samples_seen[integral] = samples_seen.get(integral, 0) + 1
        I = integrals.setdefault(integral, {"regions": {}, "variables": {}})

        prov_var_names: set[str] = set()
        for r in nodes.values():
            prov_var_names.update(_prov_vars(r))

        for rid, r in nodes.items():
            loc = _region_key(r)
            reg = I["regions"].setdefault(loc, _new_region())
            _update_region(reg, r, amp[rid], node_sens[rid], cfg)

        for sid, sens in source_sens.items():
            var = I["variables"].setdefault(
                sid, {"max_sensitivity": 0.0, "max_amp": 0.0,
                      "n_consumers": 0, "is_source_var": sid in prov_var_names})
            if sens > var["max_sensitivity"]:
                var["max_sensitivity"] = sens
            if sens > var["max_amp"]:
                var["max_amp"] = sens
            var["n_consumers"] += 1
            var["is_source_var"] = var["is_source_var"] or (sid in prov_var_names)

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "stability_shard_report",
        "samples_seen": samples_seen,
        "no_id_records": no_id_records,
        "integrals": {name: _integral_to_json(data) for name, data in integrals.items()},
    }


def _integral_to_json(data: dict) -> dict:
    regions = {}
    for loc, reg in data["regions"].items():
        r = dict(reg)
        r["rel_err_hist"] = reg["rel_err_hist"].to_dict()
        r["prov_vars"] = sorted(reg["prov_vars"])
        regions[loc] = r
    return {"regions": regions, "variables": data["variables"]}


# ---------------------------------------------------------------------------
# Merge (combine shard reports)
# ---------------------------------------------------------------------------

def merge_reports(reports: list[dict]) -> dict:
    """Combine shard reports into one merged report (associative, order-free).

    ``merge([reduce(A), reduce(B)]) == reduce(A ++ B)`` for every aggregate:
    maxes via ``max``, counts/hist via addition, value ranges via min/max, sets
    via union.
    """
    out_samples: dict[str, int] = {}
    out_integrals: dict[str, dict] = {}
    no_id = 0

    for rep in reports:
        no_id += rep.get("no_id_records", 0)
        for name, cnt in rep.get("samples_seen", {}).items():
            out_samples[name] = out_samples.get(name, 0) + cnt
        for name, idata in rep.get("integrals", {}).items():
            dst = out_integrals.setdefault(name, {"regions": {}, "variables": {}})
            for loc, reg in idata.get("regions", {}).items():
                _merge_region(dst["regions"].setdefault(loc, _new_region_json()), reg)
            for vid, var in idata.get("variables", {}).items():
                _merge_variable(dst["variables"].setdefault(vid, _new_variable_json()), var)

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "stability_merged_report",
        "samples_seen": out_samples,
        "no_id_records": no_id,
        "integrals": out_integrals,
    }


def _new_region_json() -> dict:
    return {"ops": {}, "n": 0, "max_cond": 0.0, "gate_a_count": 0,
            "max_rel_err": 0.0, "rel_err_hist": {"buckets": {}, "total": 0},
            "max_sensitivity": 0.0, "max_amp": 0.0,
            "abs_val_min": None, "abs_val_max": None, "prov_vars": []}


def _new_variable_json() -> dict:
    return {"max_sensitivity": 0.0, "max_amp": 0.0, "n_consumers": 0,
            "is_source_var": False}


def _merge_region(dst: dict, src: dict) -> None:
    for op, c in src.get("ops", {}).items():
        dst["ops"][op] = dst["ops"].get(op, 0) + c
    dst["n"] += src.get("n", 0)
    dst["max_cond"] = max(dst["max_cond"], src.get("max_cond", 0.0))
    dst["gate_a_count"] += src.get("gate_a_count", 0)
    dst["max_rel_err"] = max(dst["max_rel_err"], src.get("max_rel_err", 0.0))
    h = LogHist.from_dict(dst["rel_err_hist"])
    h.merge(LogHist.from_dict(src.get("rel_err_hist", {})))
    dst["rel_err_hist"] = h.to_dict()
    dst["max_sensitivity"] = max(dst["max_sensitivity"], src.get("max_sensitivity", 0.0))
    dst["max_amp"] = max(dst["max_amp"], src.get("max_amp", 0.0))
    dst["abs_val_min"] = _min_opt(dst["abs_val_min"], src.get("abs_val_min"))
    dst["abs_val_max"] = _max_opt(dst["abs_val_max"], src.get("abs_val_max"))
    dst["prov_vars"] = sorted(set(dst["prov_vars"]) | set(src.get("prov_vars", [])))


def _merge_variable(dst: dict, src: dict) -> None:
    dst["max_sensitivity"] = max(dst["max_sensitivity"], src.get("max_sensitivity", 0.0))
    dst["max_amp"] = max(dst["max_amp"], src.get("max_amp", 0.0))
    dst["n_consumers"] += src.get("n_consumers", 0)
    dst["is_source_var"] = dst["is_source_var"] or src.get("is_source_var", False)


def _min_opt(a, b):
    vals = [x for x in (a, b) if x is not None]
    return min(vals) if vals else None


def _max_opt(a, b):
    vals = [x for x in (a, b) if x is not None]
    return max(vals) if vals else None


# ---------------------------------------------------------------------------
# Finalize (mechanistic classification -> consolidated report for Strategy)
# ---------------------------------------------------------------------------
#
# The report is POLICY-NEUTRAL.  It carries measured quantities and a mechanistic
# signal class; it does NOT decide downcast/keep/upgrade or apply an acceptance
# margin — those are the Strategy Agent's job.  In particular
# ``predicted_rel_err_if_float`` (= u_float * cond * amp) is a *measured
# prediction* — the rel-error that would reach an output if this region were
# computed in float — using only the float-format constant and the measured
# forward-cone amplification.  Strategy compares it to ITS margin.

def _range_ok_for_float(reg: dict) -> bool:
    """Measured fact: do all recorded |val| at this region fit float's range?"""
    lo, hi = reg.get("abs_val_min"), reg.get("abs_val_max")
    if lo is not None and lo < FLT_MIN_NORMAL:
        return False
    if hi is not None and hi > FLT_MAX:
        return False
    return True


def _signal_class(reg: dict, cfg: ReducerConfig) -> tuple[str, str]:
    """Mechanistic classification of the error phenomenon (no acceptance policy).

    Returns ``(class, note)`` where the note describes the *measurement*, never a
    remediation.  The residual "stable" class means only that no elevated-error
    mechanism was detected locally — NOT that the region is downcast-safe (that
    depends on the forward-cone amp and the caller's margin, which is Strategy's).
    """
    cond = reg.get("max_cond", 0.0)
    rel = reg.get("max_rel_err", 0.0)
    n = reg.get("n", 0)
    gate_a = reg.get("gate_a_count", 0)

    if gate_a > 0 and cond == 0.0 and n == gate_a:
        return "atan2_saturation", "atan2 saturation cap (2**53) only; no genuine hotspot"
    if cond >= cfg.local_cancel_cond:
        return "local_cancellation", f"local cond {cond:.2e} exceeds 1e15 (|a-b|->0)"
    if cond >= cfg.high_cond:
        return "log_near_root", f"elevated per-op cond {cond:.2e}"
    if rel >= cfg.cascade_rel_err and cond < cfg.cascade_cond_ceiling:
        return "cancellation_cascade", (f"rel_err {rel:.2e} with low per-op cond "
                                        f"{cond:.2e} (accumulated cancellation)")
    return "stable", "no elevated conditioning or accumulated-error signal"


def _classify_region(reg: dict, cfg: ReducerConfig) -> dict:
    cond = reg.get("max_cond", 0.0)
    sens = reg.get("max_sensitivity", 0.0)
    cls, note = _signal_class(reg, cfg)
    hist = LogHist.from_dict(reg.get("rel_err_hist", {}))
    return {
        "signal_class": cls,                     # mechanistic; Strategy branches on it
        "note": note,                            # describes the measurement, not a fix
        "non_localizable": cls == "cancellation_cascade",
        "max_cond": cond,
        "gate_a_count": reg.get("gate_a_count", 0),
        "max_rel_err": reg.get("max_rel_err", 0.0),
        "p50_rel_err": hist.quantile(0.50),
        "p99_rel_err": hist.quantile(0.99),
        "max_amp": reg.get("max_amp", 0.0),
        "max_sensitivity": sens,                 # cond * amp (forward cone)
        "predicted_rel_err_if_float": U_FLOAT * sens,   # measured prediction
        "abs_val_min": reg.get("abs_val_min"),
        "abs_val_max": reg.get("abs_val_max"),
        "value_range_ok_for_float": _range_ok_for_float(reg),
        "n": reg.get("n", 0),
        "ops": reg.get("ops", {}),
        "prov_vars": reg.get("prov_vars", []),
    }


def _classify_variable(var: dict, cfg: ReducerConfig) -> dict:
    sens = var.get("max_sensitivity", 0.0)
    return {
        "max_amp": var.get("max_amp", 0.0),
        "max_sensitivity": sens,
        "predicted_rel_err_if_float": U_FLOAT * sens,   # measured prediction
        "n_consumers": var.get("n_consumers", 0),
        "is_source_var": var.get("is_source_var", False),
        # source values are not journaled by track(); float range guard N/A here.
        "value_range_checked": False,
    }


def finalize_report(merged: dict, cfg: ReducerConfig | None = None) -> dict:
    """Turn a merged report into the consolidated (policy-neutral) report.

    Ranking is by measured severity (``max_rel_err``), NOT by any remediation
    direction — the Strategy Agent applies its acceptance margin and picks the
    downcast/keep/upgrade action per region/variable.
    """
    cfg = cfg or ReducerConfig()
    out_integrals: dict[str, dict] = {}

    for name, idata in merged.get("integrals", {}).items():
        regions = {loc: _classify_region(reg, cfg)
                   for loc, reg in idata.get("regions", {}).items()}
        variables = {vid: _classify_variable(var, cfg)
                     for vid, var in idata.get("variables", {}).items()
                     if var.get("is_source_var")}

        class_counts: dict[str, int] = {}
        for r in regions.values():
            class_counts[r["signal_class"]] = class_counts.get(r["signal_class"], 0) + 1

        out_integrals[name] = {
            "samples": merged.get("samples_seen", {}).get(name, 0),
            "class_counts": class_counts,
            "top_regions_by_rel_err": [
                {"location": loc, **regions[loc]}
                for loc in sorted(regions, key=lambda l: regions[l]["max_rel_err"],
                                  reverse=True)
            ][:10],
            "regions": regions,
            "variables": variables,
        }

    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "stability_report",
        "samples_seen": merged.get("samples_seen", {}),
        "no_id_records": merged.get("no_id_records", 0),
        "integrals": out_integrals,
    }


def report_from_journals(paths: list, cfg: ReducerConfig | None = None) -> dict:
    """Convenience: reduce N shard journals, merge, finalize."""
    cfg = cfg or ReducerConfig()
    return finalize_report(merge_reports([reduce_journal(p, cfg) for p in paths]), cfg)


# ---------------------------------------------------------------------------
# CLI
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
