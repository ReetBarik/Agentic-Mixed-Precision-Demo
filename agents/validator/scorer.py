"""Scorer — the measurement half of the reframed Validator (Phase 2b).

The pre-2b Validator collapsed the whole application's answer to a single
min-precise-digits number, compared it to a pure-double baseline, and stamped one
pass/fail verdict *per intent*.  A genuine local improvement — one region moved to
``dd`` that drops its integral's relative error to ~1e-14 — was buried in the
app-level total and rejected as ``insufficient_fix`` because the floor was
calibrated for the whole app's error budget.  The measurement level (the whole
app) was correct; the *decision* level (per intent) was wrong.

This module splits the two.  It emits **no verdicts** — it produces a manifest of
``(region_id, rung) -> delta`` cells, where ``delta`` is the app-level error
metric *attributable to that region at that rung*, measured against a
per-app-configurable ground-truth reference.  Downstream consumers (today's
accept/reject loop; a future collapsed solver) apply their own decision logic
against the manifest.  The scorer is a pure reduction over coeff arrays the
Validator already builds — it spawns no extra app runs, so it adds no wall-clock.

Design decisions (locked in the Phase-2b handoff)
-------------------------------------------------
* **Primary key** ``region_id`` — canonicalized from ``(file, line_start,
  line_end)`` (:func:`canonical_region_id`).  Stable across Patcher runs and
  fan-out over-generation: the characterization report keys regions by
  ``file:line`` and the Patcher never mutates that identity, so no AST-node hash is
  needed.  Fan-out variants share the region_id of the region they promote; a read
  collapses over-generation via "min delta per ``(region_id, rung)``"
  (:func:`collapse_min_delta`).
* **Metric** = max relative error (p100) over the input battery, reported split:
  ``delta_adversarial`` (curated adversarial samples) and ``delta_random`` (random
  top-up).  Effective delta for consumers = ``max(delta_adversarial,
  delta_random)`` (:func:`effective_delta`).  ``delta_adversarial`` is ``None``
  when the adversarial slice is empty (the 2b stub).
* **Output reduction** app-output -> scalar is per-app config.  qcdloop:
  ``max`` across the six Laurent components (eps^-2 / eps^-1 / eps^0, real+imag)
  per sample.  Default for unspecified apps: identity (scalar output).
* **Baseline reference** per-app config (``baseline_spec``): qcdloop instantiates
  the whole app at ``dd`` (~31 digits) as ground truth; default is ``double``.  The
  manifest records ``baseline_id`` (a hash of the spec + app source) per cell.
* **Baseline policy** = working-tree (deltas are relative to the current app
  state, not pristine), so cells are **not** cross-iteration-comparable — each cell
  records ``iteration_id``.  Run outside an iterative loop (the 2b B1
  demonstration) that degrades cleanly to a pristine baseline at ``iteration_id=0``.

Documented limitation: a solo max is a *lower bound* on the joint max — the same
adversarial sample lighting up two regions can produce a joint delta far above
either solo value.  Joint measurement is a future concern; not addressed here.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from agents.validator.coeffs import N_COMPONENTS
from agents.validator.precise_digits import effectively_zero

MANIFEST_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Status enum — distinguishes "chose not to measure" from the codegen/build/wire
# gaps a float-hardening pass reads as a work queue.
# ---------------------------------------------------------------------------
STATUS_MEASURED = "measured"            # generated, built, wired, ran, scored
STATUS_NOT_A_CANDIDATE = "not_a_candidate"  # Strategy declined to emit an intent
STATUS_PATCHER_FAILED = "patcher_failed"    # codegen failed (e.g. llm_gen_failed)
STATUS_BUILD_FAILED = "build_failed"        # generated but did not compile
STATUS_WIRE_FAILED = "wire_failed"          # generated + built but not referenced

STATUSES = frozenset({
    STATUS_MEASURED, STATUS_NOT_A_CANDIDATE, STATUS_PATCHER_FAILED,
    STATUS_BUILD_FAILED, STATUS_WIRE_FAILED,
})

# Map a Strategy/Patcher ``patcher_status`` (and fan-out failure mode) onto the
# manifest status for a cell that never reached the scorer.  Anything unrecognized
# folds to ``patcher_failed`` (a codegen gap is the conservative default).
_PATCHER_STATUS_TO_CELL_STATUS = {
    "ok": STATUS_MEASURED,
    "llm_gen_failed": STATUS_PATCHER_FAILED,
    "empty_candidate": STATUS_PATCHER_FAILED,
    "patch_inapplicable": STATUS_PATCHER_FAILED,
    "timeout": STATUS_PATCHER_FAILED,
    "build_failed": STATUS_BUILD_FAILED,
    "call_graph_build_failed": STATUS_BUILD_FAILED,
    "variant_name_collision": STATUS_WIRE_FAILED,
    "rename_cascade_incomplete": STATUS_WIRE_FAILED,
    "silent_bypass": STATUS_WIRE_FAILED,
}


def cell_status_for(patcher_status: str | None, failure_mode: str | None = None) -> str:
    """Manifest cell status from a Patcher outcome (fan-out ``failure_mode`` first).

    ``failure_mode`` is the finest classification the per-iter error excerpt could
    recover (see ``per_integral_orchestrator.manifest``); it wins over the coarse
    ``patcher_status`` when it maps to a distinct cell status.  Neither present ->
    ``patcher_failed``.
    """
    for key in (failure_mode, patcher_status):
        if key and key in _PATCHER_STATUS_TO_CELL_STATUS:
            return _PATCHER_STATUS_TO_CELL_STATUS[key]
    return STATUS_PATCHER_FAILED


# ---------------------------------------------------------------------------
# Baseline reference + battery specs (per-app config, hashed into the cache key)
# ---------------------------------------------------------------------------

def default_baseline_spec() -> dict:
    """Default ground-truth reference: instantiate the whole app at ``double``."""
    return {"kind": "instantiate_at", "type": "double"}


def qcdloop_baseline_spec() -> dict:
    """qcdloop ground-truth reference: instantiate the whole app at ``dd`` (~31 digits)."""
    return {"kind": "instantiate_at", "type": "dd"}


def baseline_id(baseline_spec: dict, *app_source_hashes: str) -> str:
    """Stable id for a reference: ``hash(baseline_spec + app source hash(es))``.

    Folding the app source hash(es) in makes the id — and any cache keyed on it —
    change exactly when the reference would: a different ``baseline_spec`` (e.g.
    ``dd`` -> ``mpfr``) or a different app source (the vanilla working tree and/or
    the DD oracle tree).  Truncated to 16 hex chars (collision-safe for a run).
    """
    h = hashlib.sha256()
    h.update(json.dumps(baseline_spec, sort_keys=True).encode("utf-8"))
    for src in app_source_hashes:
        h.update(b"\0")
        h.update((src or "").encode("utf-8"))
    return h.hexdigest()[:16]


def make_battery_spec(random_slice: list | None = None,
                      adversarial_slice: list | None = None) -> dict:
    """Battery spec shaped for future work, stub-populated for 2b.

    ``random_slice`` / ``adversarial_slice`` are lists of ``InputSpec`` dicts (the
    concrete shape is a follow-up — for 2b the random slice is a single
    ``{"kind": "random_range", "seed": .., "count": ..}`` descriptor and the
    adversarial slice is empty).  ``version`` is a hash of both slices, folded into
    the manifest cache key so a battery change invalidates prior cells.
    """
    random_slice = list(random_slice or [])
    adversarial_slice = list(adversarial_slice or [])
    version = _battery_version(random_slice, adversarial_slice)
    return {
        "adversarial": adversarial_slice,
        "random": random_slice,
        "version": version,
    }


def _battery_version(random_slice: list, adversarial_slice: list) -> str:
    h = hashlib.sha256()
    h.update(json.dumps(random_slice, sort_keys=True).encode("utf-8"))
    h.update(b"\0")
    h.update(json.dumps(adversarial_slice, sort_keys=True).encode("utf-8"))
    return h.hexdigest()[:16]


def snapshot_battery_spec(snapshot: dict, adversarial_offsets: list | None = None) -> dict:
    """Build the 2b battery spec from a Validator ``snapshot`` (+ optional tail).

    The random slice is the ``[0, sample_count)`` stream described by the snapshot;
    the adversarial slice is the sparse tail offsets (empty for report_5k, which
    carries no ``tail_samples``).
    """
    seed = int(snapshot.get("seed", 12345))
    count = int(snapshot.get("sample_count", 0))
    random_slice = [{"kind": "random_range", "seed": seed, "count": count}]
    adv = [{"kind": "offset", "offset": int(o)}
           for o in sorted({int(x) for x in (adversarial_offsets or [])})]
    return make_battery_spec(random_slice, adv)


# ---------------------------------------------------------------------------
# Region id + rung canonicalization
# ---------------------------------------------------------------------------

def canonical_region_id(file: str, line_start: int, line_end: int | None = None) -> str:
    """Canonical, run-stable region id from the characterization span.

    Single-line -> ``"file:line"`` (matches the report's region keys); multi-line ->
    ``"file:start-end"``.  This is the primary key: derived purely from the
    characterization output, it is invariant to Patcher runs and to fan-out
    over-generation (which change the *variant*, never the region).
    """
    if line_end is None or line_end == line_start:
        return f"{file}:{line_start}"
    return f"{file}:{line_start}-{line_end}"


def rung_from_kind(kind: str) -> str:
    """The destination precision rung of an intent ``kind``.

    Transition kinds are ``"<src>-to-<dst>"`` -> ``dst``.  Reformulate kinds
    (``reformulate-kahan`` / ``reformulate-identity``) have no precision
    destination; the kind is returned verbatim as the rung label.
    """
    if "-to-" in kind:
        return kind.rsplit("-to-", 1)[1]
    return kind


# ---------------------------------------------------------------------------
# Output reduction (app-output -> scalar rel-err), per-app config
# ---------------------------------------------------------------------------

def _rel_err(cand_hi: float, cand_lo: float, ref_hi: float, ref_lo: float,
             ref_scale: float) -> float:
    """Relative error of one ``(hi,lo)`` component vs the DD reference component.

    Mirrors :func:`agents.validator.precise_digits.precise_digits_fast`'s error
    model (exact leading cancellation on the two-word difference), but returns the
    raw relative error rather than digits.  An analytic zero (DD reference below the
    per-sample noise floor for ``ref_scale``) carries no correct digits to lose, so
    the relative metric is undefined on it -> contributes ``0.0``.  A genuine zero
    reference with nonzero error (not caught by the band) is clamped to ``1.0``
    (total loss), the same ceiling ``precise_digits`` uses for ``rel >= 1``.
    """
    err = abs((cand_hi - ref_hi) + (cand_lo - ref_lo))
    true = abs(ref_hi + ref_lo)
    if err == 0.0:
        return 0.0
    if effectively_zero(true, ref_scale):
        return 0.0
    if true == 0.0:
        return 1.0
    rel = err / true
    return 1.0 if rel > 1.0 else rel


def _sample_rel_err_laurent(cand_comps: list, ref_comps: list) -> float:
    """Max relative error across a sample's six Laurent components (qcdloop reduction).

    ``cand_comps`` / ``ref_comps`` are lists of ``(hi, lo)`` pairs of length
    :data:`N_COMPONENTS`.  ``ref_scale`` is the sample's characteristic magnitude
    (max ``|DD component|``), matching the Validator's ``_score`` band.
    """
    ref_scale = 0.0
    for (rh, rl) in ref_comps:
        m = abs(rh + rl)
        if m > ref_scale:
            ref_scale = m
    worst = 0.0
    for c in range(N_COMPONENTS):
        ch, cl = cand_comps[c]
        rh, rl = ref_comps[c]
        e = _rel_err(ch, cl, rh, rl, ref_scale)
        if e > worst:
            worst = e
    return worst


# ---------------------------------------------------------------------------
# Delta reduction over coeff arrays / tail-offset dicts
# ---------------------------------------------------------------------------

def _integrals_in_scope(integrals_scope: Iterable[str] | None,
                        available: Iterable[str]) -> list[str]:
    """Region's integrals intersected with what the run produced.

    ``None``/empty scope means the default whole-app reduction (all available
    integrals) — the identity fallback for apps that do not attribute a region to
    a subset of outputs.
    """
    avail = set(available)
    if not integrals_scope:
        return sorted(avail)
    return sorted(set(integrals_scope) & avail)


def delta_over_arrays(candidate_coeffs: dict, dd_ref_coeffs: dict,
                      integrals_scope: Iterable[str] | None) -> float | None:
    """p100 (max) sample rel-err of candidate vs DD over the integrals in scope.

    ``candidate_coeffs`` / ``dd_ref_coeffs`` are the Validator's
    ``{integral: (hi_array, lo_array)}`` (flat, index = ``sample*6 + component``).
    Returns ``None`` if no in-scope integral produced data.
    """
    scope = _integrals_in_scope(integrals_scope, dd_ref_coeffs.keys())
    if not scope:
        return None
    worst = None
    for integ in scope:
        if integ not in candidate_coeffs or integ not in dd_ref_coeffs:
            continue
        c_hi, c_lo = candidate_coeffs[integ]
        r_hi, r_lo = dd_ref_coeffs[integ]
        n = min(len(c_hi), len(r_hi)) // N_COMPONENTS
        for s in range(n):
            base = s * N_COMPONENTS
            cand_comps = [(c_hi[base + c], c_lo[base + c]) for c in range(N_COMPONENTS)]
            ref_comps = [(r_hi[base + c], r_lo[base + c]) for c in range(N_COMPONENTS)]
            e = _sample_rel_err_laurent(cand_comps, ref_comps)
            if worst is None or e > worst:
                worst = e
    return worst


def delta_over_tail(candidate_tail: dict | None, dd_ref_tail: dict | None,
                    integrals_scope: Iterable[str] | None) -> float | None:
    """p100 rel-err over the sparse adversarial tail (``{integral:{offset:[(hi,lo)x6]}}``).

    Returns ``None`` when either side is empty (the 2b stub) or no in-scope integral
    has tail coeffs — the manifest then records ``delta_adversarial = null``.
    """
    if not candidate_tail or not dd_ref_tail:
        return None
    scope = _integrals_in_scope(integrals_scope, dd_ref_tail.keys())
    if not scope:
        return None
    worst = None
    for integ in scope:
        c_by_off = candidate_tail.get(integ, {})
        d_by_off = dd_ref_tail.get(integ, {})
        for off, ref_comps in d_by_off.items():
            cand_comps = c_by_off.get(off)
            if cand_comps is None:
                continue
            e = _sample_rel_err_laurent(cand_comps, ref_comps)
            if worst is None or e > worst:
                worst = e
    return worst


def effective_delta(delta_adversarial: float | None,
                    delta_random: float | None) -> float | None:
    """Consumer-facing delta = ``max(delta_adversarial, delta_random)`` (nulls dropped)."""
    vals = [d for d in (delta_adversarial, delta_random) if d is not None]
    return max(vals) if vals else None


# ---------------------------------------------------------------------------
# Manifest row
# ---------------------------------------------------------------------------

@dataclass
class ManifestRow:
    """One ``(region_id, rung)`` manifest cell.

    Minimum-viable schema from the Phase-2b handoff (fields may be added, never
    removed).  ``delta_*`` are ``None`` for a non-``measured`` status or an empty
    battery slice.  ``integrals_scope`` records the output attribution used for the
    reduction (a debugging attribute, not part of the key).
    """

    region_id: str                       # primary key
    rung: str
    iteration_id: int
    status: str
    delta_adversarial: float | None = None
    delta_random: float | None = None
    delta_effective: float | None = None
    baseline_id: str | None = None
    battery_version: str | None = None
    intent_id: str | int | None = None   # attribute (traceability)
    integrals_scope: list = field(default_factory=list)   # attribute
    patcher_metadata: dict = field(default_factory=dict)   # attribute
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.region_id:
            raise ValueError("ManifestRow requires a non-empty region_id")
        if self.status not in STATUSES:
            raise ValueError(
                f"unknown status {self.status!r}; expected one of {sorted(STATUSES)}")
        if self.delta_effective is None:
            self.delta_effective = effective_delta(
                self.delta_adversarial, self.delta_random)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ManifestRow":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        # Preserve unknown (future) fields round-trip-safe by attaching them to
        # patcher_metadata under a reserved key rather than dropping them.
        extra = {k: v for k, v in d.items() if k not in known}
        row = cls(**{k: v for k, v in d.items() if k in known})
        if extra:
            row.patcher_metadata = {**row.patcher_metadata, "_extra": extra}
        return row


def validate_row(row: dict) -> None:
    """Assert a raw dict is a schema-valid manifest row (raises ``ValueError``)."""
    if not isinstance(row, dict):
        raise ValueError(f"manifest row must be a dict, got {type(row).__name__}")
    for req in ("region_id", "rung", "iteration_id", "status"):
        if req not in row:
            raise ValueError(f"manifest row missing required field {req!r}")
    if not row["region_id"]:
        raise ValueError("manifest row has empty region_id")
    if row["status"] not in STATUSES:
        raise ValueError(f"manifest row has unknown status {row['status']!r}")


# ---------------------------------------------------------------------------
# Cell construction (the scorer's core) + JSONL manifest I/O
# ---------------------------------------------------------------------------

def score_cell(*, region_id: str, rung: str, iteration_id: int,
               candidate_coeffs: dict, dd_ref_coeffs: dict,
               integrals_scope: Iterable[str] | None,
               baseline_id: str | None = None, battery_version: str | None = None,
               candidate_tail: dict | None = None, dd_ref_tail: dict | None = None,
               intent_id: str | int | None = None,
               patcher_metadata: dict | None = None) -> ManifestRow:
    """Reduce already-computed coeff arrays into a ``measured`` manifest cell.

    A pure function over the Validator's candidate + DD-reference coeff arrays: it
    spawns no build/run, so folding it into a ``validate()`` call costs no extra
    wall-clock.  ``delta_random`` is the p100 rel-err over the random battery;
    ``delta_adversarial`` is the p100 over the tail (``None`` when the tail is
    empty — the 2b stub).
    """
    d_rand = delta_over_arrays(candidate_coeffs, dd_ref_coeffs, integrals_scope)
    d_adv = delta_over_tail(candidate_tail, dd_ref_tail, integrals_scope)
    return ManifestRow(
        region_id=region_id, rung=rung, iteration_id=int(iteration_id),
        status=STATUS_MEASURED,
        delta_adversarial=d_adv, delta_random=d_rand,
        baseline_id=baseline_id, battery_version=battery_version,
        intent_id=intent_id,
        integrals_scope=sorted(set(integrals_scope)) if integrals_scope else [],
        patcher_metadata=dict(patcher_metadata or {}),
    )


def append_row(manifest_path: str | Path, row: ManifestRow | dict) -> None:
    """Append one row to a JSONL manifest (creating parent dirs on first write)."""
    d = row.to_dict() if isinstance(row, ManifestRow) else dict(row)
    validate_row(d)
    p = Path(manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as fh:
        fh.write(json.dumps(d) + "\n")


def write_rows(manifest_path: str | Path, rows: Iterable[ManifestRow | dict]) -> None:
    """Write (overwrite) a JSONL manifest from ``rows``."""
    p = Path(manifest_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as fh:
        for row in rows:
            d = row.to_dict() if isinstance(row, ManifestRow) else dict(row)
            validate_row(d)
            fh.write(json.dumps(d) + "\n")


def read_rows(manifest_path: str | Path) -> list[dict]:
    """Read a JSONL manifest into raw dict rows (empty list if absent)."""
    p = Path(manifest_path)
    if not p.is_file():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if line:
            d = json.loads(line)
            validate_row(d)
            out.append(d)
    return out


def collapse_min_delta(rows: Iterable[dict]) -> dict:
    """Collapse fan-out over-generation: min ``delta_effective`` per ``(region_id, rung)``.

    Returns ``{(region_id, rung): row}`` keeping, among ``measured`` cells sharing a
    key, the one with the smallest (best) ``delta_effective``.  Non-``measured``
    cells are retained only when no ``measured`` cell exists for the key (so a
    ``patcher_failed`` speedup rung still surfaces, but a real measurement always
    wins over a failure for the same key).
    """
    best: dict[tuple[str, str], dict] = {}
    for row in rows:
        key = (row["region_id"], row["rung"])
        cur = best.get(key)
        if cur is None:
            best[key] = row
            continue
        cur_measured = cur["status"] == STATUS_MEASURED
        row_measured = row["status"] == STATUS_MEASURED
        if row_measured and not cur_measured:
            best[key] = row
        elif row_measured and cur_measured:
            if _delta_key(row) < _delta_key(cur):
                best[key] = row
        # else: keep cur (measured beats non-measured; non-measured ties keep first)
    return best


def _delta_key(row: dict) -> float:
    d = row.get("delta_effective")
    return d if d is not None else float("inf")


# ---------------------------------------------------------------------------
# Non-measured cells from the Strategy iteration log (the codegen/build work queue)
# ---------------------------------------------------------------------------

def rows_from_iteration_log(iters: Iterable[dict], *, iteration_id: int = 0,
                            baseline_id: str | None = None,
                            battery_version: str | None = None,
                            measured_keys: set | None = None,
                            integral_scope_for: dict | None = None) -> list[ManifestRow]:
    """Non-measured cells (codegen/build/wire failures) from the iteration trail.

    The scorer writes ``measured`` cells inline during ``validate()``.  Intents
    that never reached the Validator (``llm_gen_failed``, ``build_failed``, a
    fan-out wire failure) leave no scorer cell — this recovers them from the
    Strategy iteration log so the manifest is a complete work queue: a
    ``patcher_failed`` / ``build_failed`` / ``wire_failed`` cell per un-measured
    intent, keyed by the same ``(region_id, rung)``.

    ``measured_keys`` are the ``(region_id, rung)`` pairs already covered by a
    scorer cell (skipped here to avoid duplication).  ``integral_scope_for`` maps a
    ``region_id`` to its integrals for the ``integrals_scope`` attribute.
    """
    measured_keys = measured_keys or set()
    integral_scope_for = integral_scope_for or {}
    seen: set[tuple[str, str]] = set()
    out: list[ManifestRow] = []
    for rec in iters:
        if bool(rec.get("accepted")):
            continue  # accepted -> a measured cell already exists
        tgt = rec.get("target", {}) or {}
        file = tgt.get("file")
        if not file:
            continue
        region_id = canonical_region_id(
            file, tgt.get("line_start"), tgt.get("line_end", tgt.get("line_start")))
        rung = rung_from_kind(rec.get("kind", ""))
        patcher_status = rec.get("patcher_status")
        # A genuine dd-ceiling retain (patcher ok, validator reject) WAS measured;
        # its scorer cell exists, so skip it here.
        if patcher_status == "ok":
            continue
        key = (region_id, rung)
        if key in measured_keys or key in seen:
            continue
        seen.add(key)
        status = cell_status_for(patcher_status, rec.get("failure_mode"))
        out.append(ManifestRow(
            region_id=region_id, rung=rung, iteration_id=int(iteration_id),
            status=status,
            baseline_id=baseline_id, battery_version=battery_version,
            intent_id=rec.get("iter_id"),
            integrals_scope=integral_scope_for.get(region_id, []),
            patcher_metadata={
                "patcher_status": patcher_status,
                "verdict_reason": rec.get("verdict_reason"),
                "failure_mode": rec.get("failure_mode"),
                "kind": rec.get("kind"),
                "intent": rec.get("intent"),
                "phase": rec.get("phase"),
            },
        ))
    return out


def assemble_manifest(scored_manifest_path: str | Path,
                      iteration_log_path: str | Path | None,
                      out_path: str | Path, *,
                      iteration_id: int = 0,
                      baseline_id: str | None = None,
                      battery_version: str | None = None) -> list[dict]:
    """Merge scorer ``measured`` cells with the un-measured cells into one manifest.

    The scorer writes ``measured`` cells inline during the run (to
    ``scored_manifest_path``).  This folds in the codegen/build/wire failures from
    the Strategy iteration log (:func:`rows_from_iteration_log`) so the final
    ``out_path`` is a complete ``(region_id, rung)`` record for the pass — the
    float-hardening backlog reads the ``patcher_failed`` cells as its work queue.

    Returns the merged rows (also written to ``out_path`` as JSONL).  Idempotent:
    re-runnable from the same inputs.  ``baseline_id`` / ``battery_version`` default
    to whatever the scorer stamped on the measured cells (so the failure cells share
    the same cache key), falling back to the passed values.
    """
    measured = read_rows(scored_manifest_path)
    measured_keys = {(r["region_id"], r["rung"]) for r in measured}
    scope_for = {r["region_id"]: r.get("integrals_scope", []) for r in measured}
    if measured:
        baseline_id = baseline_id or measured[0].get("baseline_id")
        battery_version = battery_version or measured[0].get("battery_version")

    iters = []
    if iteration_log_path and Path(iteration_log_path).is_file():
        for line in Path(iteration_log_path).read_text().splitlines():
            line = line.strip()
            if line:
                iters.append(json.loads(line))

    failed = rows_from_iteration_log(
        iters, iteration_id=iteration_id, baseline_id=baseline_id,
        battery_version=battery_version, measured_keys=measured_keys,
        integral_scope_for=scope_for)

    merged = measured + [r.to_dict() for r in failed]
    write_rows(out_path, merged)
    return merged
