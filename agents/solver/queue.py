"""Solver candidate queue — pure construction over a scorer manifest (Phase 2e).

The greedy sequential-layering solver (see ``solver.py``) consumes a ranked queue
of candidate patches, each a ``(region_id, rung)`` cell that reached a real
``measured`` verdict in the per-integral fan-out manifest AND changed the region's
output (DISCRIM).  This module is pure data: it reads assembled scorer-manifest
rows and produces a ranked, deduplicated ``Candidate`` list plus the excluded
sets (for the run report).  No git, no build, no LLM.

Locked policy (Reet 2026-07-24, PLAN_overview §Loop semantics):

* Rank cheapest→most-conservative: ``float < ff < dd``.  Never past ``dd``.
* Only ``measured`` cells compete; a region with no measured **DISCRIM** rung
  stays at ``double`` (it never enters the queue).
* ``INERT`` measured cells (``delta_effective == baseline_delta_effective`` — the
  patch produced byte-identical whole-app output) are **excluded**: they carry no
  speedup (the type never changed the computation — the residue the 2c/2d
  ``promotion_no_op`` / ``write_truncation`` gates could not prove statically) and,
  if queued, a no-op "accept" would lock the region at a pointless rung ahead of a
  genuinely cheaper DISCRIM rung.  They are reported, not applied.
* Terminal-failure rungs (``write_truncation``, ``promotion_no_op``,
  ``patch_inapplicable``, and the un-measured ``patcher_failed`` / ``build_failed``
  / ``wire_failed`` cells) never reached ``measured`` and so are absent by
  construction — that is the 2c/2d gates' whole point.
"""

from __future__ import annotations

from dataclasses import dataclass

# Cost rank of a precision rung: float cheapest, dd most conservative.  double is
# the implicit baseline (a region absent from the queue stays double); it is never
# a candidate rung, so it is not ranked here.  ff and double are precision peers
# (see models.LADDER); the solver only ever demotes double→{float,ff} or promotes
# double→dd, so the three candidate rungs are float/ff/dd.
# Phase 2f: ``chain_dd`` is the whole-cascade-chain double-double promotion — a
# correctness-tier candidate keyed by chain_id (not file:line).  It is ranked in its
# own Tier-1 (by predicted lift), not by this cost table, but it is listed here so
# _candidate_from_row admits the rung.
RUNG_RANK: dict[str, int] = {"float": 0, "ff": 1, "dd": 2, "chain_dd": 3}

RUNG_CHAIN_DD = "chain_dd"

_INERT_EPS = 1e-18  # delta equality tolerance (deltas are p100 max rel-errs ~1e-4..1e-13)

STATUS_MEASURED = "measured"


@dataclass(frozen=True)
class Candidate:
    """One ``(region_id, rung)`` competitor for the greedy walk."""

    region_id: str            # "file:line" / "file:start-end" — or chain_id for chain_dd
    rung: str                 # "float" | "ff" | "dd" | "chain_dd"
    kind: str                 # e.g. "double-to-float" (from patcher_metadata)
    intent: str               # "speedup" | "correctness"
    via: str                  # "regional" | "plain" | "chain"
    delta_effective: float
    baseline_delta_effective: float
    intent_id: object = None
    # Phase 2f (chain_dd only): the whole chain's (file, line_start, line_end) regions
    # and the bound-decomposition predicted dd floor lift (ranks the Tier-1 chain queue).
    chain_lines: tuple = ()
    predicted_lift: float | None = None
    # Phase 2f kernel-scope gate (Reet 2026-07-25): the kernel/integral this candidate
    # targets.  ``None`` => whole-app scope (existing behaviour — the candidate is gated
    # against the whole-app p100).  When set, the solver reads the per-kernel baseline
    # and accumulated-min for THIS kernel only, so a candidate targeting kernel K is not
    # gated by a hotspot that lives in a different kernel (e.g. B12's global-min floor
    # pinning a B14 chain).  Chain-dd candidates carry the integral their chain belongs
    # to; single-region candidates carry their containing integral.
    target_kernel: str | None = None

    @property
    def rank(self) -> int:
        return RUNG_RANK.get(self.rung, 99)

    @property
    def is_chain(self) -> bool:
        """True iff this is a chain-scoped dd promotion (Phase 2f)."""
        return self.rung == RUNG_CHAIN_DD

    @property
    def is_discrim(self) -> bool:
        """True iff the patch changed the region's whole-app output.

        DISCRIM = ``delta_effective`` differs from the unpatched
        ``baseline_delta_effective`` measured at the same scope.  INERT (equal)
        means the patch was numerically a no-op.
        """
        de, bd = self.delta_effective, self.baseline_delta_effective
        return de is not None and bd is not None and abs(de - bd) > _INERT_EPS


def _candidate_from_row(row: dict) -> Candidate | None:
    """Build a Candidate from a measured scorer row, or None if unusable."""
    if row.get("status") != STATUS_MEASURED:
        return None
    rung = row.get("rung")
    if rung not in RUNG_RANK:
        return None
    meta = row.get("patcher_metadata") or {}
    is_chain = rung == RUNG_CHAIN_DD
    chain_lines = tuple(tuple(t) for t in (row.get("chain_lines") or []))
    return Candidate(
        region_id=row["region_id"],
        rung=rung,
        kind=meta.get("kind", "double-to-dd" if is_chain else f"double-to-{rung}"),
        intent=meta.get("intent", "correctness" if is_chain else "speedup"),
        via=meta.get("via", "chain" if is_chain else "regional"),
        delta_effective=row.get("delta_effective"),
        baseline_delta_effective=row.get("baseline_delta_effective"),
        intent_id=row.get("intent_id"),
        chain_lines=chain_lines,
        predicted_lift=row.get("predicted_lift"),
        target_kernel=_target_kernel_from_row(row),
    )


def _target_kernel_from_row(row: dict) -> str | None:
    """Kernel this candidate is gated against (Phase 2f kernel-scope).

    Prefers the explicit per-integral tag (Phase A schema v2's ``integral``); falls
    back to a SINGLE-element ``integrals_scope`` (a region attributed to exactly one
    integral).  A multi-integral scope (a shared helper affecting several kernels) or
    an empty scope leaves ``target_kernel=None`` => whole-app gating, the conservative
    default for candidates that legitimately span kernels.
    """
    integral = row.get("integral")
    if integral:
        return str(integral)
    scope = row.get("integrals_scope") or []
    if len(scope) == 1:
        return str(scope[0])
    return None


@dataclass
class QueueBuild:
    """Result of queue construction — the ranked queue plus the excluded sets."""

    queue: list[Candidate]                 # ranked DISCRIM competitors
    inert: list[Candidate]                 # measured but byte-identical (excluded)
    non_measured: list[dict]               # rows that never reached `measured`

    @property
    def regions_in_queue(self) -> set[str]:
        return {c.region_id for c in self.queue}


def _sort_key(c: Candidate) -> tuple:
    # Tiered (Phase 2f).  Tier-1 = chain_dd correctness candidates, ranked by predicted
    # dd floor lift DESCENDING (biggest wins first — highest accept confidence, so a
    # tight wall/budget locks the largest lifts first); chain_id breaks ties.  Tier-2 =
    # single-region candidates by cost rank (float < ff < dd), region_id tiebreak (a
    # documented judgment call — the measurement layer gives no principled intra-rung
    # order; flop-weighting is a v2 refinement).
    if c.is_chain:
        return (0, -(c.predicted_lift or 0.0), c.region_id)
    return (1, c.rank, c.region_id)


def build_queue(rows: list[dict]) -> QueueBuild:
    """Construct the ranked candidate queue from assembled scorer-manifest rows.

    Deduplicates on ``(region_id, rung)`` (fan-out over-generation is already
    collapsed by ``scorer.collapse_min_delta`` upstream, but we guard anyway,
    keeping the most-improving = smallest ``delta_effective``).
    """
    best: dict[tuple[str, str], Candidate] = {}
    inert: list[Candidate] = []
    non_measured: list[dict] = []

    for row in rows:
        cand = _candidate_from_row(row)
        if cand is None:
            non_measured.append(row)
            continue
        if not cand.is_discrim:
            inert.append(cand)
            continue
        key = (cand.region_id, cand.rung)
        prev = best.get(key)
        if prev is None or (cand.delta_effective is not None
                            and prev.delta_effective is not None
                            and cand.delta_effective < prev.delta_effective):
            best[key] = cand

    queue = sorted(best.values(), key=_sort_key)
    return QueueBuild(queue=queue, inert=inert, non_measured=non_measured)


def load_manifest_rows(manifest_path: str) -> list[dict]:
    """Read an assembled ``manifest_scorer_<I>.jsonl`` into row dicts."""
    import json
    rows: list[dict] = []
    with open(manifest_path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
