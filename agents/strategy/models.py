"""Strategy data model — region records, remediation intents, precision ladder.

The region-record shape ``(file, line_start, line_end, variables)`` is the
contract shared verbatim across Strategy output, the Patcher intent (P1), and
``ff_integrator``/``dd_integrator`` (Q1).  Nothing here talks to git or the LLM;
this module is pure data + the ladder algebra the retry walk depends on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Precision ladder — COST-ordered, not strictly precision-ordered (P3).
#
#   float (~7)  →  ff (~14)  →  double (~15-16)  →  qf (~29)  →  dd (~30-31)
#
# ff and double are precision peers (within one digit) that trade in either
# direction; the ladder orders them by *cost*, so a double→ff move is a legit
# speedup demotion.  The retry walk single-steps this list by index.
#
# qf (quad-float, 4xFP32, ~28.9 digits) sits between double and dd.  On the
# fp32-heavy GPU silicon this pipeline targets, cost order and accuracy order
# COINCIDE for qf — four FP32 words are cheaper than a double-double while
# resolving only ~2 digits less — so it needs no split cost/accuracy ordering and
# slots into the single cost-ordered ladder directly.  That placement is what
# makes the correctness walk a *cheapest-sufficient* search: try qf, and only pay
# for dd if qf does not clear.
#
# RANGE IS NOT MONOTONE ALONG THIS LADDER.  Cost and precision both increase
# left to right, but exponent range does not: float/ff/qf are FP32-ranged
# (~3.4e38) while double/dd are FP64-ranged (~1.8e308).  So qf is WIDER than
# double in significand yet NARROWER in range — the one place the ladder's order
# does not carry range with it.  See FP32_FAMILY below and the guard that uses it.
# ---------------------------------------------------------------------------

LADDER: tuple[str, ...] = ("float", "ff", "double", "qf", "dd")

PRECISIONS = frozenset(LADDER)

# ---------------------------------------------------------------------------
# The fp32 family — rungs built out of FP32 words, which therefore inherit
# FP32's EXPONENT RANGE (|x| in [~1.18e-38, ~3.40e38]) no matter how many words
# they stack:
#
#   float  1 x FP32   ~7 digits
#   ff     2 x FP32   ~14 digits
#   qf     4 x FP32   ~29 digits
#
# Stacking words widens the SIGNIFICAND, never the exponent.  So a value that
# overflows float overflows qf too, even though qf is nearly twice as precise as
# double.  Every other rung (double, dd) is FP64-ranged and at least as wide as
# the double baseline, so range is only ever a question for this set.
#
# This is why the range guard is keyed on the family and not on ``float``: the
# original WI1 guard dropped only the float rung and fell back to ff, which has
# the identical ceiling — the fallback was never safe.  See
# characterization.value_range_ok_for_float (the measured signal) and
# walk.RetryWalk(fp32_range_ok=...) (the consumer).
# ---------------------------------------------------------------------------

FP32_FAMILY: frozenset[str] = frozenset({"float", "ff", "qf"})

# ---------------------------------------------------------------------------
# Which rungs the REGION path can actually realize.
#
# The ladder is shared by two remediation mechanisms with different reach:
#
#   * the whole-TU precision flip (agents.patcher.tu_emit PROFILES) — has a qf
#     profile, so it can build a whole TU at qf; and
#   * the region path (agents.patcher.dispatch -> the per-precision *_integrator
#     packages) — has float / ff / dd integrators and NO qf integrator.
#
# Adding qf to LADDER makes it a candidate up-rung for BOTH walks, but a
# ``double-to-qf`` region intent has nothing to service it.  Restricting the
# region walk here keeps the ladder single and honest rather than forking it:
# qf is on the ladder because it is a real precision, and this set records that
# one mechanism cannot reach it yet.  Drop qf from this exclusion once a
# qf_integrator exists.
# ---------------------------------------------------------------------------

REGION_REALIZABLE: frozenset[str] = frozenset({"float", "ff", "double", "dd"})


def _index(precision: str) -> int:
    try:
        return LADDER.index(precision)
    except ValueError:  # pragma: no cover - guarded by callers
        raise ValueError(f"unknown precision {precision!r}; expected one of {LADDER}")


def next_up(precision: str) -> str | None:
    """The next more-expensive (toward-dd) rung, or None if already at dd."""
    i = _index(precision)
    return LADDER[i + 1] if i + 1 < len(LADDER) else None


def next_down(precision: str) -> str | None:
    """The next cheaper (toward-float) rung, or None if already at float."""
    i = _index(precision)
    return LADDER[i - 1] if i - 1 >= 0 else None


# ---------------------------------------------------------------------------
# Remediation-kind vocabulary (P1 + P3).
#
# 13 transition kinds + 2 reformulate kinds = 15 total.  `float-to-ff` is the
# single-step up-rung the P3 table omits (see HANDOFF.md); it is required for a
# cost-ladder walk that starts from a float region.  The three "skip"
# transitions (float-to-double, double-to-float, ff-to-dd) are valid Patcher
# kinds but are NOT emitted by the current single-step walk.
#
# The qf rung adds four: double-to-qf / qf-to-dd going up and dd-to-qf /
# qf-to-double coming down.  `double-to-dd` and `dd-to-double` are RETAINED even
# though qf now sits between them — inserting a ladder rung must not silently
# retire the direct double<->dd transition, which the correctness walk still
# emits whenever the qf rung is skipped (range guard) or does not clear.
# ---------------------------------------------------------------------------

TRANSITION_KINDS: frozenset[str] = frozenset({
    # single-step up (correctness)
    "float-to-ff", "ff-to-double", "double-to-qf", "qf-to-dd",
    # single-step down (speedup)
    "qf-to-double", "dd-to-qf", "double-to-ff", "ff-to-float",
    # double<->dd: no longer single-step now that qf is between them, but still
    # emitted (and still the dominant correctness rung) — see note above.
    "double-to-dd", "dd-to-double",
    # skip transitions — in vocabulary, not emitted by single-step walk
    "float-to-double", "double-to-float", "ff-to-dd",
})

REFORMULATE_KINDS: frozenset[str] = frozenset({
    "reformulate-kahan", "reformulate-identity",
})

ALL_KINDS: frozenset[str] = TRANSITION_KINDS | REFORMULATE_KINDS

# Starter identity catalog for reformulate-identity (P3b). Strategy picks the
# identity from signal-class context; the retry walk consumes this order.
IDENTITY_CATALOG: tuple[str, ...] = ("log1p", "expm1", "hypot", "1-cos->2sin2")

# Intent flavor (P3 addendum): correctness vs speedup, a peer of `kind`.
INTENT_CORRECTNESS = "correctness"
INTENT_SPEEDUP = "speedup"

# Signal classes emitted by the characterizer (report `signal_class`).
SIGNAL_LOCAL_CANCELLATION = "local_cancellation"
SIGNAL_CANCELLATION_CASCADE = "cancellation_cascade"
SIGNAL_LOG_NEAR_ROOT = "log_near_root"
SIGNAL_STABLE = "stable"


def transition_kind(src: str, dst: str) -> str:
    """The ``<src>-to-<dst>`` kind for a precision transition.

    Raises if ``src == dst`` or the pair is not a recognized transition kind.
    """
    if src == dst:
        raise ValueError(f"no-op transition {src!r} -> {dst!r}")
    kind = f"{src}-to-{dst}"
    if kind not in TRANSITION_KINDS:
        raise ValueError(f"unsupported transition {kind!r}")
    return kind


# ---------------------------------------------------------------------------
# Region record + remediation intent
# ---------------------------------------------------------------------------

@dataclass
class RegionTarget:
    """The ``(file, line_start, line_end, variables)`` contract (Q1 / P1).

    Single-line regions have ``line_start == line_end``.
    """

    file: str
    line_start: int
    line_end: int
    variables: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "file": self.file,
            "line_start": self.line_start,
            "line_end": self.line_end,
            "variables": list(self.variables),
        }

    @property
    def key(self) -> tuple[str, int, int]:
        """Join key for merging the precision + rewrite stacks on one region (Q2)."""
        return (self.file, self.line_start, self.line_end)

    @property
    def location(self) -> str:
        """Human ``file:line`` (matches characterization region keys)."""
        if self.line_start == self.line_end:
            return f"{self.file}:{self.line_start}"
        return f"{self.file}:{self.line_start}-{self.line_end}"


# How the Patcher should realize a ``-to-float`` demotion (P3 Wave-2 amendment).
#   "plain"    — plain-type-edit / git-revert path (a bare ``double`` token exists
#                to rewrite; the historical rung, kept for non-templated regions).
#   "regional" — the LLM/regional float integrator (a template-typed region has no
#                bare ``double`` token, so float is only reachable by generating a
#                ``float``-specialized shim, exactly as ff/dd are generated).
# For every non-``-to-float`` kind ``via`` is inert (those kinds have a single
# dispatch path); it defaults to "plain".
VIA_PLAIN = "plain"
VIA_REGIONAL = "regional"
# Phase 2f: a chain-scoped double-double promotion (a whole cancellation-cascade
# chain widened together).  The intent carries ``chain_lines`` and routes to the
# Patcher chain path (agents.patcher.chain_promote) via the chain integrator.
VIA_CHAIN = "chain"


@dataclass
class RemediationIntent:
    """What Strategy emits to Patcher (P1 + P3 amendment).

    ``identity`` is populated only for ``kind == "reformulate-identity"``.
    ``via`` selects how a ``-to-float`` demotion is realized (plain edit vs the
    regional float integrator) — see :data:`VIA_PLAIN` / :data:`VIA_REGIONAL`.
    """

    target: RegionTarget
    kind: str
    intent: str                    # correctness | speedup
    current_precision: str
    rationale_id: str
    identity: str | None = None
    via: str = VIA_PLAIN
    # Phase 2f: for a chain-scoped promotion (via == VIA_CHAIN), the whole chain's
    # (file, line_start, line_end) regions.  Empty for every non-chain intent.
    chain_lines: list = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.kind not in ALL_KINDS:
            raise ValueError(f"unknown kind {self.kind!r}")
        if self.intent not in (INTENT_CORRECTNESS, INTENT_SPEEDUP):
            raise ValueError(f"unknown intent {self.intent!r}")
        if self.kind == "reformulate-identity" and not self.identity:
            raise ValueError("reformulate-identity requires an `identity`")
        if self.kind != "reformulate-identity" and self.identity is not None:
            raise ValueError(f"identity only valid for reformulate-identity, got {self.kind!r}")
        if self.via not in (VIA_PLAIN, VIA_REGIONAL, VIA_CHAIN):
            raise ValueError(f"unknown via {self.via!r}")
        if self.via == VIA_CHAIN and not self.chain_lines:
            raise ValueError("via=chain requires chain_lines")

    def to_patcher(self) -> dict:
        """The wire form handed to the Patcher callable (P1 shape)."""
        payload = {
            "target": self.target.as_dict(),
            "kind": self.kind,
            "intent": self.intent,
            "current_precision": self.current_precision,
            "rationale_id": self.rationale_id,
            "via": self.via,
        }
        if self.identity is not None:
            payload["identity"] = self.identity
        if self.chain_lines:
            payload["chain_lines"] = [list(t) for t in self.chain_lines]
        return payload
