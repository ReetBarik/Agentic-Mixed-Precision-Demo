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
#   float (~7)  →  ff (~14)  →  double (~15-16)  →  dd (~30-31)
#
# ff and double are precision peers (within one digit) that trade in either
# direction; the ladder orders them by *cost*, so a double→ff move is a legit
# speedup demotion.  The retry walk single-steps this list by index.
# ---------------------------------------------------------------------------

LADDER: tuple[str, ...] = ("float", "ff", "double", "dd")

PRECISIONS = frozenset(LADDER)


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
# 9 transition kinds + 2 reformulate kinds = 11 total.  `float-to-ff` is the
# single-step up-rung the P3 table omits (see HANDOFF.md); it is required for a
# cost-ladder walk that starts from a float region.  The three "skip"
# transitions (float-to-double, double-to-float, ff-to-dd) are valid Patcher
# kinds but are NOT emitted by the current single-step walk.
# ---------------------------------------------------------------------------

TRANSITION_KINDS: frozenset[str] = frozenset({
    # single-step up (correctness)
    "float-to-ff", "ff-to-double", "double-to-dd",
    # single-step down (speedup)
    "dd-to-double", "double-to-ff", "ff-to-float",
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
