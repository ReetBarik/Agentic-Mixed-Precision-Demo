"""P6 dispatch — map a Patcher return status to Strategy's response.

Implements the design's "Strategy retry response table" as a dispatch dict.
Each entry says what the main loop does with a Patcher result and how the
iteration is tagged in ``iterations.jsonl``.

Actions (the ``action`` field):

  * ``validate``    — Patcher committed ``ok``; hand the SHA to the Validator.
  * ``advance``     — treat as reject, advance the vocabulary walk one step.
  * ``advance_terminal`` — advance the walk AND never retry this (intent, region)
                     again this run (``llm_gen_failed``, P6b).
  * ``skip_intent`` — malformed intent (Strategy's bug); skip, don't count vs
                     budget (``patch_apply_failed``).
  * ``retry_once``  — retry the same intent once; a second timeout folds into
                     ``advance`` (``timeout``).
  * ``fatal``       — hard-abort the run with status ``internal_error``
                     (``commit_failed``, Q3).

``counts_budget`` is False for the buckets the design says not to charge against
the iteration budget (llm capacity + Strategy-bug + fatal).
``dd_untested`` marks statuses that, when hit at the ``dd`` rung, mean the DD
ceiling was NOT proven (P6a) — the region is untested, not a physics limit.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DispatchEntry:
    action: str
    log_tag: str
    counts_budget: bool
    # True → this status is Bucket A "signal about the intent" (a real reject).
    is_reject: bool
    # True → hitting this at the dd rung yields ceiling_kind dd_untested (P6a).
    dd_untested: bool


# Every Patcher status (``agents.patcher.result.STATUSES``), exhaustive.  The
# test suite pins ``set(DISPATCH) == set(result.STATUSES)`` so the two can't drift.
DISPATCH: dict[str, DispatchEntry] = {
    "ok": DispatchEntry(
        action="validate", log_tag="", counts_budget=True,
        is_reject=False, dd_untested=False),
    "build_failed": DispatchEntry(
        action="advance", log_tag="compile", counts_budget=True,
        is_reject=True, dd_untested=True),
    "runtime_nan": DispatchEntry(
        action="advance", log_tag="runtime_nan", counts_budget=True,
        is_reject=True, dd_untested=True),
    "runtime_crashed": DispatchEntry(
        action="advance", log_tag="runtime_crash", counts_budget=True,
        is_reject=True, dd_untested=True),
    "llm_gen_failed": DispatchEntry(
        action="advance_terminal", log_tag="llm_capacity", counts_budget=False,
        is_reject=True, dd_untested=True),
    "patch_apply_failed": DispatchEntry(
        action="skip_intent", log_tag="strategy_bug", counts_budget=False,
        is_reject=False, dd_untested=True),
    "timeout": DispatchEntry(
        action="retry_once", log_tag="timeout", counts_budget=True,
        is_reject=True, dd_untested=True),
    "commit_failed": DispatchEntry(
        action="fatal", log_tag="fatal", counts_budget=False,
        is_reject=False, dd_untested=True),
    # gen+build ok but the candidate == parent (no net change): a benign no-op,
    # NOT the Q3-fatal commit_failed.  Treat as a Bucket-A reject that advances the
    # walk; at the dd rung the ceiling was not proven → dd_untested.
    "empty_candidate": DispatchEntry(
        action="advance", log_tag="empty_candidate", counts_budget=True,
        is_reject=True, dd_untested=True),
    # A plain-edit `-to-float` rung inapplicable to a template-typed region (no bare
    # `double` token to rewrite).  Benign — NOT a strategy_bug: advance the walk
    # (speedup settles at the current rung).  Doesn't count vs budget (git-only, no
    # build) and doesn't bump the DR streak (it is not a physics/quality signal).
    "patch_inapplicable": DispatchEntry(
        action="advance", log_tag="patch_inapplicable", counts_budget=False,
        is_reject=False, dd_untested=True),
    # Phase 2c: the fan-out promotion produced a variant byte-identical to the
    # original (empty read/write payload even after source-derivation) — the
    # region has no promotable scalar operands, so NO rung would promote anything.
    # Terminal for this intent (never retry — the result is deterministic and
    # rung-independent), git-only so it doesn't count vs budget, and a reject that
    # leaves the rung dd_untested.  Distinct from empty_candidate (which is a
    # gen+build no-op) and llm_gen_failed (an LLM capacity miss): this is a
    # deterministic structural detection at gen time, upstream of any build.
    "promotion_no_op": DispatchEntry(
        action="advance_terminal", log_tag="promotion_no_op", counts_budget=False,
        is_reject=True, dd_untested=True),
    # Phase 2d-B: an UPCAST promotion that retyped the body but truncates every landing
    # back to caller precision (no wider persistent sink) → numerically inert.  Detected
    # at gen time upstream of any build, the upcast analogue of promotion_no_op: terminal
    # for this intent (deterministic + rung-fixed — a wider rung would truncate the same
    # way), git-only so it doesn't count vs budget, a reject that leaves the rung
    # dd_untested (no ceiling was proven — the region was skipped, not measured).
    "write_truncation": DispatchEntry(
        action="advance_terminal", log_tag="write_truncation", counts_budget=False,
        is_reject=True, dd_untested=True),
    # Phase 2e signal_class filter: a precision rung declined on a cancellation-cascade
    # / local-cancellation region (wider intermediates are structurally inert; the fix
    # is an algorithmic rewrite, not a wider type).  Detected at gen time upstream of
    # any build/LLM, so like promotion_no_op / write_truncation it is terminal for this
    # intent (deterministic + rung-independent), git-only (doesn't count vs budget), and
    # a reject that leaves the rung dd_untested — the region was skipped, not measured.
    # At the dd rung this yields dd_untested, so the correctness walk emits exactly one
    # such cell per region and then settles (it never reaches the reformulate phase).
    "awaiting_algorithmic_rewrite": DispatchEntry(
        action="advance_terminal", log_tag="awaiting_algorithmic_rewrite",
        counts_budget=False, is_reject=True, dd_untested=True),
    # Blocker A: a chain's strict carrier variable (written by one link, read by another)
    # whose decl the emission layer cannot widen — so the dd value re-narrows between links
    # and the chain fix is inert.  Detected at gen time (source-derived, upstream of any
    # build), rung-fixed (a wider rung re-narrows identically) → terminal for the intent,
    # git-only so free vs budget, a real reject, dd_untested (skipped, not measured).
    # `chain_carrier_unwidenable` — the carrier's decl is a function parameter (v1 refuses
    # to rewrite signatures); `chain_carrier_external` — the carrier's decl is global / a
    # class member / an output container (v1 refuses to widen shared state).
    "chain_carrier_unwidenable": DispatchEntry(
        action="advance_terminal", log_tag="chain_carrier_unwidenable",
        counts_budget=False, is_reject=True, dd_untested=True),
    "chain_carrier_external": DispatchEntry(
        action="advance_terminal", log_tag="chain_carrier_external",
        counts_budget=False, is_reject=True, dd_untested=True),
    # Closure-scoped generalisation (CLOSURE_SCOPED_CHAINS_DESIGN §2.4): a destination
    # escape materially severs a carried value's dd flow to a designed exit (a write to
    # shared state, or a non-benign extract).  Same terminal shape as the two carrier
    # terminals — gen-time, source-derived, rung-fixed, a real reject, dd_untested.
    "chain_closure_escapes": DispatchEntry(
        action="advance_terminal", log_tag="chain_closure_escapes",
        counts_budget=False, is_reject=True, dd_untested=True),
}


def dispatch(status: str) -> DispatchEntry:
    """Look up the dispatch entry for a Patcher status.

    An unknown status is a contract violation — fail loudly rather than guess.
    """
    try:
        return DISPATCH[status]
    except KeyError:
        raise ValueError(f"unknown Patcher status {status!r}; expected one of "
                         f"{sorted(DISPATCH)}")
