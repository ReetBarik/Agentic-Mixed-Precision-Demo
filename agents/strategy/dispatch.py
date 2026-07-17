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


# The 8 Patcher statuses (P2), exhaustive.
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
