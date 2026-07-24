"""Phase 2e signal_class filter — precision rungs skipped on cancellation regions.

A precision-rung intent on a ``cancellation_cascade`` / ``local_cancellation``
region must short-circuit to the terminal ``awaiting_algorithmic_rewrite`` status
BEFORE any LLM shim generation or build; ``stable`` / ``log_near_root`` regions (and
reformulate kinds, and passes with no signal_class map) pass through unchanged.
"""

from pathlib import Path

import pytest

from agents.patcher import dispatch, result as R
from agents.patcher.fanout import (FanoutSettings, awaits_algorithmic_rewrite,
                                   signal_class_map)
from agents.strategy.models import RegionTarget, RemediationIntent


def _intent(location="B2m.h:648", kind="double-to-dd", flavor="correctness",
            identity=None):
    file, ls = location.rsplit(":", 1)
    if "-" in ls:
        a, b = ls.split("-"); line_start, line_end = int(a), int(b)
    else:
        line_start = line_end = int(ls)
    return RemediationIntent(
        target=RegionTarget(file=file, line_start=line_start, line_end=line_end),
        kind=kind, intent=flavor, current_precision="double",
        rationale_id="iter_1", identity=identity)


def _deps(signal_class_by_region):
    """A PatchDeps whose integrators/llm_call BLOW UP if touched — proving the
    filter short-circuits upstream of any generation."""
    def _boom(*a, **k):
        raise AssertionError("integrator/LLM must NOT be called for a filtered region")

    fanout = FanoutSettings(entry_point="BO", integral="B12",
                            signal_class_by_region=signal_class_by_region)
    return dispatch.PatchDeps(
        repo_root=Path("/nonexistent"), parent_sha="deadbeef",
        target_path=Path("/nonexistent/B2m.h"), shims_dir=Path("/tmp/shims"),
        patches_dir=Path("/tmp/patches"),
        integrators={"ff": _boom, "dd": _boom}, llm_call=_boom, fanout=fanout)


# --- the helper -------------------------------------------------------------
def test_awaits_helper_classifies_only_cancellation_classes():
    assert awaits_algorithmic_rewrite("cancellation_cascade")
    assert awaits_algorithmic_rewrite("local_cancellation")
    assert not awaits_algorithmic_rewrite("stable")
    assert not awaits_algorithmic_rewrite("log_near_root")
    assert not awaits_algorithmic_rewrite(None)


def test_signal_class_map_from_report_regions():
    regions = {
        "B2m.h:648": {"signal_class": "cancellation_cascade"},
        "B2m.h:10": {"signal_class": "stable"},
        "": {"signal_class": "stable"},              # empty key dropped
        "B2m.h:20": {"note": "no class"},            # no signal_class dropped
    }
    m = signal_class_map(regions)
    assert m == {"B2m.h:648": "cancellation_cascade", "B2m.h:10": "stable"}


# --- cascade / local: exactly one awaiting cell, no LLM/build ----------------
@pytest.mark.parametrize("sc", ["cancellation_cascade", "local_cancellation"])
def test_precision_rung_on_cancellation_region_awaits_rewrite(sc):
    deps = _deps({"B2m.h:648": sc})
    gen = dispatch.generate(_intent("B2m.h:648", "double-to-dd"), deps, 0,
                            dispatch.PATH_REGIONAL)
    assert not gen.ok
    assert gen.status == R.AWAITING_ALGORITHMIC_REWRITE
    assert gen.err_kind == R.ERR_AWAITING_REWRITE
    assert sc in gen.detail
    assert gen.llm_tokens == 0                        # no LLM spent


def test_multiline_cancellation_region_awaits_rewrite():
    deps = _deps({"B2m.h:100-105": "cancellation_cascade"})
    gen = dispatch.generate(_intent("B2m.h:100-105", "double-to-dd"), deps, 0,
                            dispatch.PATH_REGIONAL)
    assert gen.status == R.AWAITING_ALGORITHMIC_REWRITE


# --- stable / log_near_root: filter inert (usual enumeration proceeds) -------
@pytest.mark.parametrize("sc", ["stable", "log_near_root"])
def test_normal_regions_are_not_filtered(sc):
    deps = _deps({"B2m.h:648": sc})
    # The filter is inert (returns None), so generate() proceeds to the usual rung
    # enumeration for stable / log_near_root regions — no awaiting short-circuit.
    assert dispatch._awaiting_rewrite(_intent("B2m.h:648", "double-to-dd"), deps) is None


# --- exemptions -------------------------------------------------------------
def test_reformulate_kind_is_exempt_from_the_filter():
    # A reformulate kind IS the algorithmic fix — never short-circuited even on a
    # cascade region.
    deps = _deps({"B2m.h:648": "cancellation_cascade"})
    intent = _intent("B2m.h:648", "reformulate-kahan")
    assert dispatch._awaiting_rewrite(intent, deps) is None


def test_no_signal_class_map_is_fail_open():
    deps = _deps({})                                 # no map supplied
    assert dispatch._awaiting_rewrite(_intent("B2m.h:648", "double-to-dd"), deps) is None


def test_region_absent_from_map_is_not_filtered():
    deps = _deps({"other.h:1": "cancellation_cascade"})
    assert dispatch._awaiting_rewrite(_intent("B2m.h:648", "double-to-dd"), deps) is None
