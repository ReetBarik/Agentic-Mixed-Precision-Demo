"""Phase 2c — source-derived promotion reads + the ``promotion_no_op`` gate.

Two behaviours land in fan-out:

* when an intent carries no reads (qcdloop's template regions report
  ``region_local_vars=[]``), :func:`fan_out_region` derives the region's scalar
  reads from source, so the promoted variant body is genuinely retyped (no longer a
  bit-identical clone); and
* when even the derived reads promote nothing (an empty payload), the result flags
  ``promotion_applied=False`` — which the dispatcher turns into a terminal
  ``promotion_no_op`` failure instead of a silent, inert ``measured`` candidate.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agents.patcher import fanout
from tests.patcher.fanout.conftest import requires_libclang

# A tree where the leaf ``f`` has a promotable region (reads a scalar local) and a
# NON-promotable region (writes a view-like element from an int only — no scalar
# read), so both sides of the gate are exercised on real call-graph output.
APP_H = """\
#pragma once
namespace app {

template<class T>
T f(T x) {
    T a = x + T(1);
    T b = a * T(2);
    int k = 3;
    T c = T(k);
    return b + c;
}

template<class T>
T entry(T x) {
    return f<T>(x);
}

}  // namespace app
"""


def _graph(tree: Path):
    from agents.patcher.call_graph import build_call_graph
    fanout.clear_graph_cache()
    return build_call_graph("entry", tree, tu_file=tree / "app.h")


def _line_with(tree: Path, needle: str) -> int:
    for i, ln in enumerate(( tree / "app.h").read_text().split("\n"), start=1):
        if needle in ln:
            return i
    raise AssertionError(f"{needle!r} not in app.h")


@pytest.fixture
def tree(tmp_path) -> Path:
    (tmp_path / "app.h").write_text(APP_H)
    return tmp_path


@requires_libclang
def test_empty_reads_are_derived_and_promotion_applies(tree):
    g = _graph(tree)
    ln = _line_with(tree, "T b = a * T(2);")
    res = fanout.fan_out_region(
        file="app.h", line_start=ln, line_end=ln, reads=[], writes=[],
        integral="B1", graph=g, tree_root=tree,
        scalar_type="Ext", two_limb=False, shim_include="ext.h")

    # reads were derived from source: `a` (scalar local), not `b` (the write target).
    assert res.reads_used == ["a"]
    assert res.promotion_applied is True

    block = (tree / "app.h").read_text()
    block = block[block.index(fanout._BLOCK_BEGIN):]
    # the derived read was promoted inside the variant body — not a verbatim clone.
    assert "Ext a__ff = Ext(a);" in block


@requires_libclang
def test_empty_payload_flags_promotion_not_applied(tree):
    g = _graph(tree)
    ln = _line_with(tree, "T c = T(k);")     # writes c from int k → no scalar read
    res = fanout.fan_out_region(
        file="app.h", line_start=ln, line_end=ln, reads=[], writes=[],
        integral="B1", graph=g, tree_root=tree,
        scalar_type="Ext", two_limb=False, shim_include="ext.h")

    assert res.reads_used == []                # k is int, c is the write target
    assert res.promotion_applied is False      # empty payload → gate will fire


@requires_libclang
def test_explicit_reads_still_honored(tree):
    """A non-empty intent reads set is used verbatim (derivation only fills a gap)."""
    g = _graph(tree)
    ln = _line_with(tree, "T b = a * T(2);")
    res = fanout.fan_out_region(
        file="app.h", line_start=ln, line_end=ln, reads=["a"], writes=[],
        integral="B1", graph=g, tree_root=tree,
        scalar_type="Ext", two_limb=False, shim_include="ext.h")
    assert res.reads_used == ["a"] and res.promotion_applied is True


# --------------------------------------------------------------------------- #
# dispatch gate: promotion_applied=False -> terminal PROMOTION_NO_OP
# --------------------------------------------------------------------------- #

from types import SimpleNamespace                                      # noqa: E402

from agents.patcher import dispatch, fanout as _fo, result as R        # noqa: E402
from agents.shared import region_scan as _rs                           # noqa: E402
from agents.strategy.models import RegionTarget, RemediationIntent     # noqa: E402


def _dd_intent():
    return RemediationIntent(
        target=RegionTarget(file="app.h", line_start=6, line_end=6, variables=[]),
        kind="double-to-dd", intent="correctness", current_precision="double",
        rationale_id="r1")


def _deps(tmp_path):
    fanset = SimpleNamespace(enabled=True, integral="B1", max_paths=8)
    good_res = SimpleNamespace(ok=True, shim_paths=[], llm_tokens=0, error=None,
                               boundary_patch=None)
    return dispatch.PatchDeps(
        repo_root=tmp_path, parent_sha="HEAD",
        target_path=tmp_path / "app.h", shims_dir=tmp_path, patches_dir=tmp_path,
        integrators={"dd": lambda **kw: good_res}, llm_call=None, fanout=fanset)


def _patch_fanout(monkeypatch, *, promotion_applied):
    monkeypatch.setattr(_fo, "graph_for_pass", lambda *a, **k: object())
    monkeypatch.setattr(_rs, "extract_region_writes", lambda *a, **k: [])
    fr = _fo.FanoutResult(declared_variants=["f_B1"], files_touched=["app.h"],
                          in_place_region=False,
                          promotion_applied=promotion_applied, reads_used=[])
    monkeypatch.setattr(_fo, "fan_out_region", lambda **k: fr)


def test_gate_fires_on_empty_payload(tmp_path, monkeypatch):
    (tmp_path / "app.h").write_text("x\n")
    _patch_fanout(monkeypatch, promotion_applied=False)
    gen = dispatch._gen_regional_fanout(_dd_intent(), _deps(tmp_path), attempt=0)
    assert not gen.ok
    assert gen.status == R.PROMOTION_NO_OP
    assert "promotion_no_op" in gen.detail


def test_gate_passes_when_promotion_applied(tmp_path, monkeypatch):
    (tmp_path / "app.h").write_text("x\n")
    _patch_fanout(monkeypatch, promotion_applied=True)
    gen = dispatch._gen_regional_fanout(_dd_intent(), _deps(tmp_path), attempt=0)
    assert gen.ok
    assert gen.status is None
    assert gen.declared_variants == ["f_B1"]
