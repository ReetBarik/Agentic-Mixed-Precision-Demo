"""Phase 2f — Patcher dispatch chain path (agents.patcher.dispatch._gen_chain)."""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.integrator_base.region import RegionIntegrationResult
from agents.patcher import dispatch, result as R
from agents.patcher.fanout import FanoutSettings
from agents.strategy.models import RegionTarget, RemediationIntent, VIA_CHAIN
from tests.patcher.fanout.conftest import requires_libclang

CHAIN_H = """\
#pragma once
namespace app {

template<class T>
T f(T x) {
    T a = x + T(1);
    T b = a * T(2);
    return b;
}

template<class T>
T g(T x) {
    T c = f<T>(x);
    T d = c - T(3);
    return d;
}

template<class T>
T entry(T x) {
    T e = g<T>(x);
    T r = e + T(4);
    return r;
}

}  // namespace app
"""


class _MockIntegrator:
    """Records calls; returns an ok RegionIntegrationResult (no real LLM)."""

    def __init__(self, ok=True):
        self.ok = ok
        self.calls = []

    def __call__(self, **kw):
        self.calls.append(kw)
        if not self.ok:
            return RegionIntegrationResult.failed("mock llm failure")
        return RegionIntegrationResult(
            status="ok", shim_paths=[f"shim_{len(self.calls)}.h"],
            boundary_patch=None, llm_tokens=7)


@pytest.fixture
def chain_setup(tmp_path, monkeypatch):
    """A tree + call graph + PatchDeps wired for the chain path."""
    from agents.patcher.call_graph import build_call_graph
    from agents.patcher import fanout as fo

    root = tmp_path / "cand"
    root.mkdir()
    (root / "app.h").write_text(CHAIN_H)
    fo.clear_graph_cache()
    graph = build_call_graph("entry", root, tu_file=root / "app.h")
    # _gen_chain calls fo.graph_for_pass — feed it our tu_file-built graph.
    monkeypatch.setattr(fo, "graph_for_pass", lambda settings, tree_root: graph)

    def _line(name, needle):
        fd = graph.defs[name][0]
        lines = Path(fd.file).read_text().split("\n")
        for ln in range(fd.line_start, fd.line_end + 1):
            if needle in lines[ln - 1]:
                return ln
        raise AssertionError(needle)

    def make_deps(integrator):
        return dispatch.PatchDeps(
            repo_root=root, parent_sha="HEAD", target_path=root / "app.h",
            shims_dir=tmp_path / "shims", patches_dir=tmp_path / "patches",
            integrators={"chain_dd": integrator}, llm_call=None,
            fanout=FanoutSettings(entry_point="entry", integral="B12", enabled=True))

    return root, graph, _line, make_deps


def test_dispatch_path_routes_chain():
    assert dispatch.dispatch_path("double-to-dd", VIA_CHAIN) == dispatch.PATH_CHAIN


@requires_libclang
def test_gen_chain_promotes_and_declares_variants(chain_setup):
    root, graph, _line, make_deps = chain_setup
    intg = _MockIntegrator()
    f_line, g_line = _line("f", "T b = a * T(2);"), _line("g", "T d = c - T(3);")
    intent = RemediationIntent(
        target=RegionTarget("app.h", f_line, f_line, []), kind="double-to-dd",
        intent="correctness", current_precision="double", rationale_id="cascade_B12_x",
        via=VIA_CHAIN, chain_lines=[("app.h", f_line, f_line), ("app.h", g_line, g_line)])

    gen = dispatch.generate(intent, make_deps(intg), 0, dispatch.PATH_CHAIN)

    assert gen.ok, gen.detail
    assert set(gen.declared_variants) == {"g_B12", "f_g_B12"}
    assert gen.llm_tokens == 14                     # 7 per chain region
    assert len(intg.calls) == 2                     # one shim gen per chain region
    # the coordinated promotion actually landed a fan-out block in the tree
    assert "AMP-FANOUT-BEGIN" in (root / "app.h").read_text()


@requires_libclang
def test_gen_chain_all_empty_gates_promotion_no_op(chain_setup):
    root, graph, _line, make_deps = chain_setup
    # return-only regions: pure read, no landing -> dd upcast promotes nothing.
    ret_f, ret_g = _line("f", "return b;"), _line("g", "return d;")
    intent = RemediationIntent(
        target=RegionTarget("app.h", ret_f, ret_f, []), kind="double-to-dd",
        intent="correctness", current_precision="double", rationale_id="c",
        via=VIA_CHAIN, chain_lines=[("app.h", ret_f, ret_f), ("app.h", ret_g, ret_g)])

    gen = dispatch.generate(intent, make_deps(_MockIntegrator()), 0, dispatch.PATH_CHAIN)
    assert not gen.ok
    assert gen.status == R.PROMOTION_NO_OP


@requires_libclang
def test_gen_chain_integrator_failure_is_llm_gen_failed(chain_setup):
    root, graph, _line, make_deps = chain_setup
    f_line = _line("f", "T b = a * T(2);")
    intent = RemediationIntent(
        target=RegionTarget("app.h", f_line, f_line, []), kind="double-to-dd",
        intent="correctness", current_precision="double", rationale_id="c",
        via=VIA_CHAIN, chain_lines=[("app.h", f_line, f_line)])

    gen = dispatch.generate(intent, make_deps(_MockIntegrator(ok=False)), 0,
                            dispatch.PATH_CHAIN)
    assert not gen.ok
    assert gen.status == R.LLM_GEN_FAILED


@requires_libclang
def test_gen_chain_bypasses_awaiting_rewrite_filter(chain_setup):
    # A cascade region would normally short-circuit to awaiting_algorithmic_rewrite;
    # the chain path is the FIX for cascades, so it must NOT be gated by that filter.
    root, graph, _line, make_deps = chain_setup
    f_line = _line("f", "T b = a * T(2);")
    deps = make_deps(_MockIntegrator())
    # mark the region as a cascade in the signal_class map (would trip the filter)
    deps.fanout.signal_class_by_region = {f"app.h:{f_line}": "cancellation_cascade"}
    intent = RemediationIntent(
        target=RegionTarget("app.h", f_line, f_line, []), kind="double-to-dd",
        intent="correctness", current_precision="double", rationale_id="c",
        via=VIA_CHAIN, chain_lines=[("app.h", f_line, f_line)])

    gen = dispatch.generate(intent, deps, 0, dispatch.PATH_CHAIN)
    assert gen.status != R.AWAITING_ALGORITHMIC_REWRITE
    assert gen.ok, gen.detail
