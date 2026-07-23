"""Call-graph resolution + multi-path enumeration (libclang)."""

from __future__ import annotations

import pytest

from agents.patcher.call_graph import CallGraphError, build_call_graph
from tests.patcher.fanout.conftest import requires_libclang


@requires_libclang
def test_defs_and_extents(synth_graph):
    g = synth_graph
    for name in ("entry", "h", "h2", "g", "g2", "f"):
        assert g.has(name), f"{name} missing from defs"
    # extents are sane (start <= end, positive)
    fd = g.defs["f"][0]
    assert fd.line_start >= 1 and fd.line_end >= fd.line_start


@requires_libclang
def test_edges_recovered_from_dependent_calls(synth_graph):
    g = synth_graph
    # dependent template calls the AST drops must still be edges via the token scan
    assert g.callees_of("entry") >= {"h", "h2"}
    assert g.callees_of("h") >= {"g", "g2"}
    assert g.callees_of("h2") >= {"g"}
    assert g.callees_of("g") >= {"f"}
    assert g.callees_of("g2") >= {"f"}
    assert g.callers_of("f") >= {"g", "g2"}


@requires_libclang
def test_enclosing_function(synth_graph):
    g = synth_graph
    fd = g.defs["f"][0]
    mid = (fd.line_start + fd.line_end) // 2
    assert g.enclosing_function("app.h", mid).name == "f"
    # bare basename resolves the same as a path
    assert g.enclosing_function("app.h", fd.line_start).name == "f"


@requires_libclang
def test_multi_path_enumeration(synth_graph):
    g = synth_graph
    paths, truncated = g.enumerate_paths("f")
    assert not truncated
    got = {tuple(p) for p in paths}
    assert got == {
        ("entry", "h", "g", "f"),
        ("entry", "h", "g2", "f"),
        ("entry", "h2", "g", "f"),
    }
    # two paths to g (fan-in on g)
    gp, _ = g.enumerate_paths("g")
    assert {tuple(p) for p in gp} == {("entry", "h", "g"), ("entry", "h2", "g")}


@requires_libclang
def test_root_path_is_trivial(synth_graph):
    paths, truncated = synth_graph.enumerate_paths("entry")
    assert paths == [["entry"]] and not truncated


@requires_libclang
def test_path_cap_flags_truncation(synth_graph):
    paths, truncated = synth_graph.enumerate_paths("f", max_paths=1)
    assert truncated and len(paths) == 1


@requires_libclang
def test_fail_loud_on_unknown_entry(synth_tree):
    with pytest.raises(CallGraphError):
        build_call_graph("NoSuchEntry", synth_tree, tu_file=synth_tree / "app.h")


@requires_libclang
def test_detect_tu_file(synth_tree):
    # no tu_file passed -> auto-detect the header defining `entry`
    g = build_call_graph("entry", synth_tree)
    assert g.tu_file.endswith("app.h")
