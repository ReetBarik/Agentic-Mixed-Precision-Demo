"""Call-graph resolution + multi-path enumeration (libclang)."""

from __future__ import annotations

import pytest

from agents.patcher.call_graph import CallGraphError, build_call_graph
from tests.patcher.fanout.conftest import requires_kokkos_utils, requires_libclang


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


# --------------------------------------------------------------------------- #
# Phase 2f — template-extent token-scan fallback (Tier-B chain-promote blocker)
#
# libclang, parsing the real template-heavy kokkosUtils.h in a broken-include
# context, DROPS some function templates (ddilog @162-232, kfn @1196-1217) and
# TRUNCATES others (Lnrat -> [153,155] keeping only one overload; Li2omx2 body
# cut short).  Every dominant Tier-B cascade chain routes through these helpers,
# so chain lines resolved to "not inside any known function" and _gen_chain
# hard-failed.  The fallback must recover the real extents WITHOUT disturbing the
# non-template defs libclang already gets right.
# --------------------------------------------------------------------------- #


@requires_libclang
@requires_kokkos_utils
def test_template_ddilog_extent_recovered(kokkos_utils_graph):
    """ddilog (dropped by libclang) resolves for lines inside its body."""
    g = kokkos_utils_graph
    # blank line inside the body and the last real statement both land in ddilog
    for line in (174, 212):
        fd = g.enclosing_function("kokkosUtils.h", line)
        assert fd is not None, f"kokkosUtils.h:{line} resolved to None"
        assert fd.name == "ddilog", f"kokkosUtils.h:{line} -> {fd.name}, want ddilog"
        assert fd.line_start <= 162, f"ddilog starts at {fd.line_start}, want <=162"
        assert fd.line_end >= 232, f"ddilog ends at {fd.line_end}, want >=232"
        assert fd.is_template


@requires_libclang
@requires_kokkos_utils
def test_template_kfn_extent_recovered(kokkos_utils_graph):
    """kfn (dropped by libclang) contains B14's chain line 1208."""
    g = kokkos_utils_graph
    fd = g.enclosing_function("kokkosUtils.h", 1208)
    assert fd is not None, "kokkosUtils.h:1208 resolved to None"
    assert fd.name == "kfn", f"kokkosUtils.h:1208 -> {fd.name}, want kfn"
    assert fd.line_start <= 1196 and fd.line_end >= 1217
    assert fd.is_template


@requires_libclang
@requires_kokkos_utils
def test_template_lnrat_extent_not_truncated(kokkos_utils_graph):
    """Lnrat's first overload body (truncated away by libclang) is recovered.

    The (TOutput, TOutput) overload spans 140-150; libclang kept only the
    (TScale, TScale) overload at [153,155].  A line inside the first overload's
    body (146) must now resolve to Lnrat.
    """
    g = kokkos_utils_graph
    fd = g.enclosing_function("kokkosUtils.h", 146)
    assert fd is not None, "kokkosUtils.h:146 resolved to None"
    assert fd.name == "Lnrat", f"kokkosUtils.h:146 -> {fd.name}, want Lnrat"
    assert fd.line_start <= 140 and fd.line_end >= 150
    # both overloads present as distinct defs
    starts = sorted(d.line_start for d in g.defs["Lnrat"])
    assert starts[0] <= 140 and any(s >= 152 for s in starts), starts


@requires_libclang
@requires_kokkos_utils
def test_non_template_helper_extent_preserved(kokkos_utils_graph):
    """Fix is strictly additive: Li2omx2 (dilog-of-1-product) still resolves.

    Li2omx2's first overload body is [692,712]; line 702 (inside it) must resolve
    to Li2omx2, unchanged by the fallback.
    """
    g = kokkos_utils_graph
    fd = g.enclosing_function("kokkosUtils.h", 702)
    assert fd is not None and fd.name == "Li2omx2"
    assert fd.line_start <= 692 and fd.line_end >= 712


@requires_libclang
@requires_kokkos_utils
def test_edges_into_recovered_templates(kokkos_utils_graph):
    """Recovered templates participate in edges: Li2omx2 calls ddilog & Lnrat."""
    g = kokkos_utils_graph
    assert g.has("ddilog") and g.has("kfn")
    # Li2omx2 body calls ddilog and Lnrat (token-scan edges over the real body)
    assert g.callees_of("Li2omx2") >= {"ddilog", "Lnrat"}
