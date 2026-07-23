"""Rename cascade — variants, reroutes, in-place root edit, originals untouched."""

from __future__ import annotations

from pathlib import Path

from agents.patcher import fanout
from tests.patcher.fanout.conftest import requires_libclang


def _region_in(graph, name, needle="T b = a * T(2);"):
    """A one-line region inside ``name`` containing ``needle`` (reads a local)."""
    fd = graph.defs[name][0]
    lines = Path(fd.file).read_text().split("\n")
    for ln in range(fd.line_start, fd.line_end + 1):
        if needle in lines[ln - 1]:
            return ln, ln
    raise AssertionError(f"{needle!r} not found in {name} [{fd.line_start}-{fd.line_end}]")


@requires_libclang
def test_cascade_structure(synth_tree, synth_graph):
    g = synth_graph
    ls, le = _region_in(g, "f")
    res = fanout.fan_out_region(
        file="app.h", line_start=ls, line_end=le, reads=["a"], writes=[],
        integral="B1", graph=g, tree_root=synth_tree,
        scalar_type="Ext", two_limb=False, shim_include=None)

    assert set(res.declared_variants) == {
        "h_B1", "h2_B1", "g_h_B1", "g2_h_B1", "g_h2_B1",
        "f_g_h_B1", "f_g2_h_B1", "f_g_h2_B1",
    }
    assert res.root_edited and not res.in_place_region

    txt = (synth_tree / "app.h").read_text()

    # path entry -> h -> g -> f
    assert "T f_g_h_B1(" in txt                 # bottom variant defined
    assert "T g_h_B1(" in txt                    # intermediate defined
    assert "f_g_h_B1<" in txt                     # g_h_B1 calls f_g_h_B1
    assert "T h_B1(" in txt
    assert "g_h_B1<" in txt                        # h_B1 calls g_h_B1

    # entry body (in place, NOT renamed) reroutes to the first-level variants
    assert "T entry(" in txt and "entry_B1(" not in txt
    assert "h_B1<" in txt and "h2_B1<" in txt


@requires_libclang
def test_originals_unchanged(synth_tree, synth_graph):
    g = synth_graph
    ls, le = _region_in(g, "f")
    fanout.fan_out_region(
        file="app.h", line_start=ls, line_end=le, reads=["a"], writes=[],
        integral="B1", graph=g, tree_root=synth_tree,
        scalar_type="Ext", two_limb=False, shim_include=None)
    # slice out everything before the fan-out block: the originals must be verbatim
    txt = (synth_tree / "app.h").read_text()
    before = txt.split(fanout._BLOCK_BEGIN)[0]
    for orig in ("T f(T x) {", "T g(T x) {", "T h(T x) {"):
        assert orig in before
    # the original f body line is intact (the promoted copy lives only in the block)
    assert "    T b = a * T(2);" in before


@requires_libclang
def test_promoted_region_in_bottom_variant(synth_tree, synth_graph):
    g = synth_graph
    ls, le = _region_in(g, "f")
    fanout.fan_out_region(
        file="app.h", line_start=ls, line_end=le, reads=["a"], writes=[],
        integral="B1", graph=g, tree_root=synth_tree,
        scalar_type="Ext", two_limb=False, shim_include="ext.h")
    txt = (synth_tree / "app.h").read_text()
    block = txt[txt.index(fanout._BLOCK_BEGIN):]
    # the read 'a' was promoted to the extended scalar inside a variant
    assert "Ext a__ff = Ext(a);" in block
    assert '#include "ext.h"' in txt


@requires_libclang
def test_region_in_entry_promotes_in_place(synth_tree, synth_graph):
    g = synth_graph
    fd = g.defs["entry"][0]
    res = fanout.fan_out_region(
        file="app.h", line_start=fd.line_start + 1, line_end=fd.line_start + 1,
        reads=[], writes=[], integral="B1", graph=g, tree_root=synth_tree,
        scalar_type="Ext", two_limb=False, shim_include="ext.h")
    assert res.in_place_region and res.declared_variants == []
    txt = (synth_tree / "app.h").read_text()
    assert fanout._BLOCK_BEGIN not in txt           # no variant block
    assert "entry_B1(" not in txt                    # entry not renamed
