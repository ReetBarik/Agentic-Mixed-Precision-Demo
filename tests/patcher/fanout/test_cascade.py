"""Rename cascade — variants, reroutes, in-place root edit, originals untouched."""

from __future__ import annotations

from pathlib import Path

from agents.patcher import fanout
from agents.patcher.fanout import VariantSpec
from tests.patcher.fanout.conftest import requires_libclang


def _spec(name: str, orig: str, reroutes: dict[str, str] | None = None) -> VariantSpec:
    """A VariantSpec carrying only what ``_topo_order`` reads (name + reroutes)."""
    return VariantSpec(variant_name=name, orig_name=orig, file="x.h",
                       orig_start=1, orig_end=1, reroutes=dict(reroutes or {}))


def _assert_callees_first(specs: dict[str, VariantSpec]) -> None:
    """Every rerouted callee variant must be emitted before its caller."""
    ordered = fanout._topo_order(specs)
    pos = {s.variant_name: i for i, s in enumerate(ordered)}
    assert set(pos) == set(specs)                       # no variant dropped
    for caller in specs.values():
        for callee in caller.reroutes.values():
            if callee in pos:
                assert pos[callee] < pos[caller.variant_name], (
                    f"callee {callee} emitted after caller {caller.variant_name}")


def test_topo_order_when_alpha_inverts_topo():
    """Regression: qualified template-id ``ql::<callee>`` is looked up at definition
    time, so a callee variant must be *defined earlier* than its caller.  On real
    qcdloop the alphabetical variant name of the caller (``B0m_B1``) sorts *before*
    its callee (``B1_B0m_B1``) — the old ``sorted(specs)`` emission put the caller
    first and the vanilla build failed.  Topological emission must fix the order even
    though the synthetic ``f_g_h`` naming (leaf-first) happens to hide it."""
    specs = {
        "B0m_B1": _spec("B0m_B1", "B0m", {"B1": "B1_B0m_B1"}),  # caller (sorts first)
        "B1_B0m_B1": _spec("B1_B0m_B1", "B1"),                  # callee (sorts second)
    }
    ordered = fanout._topo_order(specs)
    assert [s.variant_name for s in ordered] == ["B1_B0m_B1", "B0m_B1"]
    _assert_callees_first(specs)


def test_topo_order_deep_chain_and_diamond():
    """Multi-level + fan-in: names chosen so alphabetical order contradicts the
    dependency order at every edge, and one callee has two callers (diamond)."""
    # dep chain c3 -> c2 -> c1 (c1 is the deepest/leaf); alpha order is c1<c2<c3,
    # i.e. leaf LAST alphabetically — the inverse of what emission needs.
    specs = {
        "c3": _spec("c3", "top", {"mid": "c2"}),
        "c2": _spec("c2", "mid", {"leaf": "c1", "leaf2": "d1"}),  # diamond: two callees
        "c1": _spec("c1", "leaf"),
        "d1": _spec("d1", "leaf2"),
    }
    _assert_callees_first(specs)


def test_topo_order_stable_and_manifest_name_sorted():
    """Emission is deterministic; the manifest key list stays name-sorted regardless
    of the (topological) emission order, so the spec record is a stable diff."""
    specs = {
        "B0m_B1": _spec("B0m_B1", "B0m", {"B1": "B1_B0m_B1"}),
        "B1_B0m_B1": _spec("B1_B0m_B1", "B1"),
    }
    assert [s.variant_name for s in fanout._topo_order(specs)] == \
           [s.variant_name for s in fanout._topo_order(specs)]
    assert sorted(specs) == ["B0m_B1", "B1_B0m_B1"]     # manifest order (name-sorted)


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
