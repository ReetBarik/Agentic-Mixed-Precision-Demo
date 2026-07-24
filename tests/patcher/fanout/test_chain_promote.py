"""Phase 2f — chain_promote: coordinated multi-region multi-file dd promotion.

Uses a self-contained call graph ``entry -> g -> f`` where every function has a
promotable decl region, so a chain can span f + g + entry.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.patcher import fanout
from agents.patcher.chain_promote import (
    ChainFanoutResult, ChainManifest, chain_promote, chain_promotion_no_op,
    chain_write_truncation,
)
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


@pytest.fixture
def chain_tree(tmp_path) -> Path:
    (tmp_path / "app.h").write_text(CHAIN_H)
    return tmp_path


@pytest.fixture
def chain_graph(chain_tree):
    from agents.patcher.call_graph import build_call_graph
    fanout.clear_graph_cache()
    return build_call_graph("entry", chain_tree, tu_file=chain_tree / "app.h")


def _line(graph, name, needle):
    fd = graph.defs[name][0]
    lines = Path(fd.file).read_text().split("\n")
    for ln in range(fd.line_start, fd.line_end + 1):
        if needle in lines[ln - 1]:
            return ln
    raise AssertionError(f"{needle!r} not found in {name}")


def _manifest_specs(tree: Path):
    """Parse the AMP-FANOUT block back into {variant_name: VariantSpec}."""
    lines = (tree / "app.h").read_text().split("\n")
    specs, _ = fanout._extract_block(lines)
    return specs


# --------------------------------------------------------------------------- #
# pure gate functions (no libclang / build)
# --------------------------------------------------------------------------- #

def test_chain_promotion_no_op_fires_only_when_all_empty():
    assert chain_promotion_no_op([False, False, False]) is True
    assert chain_promotion_no_op([]) is True
    # a single non-empty link keeps the chain alive (link exemption)
    assert chain_promotion_no_op([False, True, False]) is False
    assert chain_promotion_no_op([True]) is False


def test_chain_write_truncation_delegates_to_boundary(monkeypatch):
    captured = {}

    def fake(region_text, reads, writes, two_limb, *, caller_type="double",
             complex_tokens=frozenset(), caller_complex=None):
        captured.update(region_text=region_text, reads=reads, writes=writes,
                        two_limb=two_limb, caller_type=caller_type)
        return True

    monkeypatch.setattr("agents.patcher.chain_promote.boundary.write_truncation_inert", fake)
    out = chain_write_truncation(
        outermost_region_text="R", outermost_reads=["a"], outermost_writes=["b"],
        two_limb=True, caller_type="double")
    assert out is True
    assert captured == dict(region_text="R", reads=["a"], writes=["b"],
                            two_limb=True, caller_type="double")


# --------------------------------------------------------------------------- #
# coordinated multi-region promotion (real call graph)
# --------------------------------------------------------------------------- #

@requires_libclang
def test_chain_spans_two_functions_variant_carries_promote_and_reroute(chain_tree, chain_graph):
    g = chain_graph
    f_line = _line(g, "f", "T b = a * T(2);")
    g_line = _line(g, "g", "T d = c - T(3);")
    man = ChainManifest(
        chain_id="cascade_B12_x", integral="B12", entry_point="entry",
        lines=[("app.h", f_line, f_line), ("app.h", g_line, g_line)])

    res = chain_promote(manifest=man, graph=g, tree_root=chain_tree,
                        scalar_type="Ext", two_limb=False, shim_include=None)

    assert isinstance(res, ChainFanoutResult)
    assert res.promotion_applied is True
    assert set(res.declared_variants) == {"g_B12", "f_g_B12"}

    specs = _manifest_specs(chain_tree)
    # g's variant carries BOTH its own region promotion AND the reroute into f's variant
    assert "g_B12" in specs and "f_g_B12" in specs
    assert specs["g_B12"].reroutes.get("f") == "f_g_B12"
    assert len(specs["g_B12"].promotes) == 1          # g's chain region
    assert len(specs["f_g_B12"].promotes) == 1        # f's chain region


@requires_libclang
def test_chain_single_block_and_callee_before_caller(chain_tree, chain_graph):
    g = chain_graph
    f_line = _line(g, "f", "T b = a * T(2);")
    g_line = _line(g, "g", "T d = c - T(3);")
    chain_promote(manifest=ChainManifest(
        chain_id="c", integral="B12", entry_point="entry",
        lines=[("app.h", f_line, f_line), ("app.h", g_line, g_line)]),
        graph=g, tree_root=chain_tree, scalar_type="Ext", two_limb=False,
        shim_include=None)

    txt = (chain_tree / "app.h").read_text()
    # exactly one fan-out block for the whole chain
    assert txt.count(fanout._BLOCK_BEGIN) == 1
    # callee variant f_g_B12 defined before its caller variant g_B12 (qualified lookup)
    assert txt.index("T f_g_B12(") < txt.index("T g_B12(")
    # g_B12's body calls f_g_B12; entry body rerouted to g_B12, entry not renamed
    assert "f_g_B12<" in txt
    assert "g_B12<" in txt and "T entry(" in txt and "entry_B12(" not in txt


@requires_libclang
def test_two_regions_same_function_accumulate_two_promotes(chain_tree, chain_graph):
    g = chain_graph
    a_line = _line(g, "f", "T a = x + T(1);")
    b_line = _line(g, "f", "T b = a * T(2);")
    chain_promote(manifest=ChainManifest(
        chain_id="c", integral="B12", entry_point="entry",
        lines=[("app.h", a_line, a_line), ("app.h", b_line, b_line)]),
        graph=g, tree_root=chain_tree, scalar_type="Ext", two_limb=False,
        shim_include=None)

    specs = _manifest_specs(chain_tree)
    assert "f_g_B12" in specs
    assert len(specs["f_g_B12"].promotes) == 2        # both regions on one variant


@requires_libclang
def test_per_integral_naming_and_originals_untouched(chain_tree, chain_graph):
    g = chain_graph
    f_line = _line(g, "f", "T b = a * T(2);")
    orig_f = "\n".join(fanout._original_text(g.defs["f"][0].file,
                                             g.defs["f"][0].line_start,
                                             g.defs["f"][0].line_end))
    res = chain_promote(manifest=ChainManifest(
        chain_id="c", integral="B14", entry_point="entry",
        lines=[("app.h", f_line, f_line)]),
        graph=g, tree_root=chain_tree, scalar_type="Ext", two_limb=False,
        shim_include=None)

    # integral-scoped naming (B14, not B12); original f body still present verbatim
    assert any(v.endswith("_B14") for v in res.declared_variants)
    assert not any(v.endswith("_B12") for v in res.declared_variants)
    txt = (chain_tree / "app.h").read_text()
    before_block = txt.split(fanout._BLOCK_BEGIN)[0]
    assert orig_f in before_block                      # original never edited


@requires_libclang
def test_entry_point_region_promoted_in_place(chain_tree, chain_graph):
    g = chain_graph
    r_line = _line(g, "entry", "T r = e + T(4);")
    res = chain_promote(manifest=ChainManifest(
        chain_id="c", integral="B12", entry_point="entry",
        lines=[("app.h", r_line, r_line)]),
        graph=g, tree_root=chain_tree, scalar_type="Ext", two_limb=False,
        shim_include=None)

    assert res.in_place_regions == 1
    assert res.root_edited is True
    assert res.declared_variants == []                 # in-place produces no new symbol
    assert res.promotion_applied is True


@requires_libclang
def test_chain_scope_2c_all_empty_gates(chain_tree, chain_graph):
    # A return-only region has a pure read and no landing -> upcast promotes nothing.
    g = chain_graph
    ret_f = _line(g, "f", "return b;")
    res = chain_promote(manifest=ChainManifest(
        chain_id="c", integral="B12", entry_point="entry",
        lines=[("app.h", ret_f, ret_f)]),
        graph=g, tree_root=chain_tree, scalar_type="Ext",
        two_limb=True, shim_include=None)          # two_limb upcast, no landing
    assert res.promotion_applied is False
    assert chain_promotion_no_op([False]) is True


@requires_libclang
def test_unreachable_region_raises(chain_tree, chain_graph):
    from agents.patcher.fanout import FanoutError
    g = chain_graph
    with pytest.raises(FanoutError):
        chain_promote(manifest=ChainManifest(
            chain_id="c", integral="B12", entry_point="entry",
            lines=[("app.h", 999, 999)]),          # no enclosing function
            graph=g, tree_root=chain_tree, scalar_type="Ext", two_limb=False,
            shim_include=None)
