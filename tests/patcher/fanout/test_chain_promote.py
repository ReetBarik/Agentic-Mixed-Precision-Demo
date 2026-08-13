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


def test_chain_write_truncation_skips_outermost_region(monkeypatch):
    # A chain whose OUTERMOST region (min depth) truncates to caller precision but whose
    # INTERIOR regions all widen cleanly must NOT fire — the outermost store is the
    # chain's designed exit boundary, not evidence of inertness.
    checked = []

    def fake(region_text, reads, writes, two_limb, *, caller_type="double",
             complex_tokens=frozenset(), caller_complex=None, closure_names=frozenset()):
        checked.append(region_text)
        # Only the outermost region ("OUT") would trip the per-region detector.
        return region_text == "OUT"

    monkeypatch.setattr("agents.patcher.chain_promote.boundary.write_truncation_inert", fake)
    region_meta = [
        dict(depth=0, span=("f.h", 1, 1), region_text="OUT", reads=["a"], writes=["res"], promoted=True),
        dict(depth=1, span=("f.h", 2, 2), region_text="MID", reads=["c"], writes=["d"], promoted=True),
        dict(depth=2, span=("f.h", 3, 3), region_text="INNER", reads=["e"], writes=["f"], promoted=True),
    ]
    out = chain_write_truncation(region_meta, two_limb=True, caller_type="double")
    assert out is False                       # outermost truncation is exempt
    assert "OUT" not in checked               # outermost never handed to the detector
    assert set(checked) == {"MID", "INNER"}   # only interior regions checked


def test_chain_write_truncation_fires_on_interior_truncation(monkeypatch):
    # An INTERIOR write that truncates back to caller precision injects double roundoff
    # between links -> the chain is genuinely broken -> gate fires.
    def fake(region_text, reads, writes, two_limb, *, caller_type="double",
             complex_tokens=frozenset(), caller_complex=None, closure_names=frozenset()):
        return region_text == "MID"           # an interior region trips it

    monkeypatch.setattr("agents.patcher.chain_promote.boundary.write_truncation_inert", fake)
    region_meta = [
        dict(depth=0, span=("f.h", 1, 1), region_text="OUT", reads=["a"], writes=["res"], promoted=True),
        dict(depth=1, span=("f.h", 2, 2), region_text="MID", reads=["c"], writes=["d"], promoted=True),
    ]
    assert chain_write_truncation(region_meta, two_limb=True, caller_type="double") is True


def test_chain_write_truncation_single_region_never_fires(monkeypatch):
    # A single-region chain has no interior region — the lone region IS the designed
    # exit boundary, so the gate is a no-op even if the per-region detector would trip.
    monkeypatch.setattr(
        "agents.patcher.chain_promote.boundary.write_truncation_inert",
        lambda *a, **k: True)
    region_meta = [dict(depth=0, region_text="ONLY", reads=["a"], writes=["res"],
                        promoted=True)]
    assert chain_write_truncation(region_meta, two_limb=True) is False


# --------------------------------------------------------------------------- #
# clause (ii): widened-return designed-exit exemption (Subtask 2b)
# --------------------------------------------------------------------------- #

def test_designed_exit_kind_return_widened_on_but_plain_return_off():
    # The gate predicate: a rule-(c) widened return (``return_widened``) is exempt; a
    # plain ``return`` (frame return type NOT widened) still truncates and is checked.
    from agents.patcher.chain_promote import _designed_exit_kind
    assert _designed_exit_kind("return_widened") is True
    assert _designed_exit_kind("return") is False
    assert _designed_exit_kind("kernel_output") is True
    assert _designed_exit_kind("out_param") is True
    assert _designed_exit_kind("extract") is True


def test_gate_exempts_interior_region_all_widened_returns(monkeypatch):
    # An interior region every one of whose lines is a widened-return landing (clause ii)
    # is skipped — the value carries dd across, so it is NOT handed to the per-region
    # detector even though the detector would trip on it.
    checked = []

    def fake(region_text, reads, writes, two_limb, *, caller_type="double",
             complex_tokens=frozenset(), caller_complex=None, closure_names=frozenset()):
        checked.append(region_text)
        return True                            # detector would trip on anything

    monkeypatch.setattr("agents.patcher.chain_promote.boundary.write_truncation_inert", fake)
    region_meta = [
        dict(depth=0, span=("f.h", 1, 1), region_text="OUT", reads=["a"], writes=["res"], promoted=True),
        dict(depth=1, span=("g.h", 5, 5), region_text="RET", reads=["v"], writes=[], promoted=True),
    ]
    designed = [("g.h", 5, "return_widened", frozenset({"v"}), "v")]
    assert chain_write_truncation(
        region_meta, two_limb=True, designed_exits=designed) is False
    assert "RET" not in checked                 # widened-return region exempt, not checked


def test_gate_still_checks_plain_return_region(monkeypatch):
    # §3.3 correctness: a plain ``return`` (frame return type NOT widened — rule (c)
    # refused / did not reach it) is a real truncation, so the interior region is still
    # checked and the gate fires (B10 with rule (c) disabled must still reject at :707).
    monkeypatch.setattr(
        "agents.patcher.chain_promote.boundary.write_truncation_inert",
        lambda *a, **k: True)
    region_meta = [
        dict(depth=0, span=("f.h", 1, 1), region_text="OUT", reads=["a"], writes=["res"], promoted=True),
        dict(depth=1, span=("g.h", 5, 5), region_text="RET", reads=["v"], writes=[], promoted=True),
    ]
    designed = [("g.h", 5, "return", frozenset({"v"}), "v")]   # NOT widened
    assert chain_write_truncation(
        region_meta, two_limb=True, designed_exits=designed) is True


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


# --------------------------------------------------------------------------- #
# Blocker A end-to-end — a real carrier (Subtask 5 wiring)
#
# ``inner`` (interior, depth 2) declares a ``double`` carrier OUTSIDE the chain line
# set, writes it on one interior chain line and reads it on another; ``mid`` (depth 1)
# holds the chain's outermost/designed-exit region.  WITHOUT the carrier fix the carrier
# write is a Case-B truncating landing at the recognized caller type ``double`` → the
# interior 2d-B ``chain_write_truncation`` gate fires and the patch is (spuriously)
# rejected.  WITH the fix the carrier's decl is widened to dd, so the write is no longer
# a truncating landing, the gate stays silent, and the emitted variant carries BOTH the
# body promotions and the carrier decl-widen.
# --------------------------------------------------------------------------- #

DD = "Kokkos::Experimental::DoubleDouble"

CARRIER_CHAIN_H = """\
#pragma once
namespace app {

template<class T>
T inner(T x) {
    double carry;
    carry = x + T(1);
    T d2 = carry + x;
    return d2;
}

template<class T>
T mid(T x) {
    T c = inner<T>(x);
    T d = c - T(3);
    return d;
}

template<class T>
T entry(T x) {
    T e = mid<T>(x);
    return e + T(4);
}

}  // namespace app
"""


@pytest.fixture
def carrier_chain_tree(tmp_path) -> Path:
    (tmp_path / "app.h").write_text(CARRIER_CHAIN_H)
    return tmp_path


@pytest.fixture
def carrier_chain_graph(carrier_chain_tree):
    from agents.patcher.call_graph import build_call_graph
    fanout.clear_graph_cache()
    return build_call_graph("entry", carrier_chain_tree,
                            tu_file=carrier_chain_tree / "app.h")


@requires_libclang
def test_carrier_chain_widens_decl_and_gate_stays_silent(carrier_chain_tree,
                                                         carrier_chain_graph):
    from agents.integrator_base import boundary
    g = carrier_chain_graph
    a_line = _line(g, "inner", "carry = x + T(1);")   # interior: WRITES carrier
    b_line = _line(g, "inner", "T d2 = carry + x;")    # interior: READS carrier
    decl_line = _line(g, "inner", "double carry;")     # carrier decl (outside chain)
    outer_line = _line(g, "mid", "T d = c - T(3);")    # outermost/designed exit

    # Control: on the interior carrier-write region the per-region 2d-B detector FIRES
    # without carrier awareness (the carrier write is a truncating double landing) and
    # is SILENCED once the carrier is declared widened — this is exactly the spurious
    # rejection the fix removes.
    region_a = "    carry = x + T(1);"
    assert boundary.write_truncation_inert(
        region_a, ["x"], ["carry"], True, caller_type="double") is True
    assert boundary.write_truncation_inert(
        region_a, ["x"], ["carry"], True, caller_type="double",
        closure_names=frozenset({"carry"})) is False

    res = chain_promote(manifest=ChainManifest(
        chain_id="cascade_B10_x", integral="B10", entry_point="entry",
        lines=[("app.h", a_line, a_line), ("app.h", b_line, b_line),
               ("app.h", outer_line, outer_line)]),
        graph=g, tree_root=carrier_chain_tree, scalar_type=DD, two_limb=True,
        shim_include=None)

    # carrier recognized as widenable — no terminal carrier refusal
    assert res.chain_carrier_unwidenable is False
    assert res.chain_carrier_external is False
    assert res.closure_names == ["carry"]

    # the chain promotes AND the interior write_truncation gate does NOT fire (the
    # carrier fix is what keeps it silent — see the control assertions above)
    assert res.promotion_applied is True
    assert res.write_truncation is False

    # the inner variant carries BOTH body promotions AND the carrier decl-widen
    specs = _manifest_specs(carrier_chain_tree)
    inner_variant = specs["inner_mid_B10"]
    assert len(inner_variant.promotes) >= 1
    assert len(inner_variant.closure_decls) == 1
    cd = inner_variant.closure_decls[0]
    assert cd.decl_line == decl_line
    assert cd.orig_type == "double"
    assert cd.dd_type == DD
    assert cd.name == "carry"

    # rendered variant text: the carrier decl is widened to dd in the emitted copy,
    # while the ORIGINAL inner (before the fan-out block) keeps its double decl.
    txt = (carrier_chain_tree / "app.h").read_text()
    before_block, after_block = txt.split(fanout._BLOCK_BEGIN)
    assert "double carry;" in before_block           # original untouched
    assert f"{DD} carry;" in after_block             # variant decl widened


# --------------------------------------------------------------------------- #
# rule (c) end-to-end — cross-frame return propagation (Subtask 2b)
#
# ``callee`` (interior, depth 2) returns a carried value that ``caller`` (depth 1)
# consumes into two locals and cancels.  Rule (c) widens callee's variant RETURN TYPE
# and re-seeds caller's receiving locals so they widen — the value carries dd across
# the return.  The shared ORIGINAL callee is never edited (only the per-integral
# clone's return type is widened, Item 7 §3).
# --------------------------------------------------------------------------- #

RULEC_CHAIN_H = """\
#pragma once
namespace app {

template<class T>
T callee(T a, T b) {
    T p, r;
    p = a + T(1);
    r = p - b;
    return r;
}

template<class T>
T caller(T x) {
    T m, n, diff;
    m = callee<T>(x, x);
    n = callee<T>(x, x);
    diff = m - n;
    return diff;
}

template<class T>
T entry(T x) {
    T e = caller<T>(x);
    return e + T(4);
}

}  // namespace app
"""


@pytest.fixture
def rulec_chain_tree(tmp_path) -> Path:
    (tmp_path / "app.h").write_text(RULEC_CHAIN_H)
    return tmp_path


@pytest.fixture
def rulec_chain_graph(rulec_chain_tree):
    from agents.patcher.call_graph import build_call_graph
    fanout.clear_graph_cache()
    return build_call_graph("entry", rulec_chain_tree,
                            tu_file=rulec_chain_tree / "app.h")


@requires_libclang
def test_rulec_e2e_widens_callee_return_and_caller_locals(rulec_chain_tree,
                                                          rulec_chain_graph):
    g = rulec_chain_graph
    r_line = _line(g, "callee", "r = p - b;")          # callee's carried write (seed)
    diff_line = _line(g, "caller", "diff = m - n;")    # caller's cancellation (seed)
    sig_line = _line(g, "callee", "T callee(T a, T b) {")
    m_decl = _line(g, "caller", "T m, n, diff;")

    res = chain_promote(manifest=ChainManifest(
        chain_id="cascade_B10_rc", integral="B10", entry_point="entry",
        lines=[("app.h", r_line, r_line), ("app.h", diff_line, diff_line)]),
        graph=g, tree_root=rulec_chain_tree, scalar_type=DD, two_limb=True,
        shim_include=None)

    # no terminal refusal; the chain promotes and the interior gate stays silent (the
    # widened callee return is a designed exit — clause ii — not a truncating seam).
    assert res.chain_carrier_unwidenable is False
    assert res.chain_carrier_external is False
    assert res.chain_closure_escapes is False
    assert res.promotion_applied is True
    assert res.write_truncation is False

    specs = _manifest_specs(rulec_chain_tree)
    # callee variant carries the return-type widen (rule c); caller variant carries the
    # receiving-local decl widen (rule a re-fired on m/n).
    callee_variant = specs["callee_caller_B10"]
    assert callee_variant.return_widen is not None
    assert callee_variant.return_widen.return_line == sig_line
    assert callee_variant.return_widen.orig_type == "T"
    assert callee_variant.return_widen.dd_type == DD
    caller_variant = specs["caller_B10"]
    caller_decl_lines = {c.decl_line for c in caller_variant.closure_decls}
    assert m_decl in caller_decl_lines

    # rendered text: the callee CLONE returns dd; the shared ORIGINAL callee is untouched.
    txt = (rulec_chain_tree / "app.h").read_text()
    before_block, after_block = txt.split(fanout._BLOCK_BEGIN)
    assert "T callee(T a, T b)" in before_block         # original signature untouched
    assert f"{DD} callee_caller_B10(" in after_block    # clone return widened
