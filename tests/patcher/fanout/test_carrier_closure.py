"""Blocker A, Subtask 1 — carrier-closure analysis (BLOCKER_A_CARRIER_DESIGN.md §2/§4).

Unit tests for :func:`agents.patcher.chain_promote.compute_carrier_closure`: the
analysis pass that, given a chain's line set, classifies its carriers as widenable
/ unwidenable / external.  A **carrier** is a variable declared OUTSIDE the chain's
line set, written by one interior chain line and read by another — the value it
carries crosses a chain-line boundary at caller precision, so the interior write
truncates the widened (dd) value and the 2d-B gate spuriously rejects the patch.

The synthetic call graph is ``entry -> mid -> inner``; ``inner`` (the deepest,
interior function) holds the carriers, ``mid``'s region is the chain's outermost
(min-depth) exit boundary.  Line numbers are asserted against the header text so a
future edit that shifts them fails loudly rather than silently mis-locating a
carrier.  The real B10/B13/B14 chains are pinned against the committed
``runs/qcdloop_headers_full`` tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.patcher import fanout
from agents.patcher.chain_promote import (
    CarrierClosure, ChainManifest, compute_carrier_closure,
)
from agents.patcher.fanout import FanoutError
from tests.patcher.fanout.conftest import requires_libclang, requires_qcdloop_full

# entry -> mid -> inner.  Every carrier-definition case has a home:
#   carry   — strict carrier (write :12, read :13, decl :10 outside the chain)
#   m,k,s   — 'm' a conditional carrier (write :14 under a guard, read :15); its
#             multi-declarator siblings k,s share the decl line :9
#   solo    — written+read only on the same line :17 (not a carrier)
#   p       — a function parameter (unwidenable)
#   g_ext   — a global (external)
#   g_exit  — a global that is ALSO the outermost region's write target (condition 4
#             silently excludes it — NOT reported external)
#   inl     — declared INSIDE the chain line set (:22) → out of scope
CARRIER_H = """\
#pragma once
namespace app {

double g_ext;
double g_exit;

template<class T>
T inner(T p, T& res) {
    T k, m, s;
    T carry;
    T solo;
    carry = p + T(1);
    k = carry + T(2);
    if (p > T(0)) m = k;
    s = m + T(1);
    p = p + T(1);
    solo = solo + T(1);
    g_ext = k;
    res = g_ext + s;
    g_exit = s;
    res = res + g_exit;
    T inl = k;
    inl = inl + T(1);
    res = res + inl;
    return res;
}

template<class T>
T mid(T x) {
    T c = inner<T>(x, x);
    g_exit = c;
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

# Chain line numbers, asserted against CARRIER_H below so a header edit fails loudly.
L_CARRY_W, L_CARRY_R = 12, 13         # carry: interior write, interior read
L_M_W, L_M_R = 14, 15                 # m: conditional write, read (siblings k,s @ :9)
L_SOLO = 17                           # solo: same-line write+read only
L_P_R, L_P_W = 12, 16                 # p: read (in carry's rhs) + param write
L_GEXT_W, L_GEXT_R = 18, 19           # g_ext: interior write + read (global)
L_GEXIT_W, L_GEXIT_R = 20, 21         # g_exit: interior write + read
L_INL_DECL, L_INL_W, L_INL_R = 22, 23, 24   # inl: decl-in-chain, write, read
L_DECL_SIBS = 9                       # `T k, m, s;`
L_DECL_CARRY = 10                     # `T carry;`
L_OUTER = 32                          # mid's region `T d = c - T(3);` (min depth)
L_OUTER_GEXIT = 31                    # mid writes g_exit (outermost write target)


@pytest.fixture
def carrier_tree(tmp_path) -> Path:
    (tmp_path / "app.h").write_text(CARRIER_H)
    return tmp_path


@pytest.fixture
def carrier_graph(carrier_tree):
    from agents.patcher.call_graph import build_call_graph
    fanout.clear_graph_cache()
    return build_call_graph("entry", carrier_tree, tu_file=carrier_tree / "app.h")


def _closure(graph, lines, **kw):
    man = ChainManifest(chain_id="c", integral="B", entry_point="entry",
                        lines=[("app.h", ln, ln) for ln in lines])
    return compute_carrier_closure(manifest=man, graph=graph, scalar_type="Ext", **kw)


def _widen_names(cc: CarrierClosure) -> set[str]:
    return {name for _f, _l, name, _t in cc.widenable}


# --------------------------------------------------------------------------- #
# header line-number guard — fail loudly if CARRIER_H is edited out of sync
# --------------------------------------------------------------------------- #

def test_header_line_numbers_match_constants():
    lines = CARRIER_H.split("\n")
    assert lines[L_CARRY_W - 1].strip() == "carry = p + T(1);"
    assert lines[L_CARRY_R - 1].strip() == "k = carry + T(2);"
    assert lines[L_M_W - 1].strip() == "if (p > T(0)) m = k;"
    assert lines[L_M_R - 1].strip() == "s = m + T(1);"
    assert lines[L_SOLO - 1].strip() == "solo = solo + T(1);"
    assert lines[L_P_W - 1].strip() == "p = p + T(1);"
    assert lines[L_GEXT_W - 1].strip() == "g_ext = k;"
    assert lines[L_GEXT_R - 1].strip() == "res = g_ext + s;"
    assert lines[L_GEXIT_W - 1].strip() == "g_exit = s;"
    assert lines[L_GEXIT_R - 1].strip() == "res = res + g_exit;"
    assert lines[L_INL_DECL - 1].strip() == "T inl = k;"
    assert lines[L_DECL_SIBS - 1].strip() == "T k, m, s;"
    assert lines[L_DECL_CARRY - 1].strip() == "T carry;"
    assert lines[L_OUTER - 1].strip() == "T d = c - T(3);"
    assert lines[L_OUTER_GEXIT - 1].strip() == "g_exit = c;"


# --------------------------------------------------------------------------- #
# carrier definition — one test per §2 case
# --------------------------------------------------------------------------- #

@requires_libclang
def test_strict_carrier_is_widenable(carrier_graph):
    # interior write (:12) + interior read on a DIFFERENT line (:13) → strict carrier;
    # its decl (:10) lies outside the chain line set → widenable.
    cc = _closure(carrier_graph, [L_CARRY_W, L_CARRY_R, L_OUTER])
    assert _widen_names(cc) == {"carry"}
    (file, decl_line, name, dd_type), = cc.widenable
    assert Path(file).name == "app.h"
    assert decl_line == L_DECL_CARRY
    assert name == "carry"
    assert dd_type == "Ext"
    assert cc.unwidenable_reasons == []
    assert cc.external_reasons == []


@requires_libclang
def test_same_line_write_and_read_only_is_not_a_carrier(carrier_graph):
    # `solo = solo + T(1);` writes AND reads solo on the SAME line, and no OTHER chain
    # line touches it → the value never crosses a chain-line boundary → not a carrier.
    cc = _closure(carrier_graph, [L_SOLO, L_OUTER])
    assert cc.widenable == []
    assert cc.unwidenable_reasons == []
    assert cc.external_reasons == []


@requires_libclang
def test_outermost_write_target_excluded_condition_4(carrier_graph):
    # g_exit is written by an interior line (:20) and read by another (:21) — but it is
    # ALSO a write target of the outermost region (mid @ :31).  Condition 4 removes it
    # from the closure SILENTLY (designed exit boundary, §5) — not reported external.
    cc = _closure(carrier_graph, [L_GEXIT_W, L_GEXIT_R, L_OUTER_GEXIT])
    assert cc.widenable == []
    assert cc.unwidenable_reasons == []
    assert cc.external_reasons == []          # condition 4 pre-empts external refusal


@requires_libclang
def test_decl_inside_chain_lines_is_out_of_scope(carrier_graph):
    # inl is declared at :22 (a chain line), written :23, read :24 — the body transform
    # already widens the decl, so condition 3 drops it (not returned in ANY bucket).
    cc = _closure(carrier_graph, [L_INL_DECL, L_INL_W, L_INL_R, L_OUTER])
    assert "inl" not in _widen_names(cc)
    assert cc.unwidenable_reasons == []
    assert cc.external_reasons == []


@requires_libclang
def test_parameter_carrier_is_unwidenable(carrier_graph):
    # p is written on an interior line (:16) and read on another (:12) — a strict
    # carrier whose decl is a function parameter → v1 refuses (chain_carrier_unwidenable).
    cc = _closure(carrier_graph, [L_P_R, L_P_W, L_OUTER])
    assert _widen_names(cc) == set()
    names = [n for n, _r in cc.unwidenable_reasons]
    assert names == ["p"]
    assert cc.external_reasons == []


@requires_libclang
def test_global_carrier_is_external(carrier_graph):
    # g_ext is written on an interior line (:18) and read on another (:19) — a strict
    # carrier whose decl is a module-global → v1 refuses (chain_carrier_external).
    cc = _closure(carrier_graph, [L_GEXT_W, L_GEXT_R, L_OUTER])
    assert _widen_names(cc) == set()
    names = [n for n, _r in cc.external_reasons]
    assert names == ["g_ext"]
    assert cc.unwidenable_reasons == []


@requires_libclang
def test_conditional_write_treated_as_carrier(carrier_graph):
    # `if (p > T(0)) m = k;` — a write under a guard is still a carrier-write
    # (conservative: over-widen rather than miss).  m is read at :15 → carrier.
    cc = _closure(carrier_graph, [L_M_W, L_M_R, L_OUTER])
    assert "m" in _widen_names(cc)


@requires_libclang
def test_multi_declarator_siblings_widened_alongside_carrier(carrier_graph):
    # m is the real carrier; its decl `T k, m, s;` (:9) is a multi-declarator, so
    # widening its type token widens k and s too (§2 conservative policy) — all three
    # returned, all anchored to the one decl line.
    cc = _closure(carrier_graph, [L_M_W, L_M_R, L_OUTER])
    assert _widen_names(cc) == {"k", "m", "s"}
    assert {decl_line for _f, decl_line, _n, _t in cc.widenable} == {L_DECL_SIBS}


# --------------------------------------------------------------------------- #
# structural
# --------------------------------------------------------------------------- #

@requires_libclang
def test_carrier_names_property(carrier_graph):
    cc = _closure(carrier_graph, [L_M_W, L_M_R, L_OUTER])
    assert cc.carrier_names == {"k", "m", "s"}


@requires_libclang
def test_complex_carrier_widens_to_complex_container(carrier_tree):
    # A carrier whose core declared type is a complex-bound token widens to the complex
    # container, not the scalar dd type (§7).
    from agents.patcher.call_graph import build_call_graph
    (carrier_tree / "app.h").write_text(CARRIER_H.replace("T carry;", "CT carry;"))
    fanout.clear_graph_cache()
    graph = build_call_graph("entry", carrier_tree, tu_file=carrier_tree / "app.h")
    man = ChainManifest(chain_id="c", integral="B", entry_point="entry",
                        lines=[("app.h", ln, ln) for ln in (L_CARRY_W, L_CARRY_R, L_OUTER)])
    cc = compute_carrier_closure(manifest=man, graph=graph, scalar_type="Ext",
                                 complex_type="ExtC", complex_tokens=frozenset({"CT"}))
    assert cc.widenable == [(str(carrier_tree / "app.h"), L_DECL_CARRY, "carry", "ExtC")]


@requires_libclang
def test_region_not_in_any_function_raises(carrier_graph):
    with pytest.raises(FanoutError):
        _closure(carrier_graph, [999])


def test_empty_chain_returns_empty_closure(carrier_graph=None):
    # No graph access needed: an empty line set yields an empty closure.
    man = ChainManifest(chain_id="c", integral="B", entry_point="entry", lines=[])
    cc = compute_carrier_closure(manifest=man, graph=None, scalar_type="Ext")
    assert cc.widenable == [] and cc.unwidenable_reasons == [] and cc.external_reasons == []


# --------------------------------------------------------------------------- #
# real B10 / B13 / B14 chains (committed qcdloop_headers_full tree)
# --------------------------------------------------------------------------- #

_REAL_CHAINS = {
    "B10": ["B1m.h:227", "B1m.h:240", "B1m.h:241",
            "kokkosUtils.h:174", "kokkosUtils.h:177", "kokkosUtils.h:199",
            "kokkosUtils.h:212", "kokkosUtils.h:702", "kokkosUtils.h:703",
            "kokkosUtils.h:704"],
    "B13": ["B2m.h:300", "B2m.h:301", "B2m.h:305", "B2m.h:306", "B2m.h:355",
            "B2m.h:533", "kokkosUtils.h:212", "kokkosUtils.h:702"],
    "B14": ["B2m.h:401", "B2m.h:578", "kokkosUtils.h:1208"],
}


def _real_closure(graph, integral):
    lines = [(c.split(":")[0], int(c.split(":")[1]), int(c.split(":")[1]))
             for c in _REAL_CHAINS[integral]]
    man = ChainManifest(chain_id=f"cascade_{integral}", integral=integral,
                        entry_point="BO", lines=lines)
    return compute_carrier_closure(manifest=man, graph=graph,
                                   scalar_type="quad::ddfun::ddouble")


@requires_libclang
@requires_qcdloop_full
def test_real_b10_finds_YSA_carriers_on_ddilog(qcdloop_full_graph):
    # The blocker's worked example: ddilog's `TMass Y, S, A;` at :157 — Y and A are
    # strict carriers (Y write :174/read :199; A write :177/read :212), so the whole
    # multi-declarator (Y, S, A) is widened to ddouble at :157.
    cc = _real_closure(qcdloop_full_graph, "B10")
    assert _widen_names(cc) == {"Y", "S", "A"}
    decl_lines = {ln for _f, ln, _n, _t in cc.widenable}
    files = {Path(f).name for f, _l, _n, _t in cc.widenable}
    assert decl_lines == {157}
    assert files == {"kokkosUtils.h"}
    assert all(t == "quad::ddfun::ddouble" for _f, _l, _n, t in cc.widenable)


@requires_libclang
@requires_qcdloop_full
def test_real_b13_has_no_carriers(qcdloop_full_graph):
    # B13's interior writes (ga34*/ga43*) are read only on their own write lines /
    # the output stores — none crosses a chain-line boundary → no carrier.
    cc = _real_closure(qcdloop_full_graph, "B13")
    assert cc.widenable == []
    assert cc.unwidenable_reasons == []
    assert cc.external_reasons == []


@requires_libclang
@requires_qcdloop_full
def test_real_b14_has_no_carriers(qcdloop_full_graph):
    # B14 `fac` is written on a chain line (:401) but read only at the NON-chain output
    # stores (res(i,1)/res(i,0)) → fails carrier condition 2 → not a carrier (§3).
    cc = _real_closure(qcdloop_full_graph, "B14")
    assert cc.widenable == []
    assert cc.unwidenable_reasons == []
    assert cc.external_reasons == []
