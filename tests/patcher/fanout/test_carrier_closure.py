"""Value-closure analysis (CLOSURE_SCOPED_CHAINS_DESIGN.md §2; Subtask 1a).

Unit tests for :func:`agents.patcher.chain_promote.compute_value_closure`, which
carries two views of a chain's closure on one :class:`CarrierClosure`:

* the **Fix-A compat subset** (``widenable`` / ``unwidenable_reasons`` /
  ``external_reasons`` / ``carrier_names``) — the Blocker-A strict-carrier test
  (a variable declared OUTSIDE the chain line set, written by an interior chain line
  and read by *another chain line*).  Every current consumer reads only this, so the
  tests in the first two sections pin it byte-for-byte (the Fix-A regression guard);

* the **enlarged value closure** (``closure_names`` / ``closure_decl_widens`` /
  ``designed_exits`` / ``escape_reasons`` / ``return_widens``) — the least fixed
  point of rule (a) [read on ANY frame line] and rule (b) [forward flow to a local /
  out-param / return / kernel output] and rule (c) [cross-frame return propagation],
  with ``chain_closure_escapes`` refusals at the frontier.  Rule (c) (Subtask 2b)
  carries dd across ``Li2omx2``'s / ``ddilog``'s returns into B10's cancellation, so
  ``return_widens`` names both callees and dilog4/dilog5 join the closure.

The ``entry -> mid -> inner`` synthetic (``CARRIER_H``) exercises the compat subset;
``top -> mid2 -> leaf`` (``CLOSURE_H``) exercises rules (a)/(b), escapes, designed
exits, and refusal precedence.  Line numbers are asserted against the header text so
a future edit that shifts them fails loudly.  The real B10/B13/B14 chains are pinned
against the committed ``runs/qcdloop_headers_full`` tree.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.patcher import fanout
from agents.patcher.chain_promote import (
    CarrierClosure, ChainManifest, compute_value_closure,
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
    return compute_value_closure(manifest=man, graph=graph, scalar_type="Ext", **kw)


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
    cc = compute_value_closure(manifest=man, graph=graph, scalar_type="Ext",
                                 complex_type="ExtC", complex_tokens=frozenset({"CT"}))
    assert cc.widenable == [(str(carrier_tree / "app.h"), L_DECL_CARRY, "carry", "ExtC")]


@requires_libclang
def test_region_not_in_any_function_raises(carrier_graph):
    with pytest.raises(FanoutError):
        _closure(carrier_graph, [999])


def test_empty_chain_returns_empty_closure(carrier_graph=None):
    # No graph access needed: an empty line set yields an empty closure.
    man = ChainManifest(chain_id="c", integral="B", entry_point="entry", lines=[])
    cc = compute_value_closure(manifest=man, graph=None, scalar_type="Ext")
    assert cc.widenable == [] and cc.unwidenable_reasons == [] and cc.external_reasons == []


# --------------------------------------------------------------------------- #
# enlarged value closure — synthetic rules-(a)/(b) fixture (termination +
# refusal precedence).  top -> mid2 -> leaf; `ext` is external (∉ F), `sink`
# is a module global.  Only `c` (:16) is a chain line.
# --------------------------------------------------------------------------- #

CLOSURE_H = """\
#pragma once
namespace app {

double sink;

template<class T>
T ext(T x) {
    return x + T(1);
}

template<class T>
T leaf(T a, T& outp) {
    T c;
    T d;
    T e;
    c = a + T(1);
    d = ext(c);
    e = c + T(2);
    outp = e;
    sink = c;
    return c;
}

template<class T>
T mid2(T x) {
    T r = x;
    return leaf<T>(x, r);
}

template<class T>
T top(T x) {
    return mid2<T>(x);
}

}  // namespace app
"""

CL_DECL_C, CL_DECL_D, CL_DECL_E = 13, 14, 15   # `T c; T d; T e;`
CL_C_W = 16                                    # `c = a + T(1);`  (the only chain line)
CL_ESCAPE_EXT = 17                             # `d = ext(c);`    (b→local escapes at ext)
CL_E_JOINS = 18                                # `e = c + T(2);`  (b→local joins)
CL_OUT_PARAM = 19                              # `outp = e;`      (b→out-param exit)
CL_GLOBAL_W = 20                               # `sink = c;`      (b→global escape)
CL_RETURN = 21                                 # `return c;`      (b→return exit)


@pytest.fixture
def closure_tree(tmp_path) -> Path:
    (tmp_path / "app.h").write_text(CLOSURE_H)
    return tmp_path


@pytest.fixture
def closure_graph(closure_tree):
    from agents.patcher.call_graph import build_call_graph
    fanout.clear_graph_cache()
    return build_call_graph("top", closure_tree, tu_file=closure_tree / "app.h")


def _closure_h(graph, lines, **kw):
    man = ChainManifest(chain_id="c", integral="B", entry_point="top",
                        lines=[("app.h", ln, ln) for ln in lines])
    return compute_value_closure(manifest=man, graph=graph, scalar_type="Ext", **kw)


def test_closure_header_line_numbers_match_constants():
    lines = CLOSURE_H.split("\n")
    assert lines[CL_DECL_C - 1].strip() == "T c;"
    assert lines[CL_DECL_D - 1].strip() == "T d;"
    assert lines[CL_DECL_E - 1].strip() == "T e;"
    assert lines[CL_C_W - 1].strip() == "c = a + T(1);"
    assert lines[CL_ESCAPE_EXT - 1].strip() == "d = ext(c);"
    assert lines[CL_E_JOINS - 1].strip() == "e = c + T(2);"
    assert lines[CL_OUT_PARAM - 1].strip() == "outp = e;"
    assert lines[CL_GLOBAL_W - 1].strip() == "sink = c;"
    assert lines[CL_RETURN - 1].strip() == "return c;"


@requires_libclang
def test_closure_reaches_fixed_point_forward_flow(closure_graph):
    # Rule (a) seeds `c` (written on the chain line :16, read on frame lines :17-:21,
    # decl :13).  Rule (b) forwards to the local `e` (:18) whose decl :15 then widens
    # by rule (a).  The iteration converges to exactly {c, e}; the compat view is empty
    # (no strict Fix-A carrier), proving the enlarged closure is a superset that does
    # not leak into consumers.
    cc = _closure_h(closure_graph, [CL_C_W])
    assert cc.closure_names == {"c", "e"}
    assert {(ln, n) for _f, ln, n, _t in cc.closure_widenable} == {
        (CL_DECL_C, "c"), (CL_DECL_E, "e")}
    assert cc.carrier_names == set()
    # designed exits (A.2 5-tuple: file, line, kind, carried_values, detail): the
    # out-param write (:19, carries {outp}) and the return (:21, carries {c}).
    exits = {(ln, kind, detail) for _f, ln, kind, _cv, detail in cc.designed_exits}
    assert exits == {(CL_OUT_PARAM, "out_param", "outp"), (CL_RETURN, "return", "c")}
    carried = {(ln, kind): cv for _f, ln, kind, cv, _d in cc.designed_exits}
    assert carried[(CL_OUT_PARAM, "out_param")] == frozenset({"outp"})
    assert carried[(CL_RETURN, "return")] == frozenset({"c"})


@requires_libclang
def test_closure_refusal_precedence_blocks_destination_not_source(closure_graph):
    # `d = ext(c)` (:17) reads carried `c` and passes it into ext ∉ F.  `c` is a SOURCE
    # escape (diagnostic only — it still widens by rule (a)).  `d`'s producing value `c`
    # comes from `c = a + T(1)` (an input, not a cancellation of carried operands) → `d`
    # is a NON-benign extract → a DESTINATION escape that never joins.  `sink = c` (:20)
    # is a shared-global write → a second destination escape.  Neither `d` nor `sink`
    # widens; `c` (the rule-(a) source) does.
    cc = _closure_h(closure_graph, [CL_C_W])
    assert "d" not in cc.closure_names
    assert "sink" not in cc.closure_names
    # A.1 split: source escapes are purely diagnostic; destination escapes block.
    src = {n for n, _r in cc.source_escapes}
    dst = {n for n, _r in cc.destination_escapes}
    assert src == {"c"}
    assert dst == {"d", "sink"}
    # compat union retained for legacy readers
    assert {n for n, _r in cc.escape_reasons} == {"c", "d", "sink"}
    src_reasons = {n: r for n, r in cc.source_escapes}
    dst_reasons = {n: r for n, r in cc.destination_escapes}
    assert "ext" in src_reasons["c"]             # callee-∉-F source escape
    assert "global" in dst_reasons["sink"] or "shared" in dst_reasons["sink"]
    # the rule-(a) source survives the escape of its rule-(b) destination
    assert "c" in cc.closure_names


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
    return compute_value_closure(manifest=man, graph=graph,
                                   scalar_type="Kokkos::Experimental::DoubleDouble")


@requires_libclang
@requires_qcdloop_full
def test_real_b10_finds_YSA_carriers_on_ddilog(qcdloop_full_graph):
    # The blocker's worked example: ddilog's `TMass Y, S, A;` at :157 — Y and A are
    # strict carriers (Y write :174/read :199; A write :177/read :212), so the whole
    # multi-declarator (Y, S, A) is widened to DoubleDouble at :157.
    cc = _real_closure(qcdloop_full_graph, "B10")
    assert _widen_names(cc) == {"Y", "S", "A"}
    decl_lines = {ln for _f, ln, _n, _t in cc.widenable}
    files = {Path(f).name for f, _l, _n, _t in cc.widenable}
    assert decl_lines == {157}
    assert files == {"kokkosUtils.h"}
    assert all(t == "Kokkos::Experimental::DoubleDouble" for _f, _l, _n, t in cc.widenable)


@requires_libclang
@requires_qcdloop_full
def test_real_b13_has_no_compat_carriers(qcdloop_full_graph):
    # COMPAT (Fix-A) view: B13's ga34*/ga43* are read only on the NON-chain extracts,
    # never on another chain line → strict condition 2 fails → no Fix-A carrier.  The
    # enlarged rule (a) DOES capture them (see test_real_b13_closure_widens_ga34).
    cc = _real_closure(qcdloop_full_graph, "B13")
    assert cc.widenable == []
    assert cc.unwidenable_reasons == []
    assert cc.external_reasons == []


@requires_libclang
@requires_qcdloop_full
def test_real_b14_has_no_compat_carriers(qcdloop_full_graph):
    # COMPAT (Fix-A) view: B14 `fac` is read only at the NON-chain output stores
    # (res(i,1)/res(i,0)) → strict condition 2 fails → not a Fix-A carrier.  The
    # enlarged rule (a) DOES capture it (see test_real_b14_closure_widens_fac).
    cc = _real_closure(qcdloop_full_graph, "B14")
    assert cc.widenable == []
    assert cc.unwidenable_reasons == []
    assert cc.external_reasons == []


# --------------------------------------------------------------------------- #
# enlarged value closure — rules (a)+(b) on the real B10 / B13 / B14 chains
# --------------------------------------------------------------------------- #

@requires_libclang
@requires_qcdloop_full
def test_real_b13_closure_widens_ga34(qcdloop_full_graph):
    # Rule (a) generalisation: ga34*/ga43* are written on chain lines and read on the
    # (non-chain) extract lines, so their decls at :282/:283 enter the ENLARGED closure
    # — empty under Fix A (test_real_b13_has_no_compat_carriers).  Widening the leading
    # type token widens every same-type sibling, so the :282 declarator `root` rides
    # along (an over-widened same-type sibling never truncates, §2).
    cc = _real_closure(qcdloop_full_graph, "B13")
    assert {"ga34m", "ga34pm1", "ga43m", "ga43pm1"} <= cc.closure_names
    decl_lines = {ln for _f, ln, _n, _t in cc.closure_widenable}
    assert decl_lines == {282, 283}
    assert "root" in cc.closure_names          # :282 multi-declarator sibling
    # compat view stays empty — the enlarged closure does not leak into what consumers see
    assert cc.carrier_names == set()


@requires_libclang
@requires_qcdloop_full
def test_real_b13_closure_escapes_at_ql_real(qcdloop_full_graph):
    # Rule (b) → local at the frontier: `x34* = ql::Real(ga34*)` reads a carried ga34*
    # and passes it into ql::Real, a callee NOT in the chain function set → ga34* is a
    # SOURCE escape (diagnostic; ga34* still widens by rule (a)).  Its producing chain
    # line `ga34m = TOutput(...) - root` is a binary subtraction of carried/widened
    # operands, so the extract projection is provably BENIGN (§3.2 iii) → the x34*
    # destinations are designed exits, NOT destination escapes.
    cc = _real_closure(qcdloop_full_graph, "B13")
    src_names = {n for n, _r in cc.source_escapes}
    assert "ga34m" in src_names
    assert any("ql::Real" in r for _n, r in cc.source_escapes)
    # all four extracts benign → no destination escape, so the terminal never fires
    assert cc.destination_escapes == []
    assert cc.blocking_escapes == []
    # the source ga34* still widen (source escape does not block)
    assert "ga34m" in cc.closure_names and "ga34pm1" in cc.closure_names


@requires_libclang
@requires_qcdloop_full
def test_real_b14_closure_widens_fac_and_marks_output_store(qcdloop_full_graph):
    # B14 clean within-frame case: rule (a) widens `fac` (decl :396, written :401, read
    # at the output stores); rule (b) records the res(i,k) stores at :404/:405 as
    # kernel-output designed exits (the chain's designed landing at caller precision).
    cc = _real_closure(qcdloop_full_graph, "B14")
    assert "fac" in cc.closure_names
    assert {ln for _f, ln, _n, _t in cc.closure_widenable} == {396}
    exits = {(ln, kind) for _f, ln, kind, _cv, _d in cc.designed_exits}
    assert (404, "kernel_output") in exits
    assert (405, "kernel_output") in exits
    assert cc.carrier_names == set()           # compat view stays empty


@requires_libclang
@requires_qcdloop_full
def test_real_b10_closure_extends_across_li2omx2_return(qcdloop_full_graph):
    # Rule (c) — the Subtask-2b headline.  Rule (a) widens ddilog's {Y,S,A} @157 and
    # Li2omx2's `prod, Li2omx2;` @691; rule (b) reaches Li2omx2's `return Li2omx2` @707
    # and ddilog's `return -(...+A)` @212.  Rule (c) then fires on BOTH chain-internal
    # return edges: ddilog -> Li2omx2 (consumed at :698/:704) and Li2omx2 -> B10
    # (consumed at B1m.h:{235,236,237}).  The callee return types widen (recorded in
    # return_widens) and the caller's receiving locals dilog3/dilog4/dilog5 re-enter
    # rule (a) in B10 and widen — so the :241 `res(i,0) = dilog4 - dilog5 ...`
    # cancellation now executes at dd and its store to res(i,k) is the designed exit.
    cc = _real_closure(qcdloop_full_graph, "B10")
    assert {"Y", "S", "A", "prod", "Li2omx2"} <= cc.closure_names
    # rule (c): the cancellation operands NOW join the closure and widen (was the
    # Subtask-1b falsifier; inverted here — the whole point of rule (c)).
    assert "dilog4" in cc.closure_names
    assert "dilog5" in cc.closure_names
    b10_decl_lines = {ln for f, ln, _n, _t in cc.closure_widenable
                      if Path(f).name == "B1m.h"}
    assert {236, 237} <= b10_decl_lines
    # rule (c) records a return-type widen for BOTH chain-internal callees, naming the
    # ORIGINAL function (attach binds it to every per-caller-path variant at emission).
    rw_by_fn = {rw.function_name: rw for rw in cc.return_widens}
    assert "Li2omx2" in rw_by_fn and "ddilog" in rw_by_fn
    assert (rw_by_fn["Li2omx2"].return_line, rw_by_fn["Li2omx2"].orig_type) == (688, "TOutput")
    assert (rw_by_fn["ddilog"].return_line, rw_by_fn["ddilog"].orig_type) == (149, "TMass")
    assert all(rw.dd_type == "Kokkos::Experimental::DoubleDouble" for rw in cc.return_widens)
    # the chain's designed exit is now B10's res(i,0) cancellation store, NOT Li2omx2's
    # return (which carries dd across, no truncation — clause (ii)).
    exit_kinds = {(ln, kind) for _f, ln, kind, _cv, _d in cc.designed_exits}
    assert (241, "kernel_output") in exit_kinds
    # Fix-A compat subset unchanged (regression guard): rule (c) widens RETURNS and
    # caller locals but leaves the strict-carrier compat view byte-identical.
    assert cc.carrier_names == {"Y", "S", "A"}
    # Li2omx2's internal decl-init locals lnarg/lnomarg (@702/703) are read by the dd
    # cancellation @704 — body-owned chain-line carriers threaded into closure_names so
    # the boundary transform keeps them wide instead of demoting the decl-init landing.
    assert {"lnarg", "lnomarg"} <= cc.closure_body_names
    assert {"lnarg", "lnomarg"} <= cc.closure_names


# --------------------------------------------------------------------------- #
# rule (c) — cross-frame return propagation (synthetic, CLOSURE_SCOPED §2.3)
#
# entry -> caller -> callee: callee returns a carried value consumed by caller in a
# cancellation; a further hop caller -> deepr climbs the DAG.  refuse: callee2 calls
# an ext ∉ F.  Chain lines are the writes that seed each frame.
# --------------------------------------------------------------------------- #

RULEC_H = """\
#pragma once
namespace app {

template<class T>
T deepr(T u) {
    T w, z;
    w = u + T(1);
    z = w - T(2);
    return z;
}

template<class T>
T callee(T a, T b) {
    T p, q, r;
    p = deepr<T>(a);
    q = a - b;
    r = p + q;
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
    return caller<T>(x);
}

}  // namespace app
"""

# line map (1-based against RULEC_H)
RC_DEEPR_SIG = 5          # `T deepr(T u) {`
RC_DEEPR_DECL = 6         # `T w, z;`
RC_DEEPR_W = 7            # `w = u + T(1);`
RC_DEEPR_Z = 8            # `z = w - T(2);`
RC_DEEPR_RET = 9          # `return z;`
RC_CALLEE_SIG = 13        # `T callee(T a, T b) {`
RC_CALLEE_DECL = 14       # `T p, q, r;`
RC_CALLEE_P = 15          # `p = deepr<T>(a);`
RC_CALLEE_Q = 16          # `q = a - b;`
RC_CALLEE_R = 17          # `r = p + q;`
RC_CALLEE_RET = 18        # `return r;`
RC_CALLER_SIG = 22        # `T caller(T x) {`
RC_CALLER_DECL = 23       # `T m, n, diff;`
RC_CALLER_M = 24          # `m = callee<T>(x, x);`
RC_CALLER_N = 25          # `n = callee<T>(x, x);`
RC_CALLER_DIFF = 26       # `diff = m - n;`


@pytest.fixture
def rulec_tree(tmp_path) -> Path:
    (tmp_path / "app.h").write_text(RULEC_H)
    return tmp_path


@pytest.fixture
def rulec_graph(rulec_tree):
    from agents.patcher.call_graph import build_call_graph
    fanout.clear_graph_cache()
    return build_call_graph("entry", rulec_tree, tu_file=rulec_tree / "app.h")


def _rulec_closure(graph, lines, **kw):
    man = ChainManifest(chain_id="rc", integral="B", entry_point="entry",
                        lines=[("app.h", ln, ln) for ln in lines])
    return compute_value_closure(manifest=man, graph=graph, scalar_type="Ext", **kw)


def test_rulec_header_line_numbers_match_constants():
    lines = RULEC_H.split("\n")
    assert lines[RC_DEEPR_SIG - 1].strip() == "T deepr(T u) {"
    assert lines[RC_DEEPR_Z - 1].strip() == "z = w - T(2);"
    assert lines[RC_DEEPR_RET - 1].strip() == "return z;"
    assert lines[RC_CALLEE_SIG - 1].strip() == "T callee(T a, T b) {"
    assert lines[RC_CALLEE_R - 1].strip() == "r = p + q;"
    assert lines[RC_CALLEE_RET - 1].strip() == "return r;"
    assert lines[RC_CALLER_DECL - 1].strip() == "T m, n, diff;"
    assert lines[RC_CALLER_M - 1].strip() == "m = callee<T>(x, x);"
    assert lines[RC_CALLER_DIFF - 1].strip() == "diff = m - n;"


@requires_libclang
def test_rulec_fires_across_one_return_edge(rulec_graph):
    # callee returns carried `r`; caller consumes it into m/n and cancels at :25.
    # Chain seeds: callee's `r` write (:17) and caller's `diff` write (:25).  Rule (c)
    # widens callee's return type and re-seeds m/n in caller -> they widen by rule (a).
    cc = _rulec_closure(rulec_graph, [RC_CALLEE_R, RC_CALLER_DIFF])
    assert "r" in cc.closure_names
    assert "m" in cc.closure_names and "n" in cc.closure_names
    rw = {r.function_name: r for r in cc.return_widens}
    assert "callee" in rw
    assert (rw["callee"].return_line, rw["callee"].orig_type) == (RC_CALLEE_SIG, "T")
    assert rw["callee"].dd_type == "Ext"
    # caller receiving-local decls widen (m/n decl on the shared :22 line)
    decl_lines = {ln for _f, ln, _n, _t in cc.closure_widenable}
    assert RC_CALLER_DECL in decl_lines


@requires_libclang
def test_rulec_climbs_the_dag_two_hops(rulec_graph):
    # deepr -> callee -> caller: seed all three frames' carried writes.  Rule (c) fires
    # on BOTH internal return edges (deepr->callee at :15, callee->caller at :23/:24),
    # so both deepr and callee return types widen and p (in callee) + m/n (in caller)
    # re-seed and widen — the climb terminates at caller (whose return feeds entry).
    cc = _rulec_closure(rulec_graph, [RC_DEEPR_Z, RC_CALLEE_R, RC_CALLER_DIFF])
    rw = {r.function_name for r in cc.return_widens}
    assert {"deepr", "callee"} <= rw
    assert "p" in cc.closure_names        # callee's receiving local from deepr
    assert {"m", "n"} <= cc.closure_names # caller's receiving locals from callee


@requires_libclang
def test_rulec_terminates_no_runaway(rulec_graph):
    # Termination property (§2.5): the fixed point converges without hitting MAX_ROUNDS.
    # A converged closure is stable — recomputing yields identical closure_names and
    # return_widens (idempotent, monotone lattice).
    a = _rulec_closure(rulec_graph, [RC_DEEPR_Z, RC_CALLEE_R, RC_CALLER_DIFF])
    b = _rulec_closure(rulec_graph, [RC_DEEPR_Z, RC_CALLEE_R, RC_CALLER_DIFF])
    assert a.closure_names == b.closure_names
    assert {r.function_name for r in a.return_widens} == {
        r.function_name for r in b.return_widens}


@requires_libclang
def test_rulec_does_not_fire_without_internal_consumer(rulec_graph):
    # deepr's return is consumed by callee, but if the chain seeds ONLY deepr (no caller
    # frame in F consuming callee's onward return), rule (c) still fires deepr->callee
    # because callee IS in F once its line is seeded — so seed only deepr's frame and
    # NOT callee's: callee is not a chain frame, deepr's return has no in-F consumer, so
    # no return widen is recorded (the return is a plain designed-exit / gate-checked).
    cc = _rulec_closure(rulec_graph, [RC_DEEPR_Z])
    assert cc.return_widens == []
    # deepr's own return is marked (rule b) but not widened (no in-F caller); with rule
    # (c) not firing it stays a plain gate-checked return, never return_widened.
    kinds = {kind for _f, _l, kind, _cv, _d in cc.designed_exits}
    assert "return_widened" not in kinds


@requires_libclang
def test_rulec_callee_not_in_F_stays_plain_return(closure_graph):
    # §2.4 asymmetry: a call to a function NOT in the chain function set is an ESCAPE,
    # not a rule-(c) edge — v1 does not widen a foreign signature.  Here the chain is
    # only `leaf`'s `c` (:16); `leaf` returns `c` (:21) but no OTHER chain frame consumes
    # that return internally (mid2 is not seeded), so rule (c) records nothing and the
    # return stays a plain (gate-checked) exit.  `d = ext(c)` sends c into ext ∉ F — a
    # source escape, never a rule-(c) return widen on ext.
    cc = _closure_h(closure_graph, [CL_C_W])
    assert cc.return_widens == []
    assert "ext" not in {r.function_name for r in cc.return_widens}
    kinds = {kind for _f, _l, kind, _cv, _d in cc.designed_exits}
    assert "return_widened" not in kinds       # leaf's return not widened (no in-F caller)


def test_decl_init_writes_recovers_chain_line_carriers():
    # _decl_init_writes recovers a decl-init LHS (excluded by region_writes_from_source)
    # so a chain-line decl-init carrier (Li2omx2's lnarg) is recognized.  A bare decl and
    # a plain assign yield nothing (region_writes_from_source owns the latter).
    from agents.patcher.chain_promote import _decl_init_writes
    assert _decl_init_writes("const TOutput lnarg = TOutput(a - b);") == {"lnarg"}
    assert _decl_init_writes("const T dilog4 = ql::Li2omx2<T,U,V>(a, b);") == {"dilog4"}
    assert _decl_init_writes("TOutput prod, Li2omx2;") == set()      # bare decl, no init
    assert _decl_init_writes("Li2omx2 = -TOutput(x) + lnarg;") == set()  # plain assign
    assert _decl_init_writes("if (a < b) c = d;") == set()           # not a decl


@requires_libclang
def test_rulec_attach_binds_frame_record_to_variant(rulec_graph):
    # STOP #5 wiring end-to-end: the frame-level ReturnWiden that rule (c) records
    # (function_name = ORIGINAL name, return_line = signature line) binds to a per-path
    # VariantSpec via _attach_return_widens (orig_name + line containment).
    from agents.patcher.chain_promote import _attach_return_widens
    from agents.patcher.fanout import VariantSpec
    cc = _rulec_closure(rulec_graph, [RC_CALLEE_R, RC_CALLER_DIFF])
    rw = [r for r in cc.return_widens if r.function_name == "callee"]
    assert rw
    spec = VariantSpec(variant_name="callee_caller_B", orig_name="callee",
                       file="app.h", orig_start=12, orig_end=19)
    _attach_return_widens(cc.return_widens, {"app.h": {"callee_caller_B": spec}})
    assert spec.return_widen is not None
    assert spec.return_widen.function_name == "callee"
    assert spec.return_widen.return_line == RC_CALLEE_SIG
