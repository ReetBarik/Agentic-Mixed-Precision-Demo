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
  out-param / return / kernel output], with ``chain_closure_escapes`` refusals at the
  frontier.  Rule (c) (cross-frame return propagation) is Subtask 2a/2b, so
  ``return_widens`` is empty and B10 stops at ``Li2omx2``'s return here.

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
def test_real_b10_closure_extends_to_li2omx2_but_stops_at_return(qcdloop_full_graph):
    # Rule (a) also widens Li2omx2's `prod, Li2omx2;` (decl :691; Li2omx2 written :704,
    # read at the :707 return), on top of ddilog's Fix-A {Y,S,A}.  Rule (b) marks the
    # :707 `return Li2omx2` as a designed-exit candidate but — with rule (c) OUT of
    # scope — does NOT cross into B1m.h, so dilog4/dilog5 stay double and B10's headline
    # cancellation is still unrecovered (the Subtask 2a/2b job).
    cc = _real_closure(qcdloop_full_graph, "B10")
    assert {"Y", "S", "A", "prod", "Li2omx2"} <= cc.closure_names
    ret_exits = {ln for _f, ln, kind, _cv, _d in cc.designed_exits if kind == "return"}
    assert 707 in ret_exits
    # no rule (c): the cancellation operands never join under rules (a),(b) alone
    assert "dilog4" not in cc.closure_names
    assert "dilog5" not in cc.closure_names
    assert cc.return_widens == []       # Subtask 2a: list (was frozenset), still empty
    # Fix-A compat subset unchanged (regression guard)
    assert cc.carrier_names == {"Y", "S", "A"}
