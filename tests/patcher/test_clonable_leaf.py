"""Unit tests for the ``clonable_leaf`` predicate (Subtask L2, deliverable a).

Covers :mod:`agents.patcher.clonable_leaf` — the pure, side-effect-free predicate
encoding ``LEAF_CALLEE_PROMOTION_DESIGN.md`` §1.2 clauses (1)–(4).  Every fixture is
a qcdloop-*representative* body shape (``Lnrat``/``ddilog``-like), but the predicate
is exercised structurally: it carries NO app-specific identifiers, so the names are
examples, never an enumerated allow-list.

The conservative-parser contract is the spine of these tests: a clonable leaf must
be accepted (no false negative that would needlessly refuse B10's ``Lnrat``), and
every uncertainty — missing body, unresolvable callee, same-name self-recursion,
inward-param widening — must be a refusal (a false positive is the STOP #K hard-fail
we must never ship).
"""

from __future__ import annotations

import pytest

from agents.patcher.call_graph import CallGraph, FuncDef
from agents.patcher.clonable_leaf import (
    ClonableLeafResult, clonable_leaf, scan_call_targets,
)
from agents.integrator_base.shallow_wrapper import (
    is_class1_synthesizable, surface_from_spelling,
)


# --------------------------------------------------------------------------- #
# fixtures — a vendored surface + qcdloop-representative primary bodies
# --------------------------------------------------------------------------- #

_SCALAR_OPS = frozenset("abs sqrt exp log log2 pow sin cos".split())
_COMPLEX_OPS = frozenset("abs conj sqrt exp log pow real imag".split())


@pytest.fixture
def surface():
    return surface_from_spelling(
        "Kokkos::Experimental::DoubleDouble", "Kokkos::Experimental::DoubleDoubleComplex",
        scalar_ops=_SCALAR_OPS, complex_ops=_COMPLEX_OPS)


# The Class-1 shallow wrappers Lnrat's body names (double primaries, src/kokkosMaths.h
# shapes).  A leaf's body classifies each of these via is_class1_synthesizable.
_CLASS1 = {
    "kAbs": "template<typename T> KOKKOS_INLINE_FUNCTION T kAbs(T const& x)"
            "{ return Kokkos::abs(x); }",
    "kLog": "template<typename T> KOKKOS_INLINE_FUNCTION T kLog(T const& x)"
            "{ return Kokkos::log(x); }",
    "Sign": "KOKKOS_INLINE_FUNCTION int Sign(double const& x)"
            "{ return (double(0) < x) - (x < double(0)); }",
    "Real": "KOKKOS_INLINE_FUNCTION double Real(Kokkos::complex<double> const& z)"
            "{ return z.real(); }",
}

# Class-2 / source-instantiated symbols (Constants accessors) — a value, not a frame.
_SOURCE_DD = frozenset(
    "_ipio2 _half _pi2o6 _pi _one _two _zero _C _num_C".split())


def _resolver(bodies):
    def resolve(name):
        return bodies.get(name)
    return resolve


def _src_dd(name):
    return name in _SOURCE_DD


# Lnrat's TScale overload (kokkosUtils.h:139-141): the B10 leaf.  Straight-line —
# kLog/kAbs/Sign are Class-1, _ipio2 is source, the rest are casts.
_LNRAT_BODY = (
    "template<typename TOutput, typename TMass, typename TScale> "
    "KOKKOS_INLINE_FUNCTION TOutput Lnrat(TScale const& x, TScale const& y) { "
    "return TOutput(ql::kLog(ql::kAbs(x / y))) - "
    "(ql::Constants<TScale>::template _ipio2<TOutput,TMass,TScale>() * "
    "TOutput(ql::Sign(-x) - ql::Sign(-y))); }")


def _kw(surface, bodies):
    return dict(
        call_graph=None, surface=surface,
        is_class1_synthesizable=is_class1_synthesizable,
        source_instantiates_at_dd=_src_dd,
        resolve_primary_body=_resolver(bodies),
        scalar_type="Kokkos::Experimental::DoubleDouble",
        type_tokens={"TOutput", "TMass", "TScale"})


# --------------------------------------------------------------------------- #
# scan_call_targets — the body call scanner
# --------------------------------------------------------------------------- #

def test_scan_call_targets_plain_and_template_id():
    body = "return ql::kLog(x) + ql::ddilog<T,U,V>(y) + Cast(z);"
    got = {last for _q, last in scan_call_targets(body)}
    assert {"kLog", "ddilog", "Cast"} <= got


def test_scan_call_targets_member_accessor_template():
    # Constants<TScale>::template _ipio2<...>() — the qualified accessor is scanned.
    body = "return ql::Constants<TScale>::template _ipio2<A,B,C>();"
    lasts = {last for _q, last in scan_call_targets(body)}
    assert "_ipio2" in lasts


# --------------------------------------------------------------------------- #
# clause (1) — body available
# --------------------------------------------------------------------------- #

def test_clause1_no_body_refuses(surface):
    r = clonable_leaf("ql::Foo", None, None, **_kw(surface, {}))
    assert not r.ok
    assert "clause (1)" in r.reason


def test_clause1_empty_body_refuses(surface):
    r = clonable_leaf("ql::Foo", "   ", None, **_kw(surface, {}))
    assert not r.ok
    assert "clause (1)" in r.reason


# --------------------------------------------------------------------------- #
# clause (2) — every callee at the dd boundary (the headline B10 case)
# --------------------------------------------------------------------------- #

def test_clause2_lnrat_is_clonable(surface):
    # THE B10 unblock: Lnrat's whole support surface is Class-1 wrappers + a source
    # Constants accessor.  A false negative here would keep B10 blocked.
    r = clonable_leaf("ql::Lnrat", _LNRAT_BODY, [("TScale", "x"), ("TScale", "y")],
                      **_kw(surface, _CLASS1))
    assert r.ok, r.reason
    assert r.transitive_deps == []      # Lnrat is a sink (§2.7): no clonable-leaf deps


def test_clause2_vendored_and_math_ops_are_boundary(surface):
    # A body naming only a vendored quad:: op and a <cmath> op the surface provides.
    body = ("template<typename T> KOKKOS_INLINE_FUNCTION T Foo(T const& x)"
            "{ return Kokkos::Experimental::abs(x) + ql::sqrt(x); }")
    r = clonable_leaf("ql::Foo", body, None, **_kw(surface, {}))
    assert r.ok, r.reason


def test_clause2_math_op_absent_from_surface_refuses(surface):
    # STOP #S analogue: a <cmath> op the vendored surface does not provide.
    body = ("template<typename T> KOKKOS_INLINE_FUNCTION T Foo(T const& x)"
            "{ return ql::tgamma(x); }")     # tgamma not in _SCALAR_OPS/_COMPLEX_OPS
    r = clonable_leaf("ql::Foo", body, None, **_kw(surface, {}))
    assert not r.ok
    assert "STOP #S" in r.reason and "clause (2)" in r.reason


def test_clause2_unresolvable_callee_refuses(surface):
    # A body naming an app call that is neither vendored, Class-1, source, nor a
    # resolvable leaf → refuse (conservative; would be a chain_closure_escapes).
    body = ("template<typename T> KOKKOS_INLINE_FUNCTION T Foo(T const& x)"
            "{ return ql::mystery(x); }")
    r = clonable_leaf("ql::Foo", body, None, **_kw(surface, {}))
    assert not r.ok
    assert "clause (2)" in r.reason


def test_clause2_transitive_clonable_leaf_recorded(surface):
    # A leaf whose body calls ANOTHER clonable leaf that is NOT Class-1 (multi-
    # statement, so the shallow-wrapper recognizer refuses it) but whose own callees
    # are all boundary: the inner leaf is recorded in transitive_deps and pulled into
    # F by rule (d).
    inner = ("template<typename T> KOKKOS_INLINE_FUNCTION T Helper(T const& x)"
             "{ T a = ql::kLog(x); T b = ql::kAbs(a); return a + b; }")
    outer = ("template<typename T> KOKKOS_INLINE_FUNCTION T Outer(T const& x)"
             "{ return ql::Helper(x) + ql::kLog(x); }")
    bodies = dict(_CLASS1, Helper=inner)
    r = clonable_leaf("ql::Outer", outer, None, **_kw(surface, bodies))
    assert r.ok, r.reason
    assert r.transitive_deps == ["Helper"]


def test_clause2_transitive_refusal_propagates(surface):
    # If the inner leaf is NOT clonable, the outer leaf is refused too (conservative).
    inner = ("template<typename T> KOKKOS_INLINE_FUNCTION T Helper(T const& x)"
             "{ T a = ql::mystery(x); return a; }")
    outer = ("template<typename T> KOKKOS_INLINE_FUNCTION T Outer(T const& x)"
             "{ return ql::Helper(x); }")
    bodies = dict(_CLASS1, Helper=inner)
    r = clonable_leaf("ql::Outer", outer, None, **_kw(surface, bodies))
    assert not r.ok
    assert "Helper" in r.reason and "clause (2)" in r.reason


def test_frame_names_edge_is_not_a_leaf(surface):
    # A call to a name already in F is a chain-internal edge (rule c), not a leaf to
    # classify — so a body that ONLY calls a frame is trivially clonable.
    body = ("template<typename T> KOKKOS_INLINE_FUNCTION T Foo(T const& x)"
            "{ return ql::ddilog<T,U,V>(x); }")
    kw = _kw(surface, {})
    kw["frame_names"] = frozenset({"ddilog"})
    r = clonable_leaf("ql::Foo", body, None, **kw)
    assert r.ok, r.reason
    assert r.transitive_deps == []


# --------------------------------------------------------------------------- #
# clause (3) — self-recursion under a same-name overload set (STOP #K guard)
# --------------------------------------------------------------------------- #

def test_clause3_self_call_single_def_is_ok(surface):
    # A single-def self-recursive leaf: the rename g -> g_B10 rewrites the self-call,
    # so it is safe (§3.2).
    body = ("template<typename T> KOKKOS_INLINE_FUNCTION T Foo(T const& x)"
            "{ return ql::Foo(x) + ql::kLog(x); }")
    g = CallGraph(root="R", tu_file="t.h")
    g.defs["Foo"] = [FuncDef("Foo", "f.h", 1, 3, True)]
    kw = _kw(surface, _CLASS1)
    kw["call_graph"] = g
    r = clonable_leaf("ql::Foo", body, None, **kw)
    assert r.ok, r.reason


def test_clause3_self_call_overload_set_refuses(surface):
    # A same-name OVERLOAD SET the rename cannot separate: C++ re-selects a sibling by
    # argument type ignoring <...> — the STOP #K recursion pit.  Refuse.
    body = ("template<typename T> KOKKOS_INLINE_FUNCTION T Foo(T const& x)"
            "{ return ql::Foo(x) + ql::kLog(x); }")
    g = CallGraph(root="R", tu_file="t.h")
    g.defs["Foo"] = [FuncDef("Foo", "f.h", 1, 3, True),
                     FuncDef("Foo", "f.h", 5, 7, True)]
    kw = _kw(surface, _CLASS1)
    kw["call_graph"] = g
    r = clonable_leaf("ql::Foo", body, None, **kw)
    assert not r.ok
    assert "STOP #K" in r.reason and "clause 3" in r.reason


# --------------------------------------------------------------------------- #
# clause (4) — no inward widening of a shared parameter
# --------------------------------------------------------------------------- #

def test_clause4_pure_clone_is_ok_by_default(surface):
    # No binds_shared_param evidence → a pure clone (own params) → clause (4) holds.
    r = clonable_leaf("ql::Lnrat", _LNRAT_BODY, [("TScale", "x"), ("TScale", "y")],
                      **_kw(surface, _CLASS1))
    assert r.ok, r.reason


def test_clause4_shared_param_bind_refuses(surface):
    body = ("template<typename T> KOKKOS_INLINE_FUNCTION T Foo(T const& x)"
            "{ return ql::kLog(x); }")
    kw = _kw(surface, _CLASS1)
    kw["binds_shared_param"] = lambda p: p == "x"
    r = clonable_leaf("ql::Foo", body, [("T", "x")], **kw)
    assert not r.ok
    assert "clause (4)" in r.reason


# --------------------------------------------------------------------------- #
# recursion depth cap (§2.8 backstop) + cycle safety
# --------------------------------------------------------------------------- #

def test_depth_cap_refuses_with_circuit_breaker_reason(surface):
    # A chain of leaves deeper than max_depth trips the predicate's backstop, which
    # rule (d) maps to chain_closure_oversized.
    bodies = {
        "L1": "template<typename T> T L1(T const& x){ return ql::L2(x); }",
        "L2": "template<typename T> T L2(T const& x){ return ql::L3(x); }",
        "L3": "template<typename T> T L3(T const& x){ return ql::L4(x); }",
        "L4": "template<typename T> T L4(T const& x){ return ql::kLog(x); }",
    }
    bodies.update(_CLASS1)
    kw = _kw(surface, bodies)
    r = clonable_leaf("ql::L1", bodies["L1"], None, depth=1, max_depth=2, **kw)
    assert not r.ok
    assert "circuit breaker" in r.reason


def test_cycle_is_safe(surface):
    # A -> B -> A source cycle must not spin; the seen-guard treats a revisit as an
    # accepted internal edge (§2.7 — the real graph is a DAG, but never loop).
    bodies = {
        "A": "template<typename T> T A(T const& x){ return ql::B(x); }",
        "B": "template<typename T> T B(T const& x){ return ql::A(x) + ql::kLog(x); }",
    }
    bodies.update(_CLASS1)
    r = clonable_leaf("ql::A", bodies["A"], None, **_kw(surface, bodies))
    assert r.ok, r.reason


# --------------------------------------------------------------------------- #
# result type
# --------------------------------------------------------------------------- #

def test_result_is_dataclass_shape():
    r = ClonableLeafResult(ok=True, reason="x", transitive_deps=["a"])
    assert r.ok and r.reason == "x" and r.transitive_deps == ["a"]
    assert ClonableLeafResult(ok=False, reason="y").transitive_deps == []


# --------------------------------------------------------------------------- #
# is_dd_boundary — the shared boundary classifier (§2.6)
# --------------------------------------------------------------------------- #

def test_is_dd_boundary_vendored_math_source_class1(surface):
    from agents.patcher.clonable_leaf import is_dd_boundary
    kw = dict(surface=surface, source_instantiates_at_dd=_src_dd,
              is_class1_synthesizable=is_class1_synthesizable,
              resolve_primary_body=_resolver(_CLASS1))
    # (i) vendored quad:: op
    assert is_dd_boundary("Kokkos::Experimental::abs", "abs", **kw)
    # (i') <cmath> op the surface provides
    assert is_dd_boundary("ql::log", "log", **kw)
    # (i') <cmath> op the surface LACKS -> not a boundary
    assert not is_dd_boundary("ql::tgamma", "tgamma", **kw)
    # (iii) source-instantiated Constants accessor
    assert is_dd_boundary("Constants::_ipio2", "_ipio2", **kw)
    # (ii) Class-1 synthesizable wrapper
    assert is_dd_boundary("ql::kLog", "kLog", **kw)
    # a plain unresolvable app call -> not a boundary (it is a leaf candidate)
    assert not is_dd_boundary("ql::mystery", "mystery", **kw)
