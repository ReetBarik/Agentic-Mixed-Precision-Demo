"""Unit + compile tests for the Class-1 shallow-wrapper synthesis (Subtask L1′).

Covers :mod:`agents.integrator_base.shallow_wrapper` (the recognizer + emitter that
extends the Gap-A machinery) and its wire-in to
:mod:`agents.integrator_base.regional`.  The recognizer classifies a shallow app
wrapper by BODY SHAPE (delegation / accessor / scalar-expr / transitive), so the
qcdloop wrappers it happens to synthesize (kAbs/kLog/Real/Sign/iszero) are an
emergent consequence, never an enumerated list — every fixture here is a
qcdloop-*representative* shape, but the recognizer is exercised structurally.

The compile test (``@pytest.mark.kokkos``) makes the §6 P2 probe a permanent
regression: for each of the four shapes, the emitter's actual output is compiled
against the vendored dd surface + real Kokkos, discharging STOP #P (a recognizer
false positive would fail to compile here).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from agents.integrator_base import shallow_wrapper as sw

_REPO = Path(__file__).resolve().parents[2]
_VENDORED = _REPO / "third_party" / "include"
_HEADERS = _REPO / "runs" / "qcdloop_headers_full"
_KOKKOS = Path.home() / "kokkos-install"


# --------------------------------------------------------------------------- #
# fixtures — qcdloop-representative primary bodies (one per recognized shape)
# --------------------------------------------------------------------------- #

# A vendored surface that mirrors the real dd headers' op split (so STOP #S is
# exercised faithfully in the unit tests without reading the headers).
_SCALAR_OPS = frozenset(
    "abs sqrt cbrt exp log log2 log10 pow sin cos tan floor ceil".split())
_COMPLEX_OPS = frozenset("abs conj sqrt exp log pow sin cos real imag".split())


@pytest.fixture
def surface():
    return sw.surface_from_spelling(
        "Kokkos::Experimental::DoubleDouble", "Kokkos::Experimental::DoubleDoubleComplex",
        scalar_ops=_SCALAR_OPS, complex_ops=_COMPLEX_OPS)


# Delegation: T kAbs(T x){ return Kokkos::abs(x); }
_KABS = ("template<typename T> KOKKOS_INLINE_FUNCTION "
         "T kAbs(T const& x) { return Kokkos::abs(x); }")
# Delegation: T kLog(T x){ return Kokkos::log(x); }
_KLOG = ("template<typename T> KOKKOS_INLINE_FUNCTION "
         "T kLog(T const& x) { return Kokkos::log(x); }")
# Accessor: double Real(complex<double> z){ return z.real(); }
_REAL = ("KOKKOS_INLINE_FUNCTION double Real("
         "Kokkos::complex<double> const& z) { return z.real(); }")
_IMAG = ("KOKKOS_INLINE_FUNCTION double Imag("
         "Kokkos::complex<double> const& z) { return z.imag(); }")
# Scalar-expr: int Sign(double x){ return (double(0)<x)-(x<double(0)); }
_SIGN = ("KOKKOS_INLINE_FUNCTION int Sign(double const& x) "
         "{ return (double(0) < x) - (x < double(0)); }")
# Transitive (synthesizable): body names a Class-1 dep + a functional cast to the
# parameter's OWN type, and NO foreign template parameter.
_ISZERO = ("template<typename T> "
           "KOKKOS_INLINE_FUNCTION bool iszero(T const& x) { "
           "return ql::kAbs(x) < T(1e-20) ? true : false; }")
# Transitive (NOT synthesizable — STOP #T): the REAL qcdloop iszero body names the
# *other* template parameters TOutput/TMass in a Constants accessor's template-arg
# list; a concrete emitted overload cannot bind them, and the emitter does not do
# full template-argument substitution, so the recognizer must REFUSE.
_ISZERO_REAL = ("template<typename TOutput, typename TMass, typename TScale> "
                "KOKKOS_INLINE_FUNCTION bool iszero(TScale const& x) { "
                "return ql::kAbs(x) < ql::Constants<TScale>::template "
                "_qlonshellcutoff<TOutput, TMass, TScale>() ? true : false; }")


def _dep_kabs(name: str) -> bool:
    """Transitive-dep predicate: only ``kAbs`` is a (known) Class-1 wrapper."""
    return name == "kAbs"


# --------------------------------------------------------------------------- #
# recognizer positives — the four shapes
# --------------------------------------------------------------------------- #

def test_recognize_delegation(surface):
    r = sw.recognize(_KABS, surface)
    assert r is not None
    assert r.form == sw.FORM_DELEGATION
    assert r.inner_fn == "abs" and r.inner_root == "Kokkos"
    assert r.param_is_template


def test_recognize_delegation_log(surface):
    r = sw.recognize(_KLOG, surface)
    assert r is not None and r.form == sw.FORM_DELEGATION and r.inner_fn == "log"


def test_recognize_accessor(surface):
    r = sw.recognize(_REAL, surface)
    assert r is not None
    assert r.form == sw.FORM_ACCESSOR and r.member == "real"


def test_recognize_accessor_imag(surface):
    r = sw.recognize(_IMAG, surface)
    assert r is not None and r.form == sw.FORM_ACCESSOR and r.member == "imag"


def test_recognize_scalar_expr(surface):
    r = sw.recognize(_SIGN, surface)
    assert r is not None
    assert r.form == sw.FORM_SCALAR_EXPR
    assert r.param_type == "double"


def test_recognize_transitive(surface):
    r = sw.recognize(_ISZERO, surface, is_synth_dep=_dep_kabs)
    assert r is not None
    assert r.form == sw.FORM_TRANSITIVE
    assert r.transitive_dep == "kAbs"


def test_transitive_disabled_without_dep_predicate(surface):
    # Without ``is_synth_dep`` an inner app-call is not a known Class-1 dep → refuse
    # (conservative default; the caller must opt in to transitive recognition).
    assert sw.recognize(_ISZERO, surface) is None


def test_stop_t_refuses_foreign_template_param(surface):
    # STOP #T: the real qcdloop iszero body names TOutput/TMass (template params
    # other than the parameter's own type) in a Constants accessor's template-arg
    # list.  A concrete emitted overload cannot bind them and the emitter does not
    # do full template-arg substitution → the recognizer must REFUSE (falls to LLM).
    assert sw.recognize(_ISZERO_REAL, surface, is_synth_dep=_dep_kabs) is None
    assert not sw.is_class1_synthesizable("ql::iszero", _ISZERO_REAL, surface,
                                          is_synth_dep=_dep_kabs)


# --------------------------------------------------------------------------- #
# recognizer negatives — every shape the parser must refuse (STOP #P guards)
# --------------------------------------------------------------------------- #

def test_reject_multi_statement(surface):
    body = ("KOKKOS_INLINE_FUNCTION double f(double const& x) "
            "{ double t = x * 2.0; return t; }")
    assert sw.recognize(body, surface) is None


def test_reject_control_flow(surface):
    body = ("KOKKOS_INLINE_FUNCTION int f(double const& x) "
            "{ if (x > 0) return 1; return 0; }")
    assert sw.recognize(body, surface) is None


def test_reject_non_class1_inner_call(surface):
    # inner call to a symbol that is NOT a known Class-1 wrapper → not Class-1
    body = ("KOKKOS_INLINE_FUNCTION double f(double const& x) "
            "{ return helper(x) + double(1); }")
    assert sw.recognize(body, surface, is_synth_dep=_dep_kabs) is None


def test_reject_unknown_namespace_delegation(surface):
    # delegation to a non-_MATH_FN_NAMES op in an unknown namespace → not Class-1
    body = ("KOKKOS_INLINE_FUNCTION double f(double const& x) "
            "{ return foo::bar(x); }")
    assert sw.recognize(body, surface) is None


def test_reject_multi_param(surface):
    body = ("KOKKOS_INLINE_FUNCTION double f(double const& a, double const& b) "
            "{ return Kokkos::abs(a); }")
    assert sw.recognize(body, surface) is None


def test_reject_pointer_param(surface):
    body = ("KOKKOS_INLINE_FUNCTION double f(double* x) { return Kokkos::abs(*x); }")
    assert sw.recognize(body, surface) is None


def test_reject_empty_body(surface):
    assert sw.recognize("double f(double const& x) { }", surface) is None


def test_stop_s_refuses_unprovided_op():
    # A delegation to a valid _MATH_FN_NAMES op the vendored surface does NOT
    # provide (for either operand kind) is refused — never invent a mapping.
    narrow = sw.surface_from_spelling(
        "Kokkos::Experimental::DoubleDouble", "Kokkos::Experimental::DoubleDoubleComplex",
        scalar_ops=frozenset({"abs", "log"}), complex_ops=frozenset({"abs", "log"}))
    erfc = ("template<typename T> KOKKOS_INLINE_FUNCTION "
            "T kErfc(T const& x) { return Kokkos::erfc(x); }")
    assert sw.recognize(erfc, narrow) is None


# --------------------------------------------------------------------------- #
# emitter positives — expected output per shape (whitespace-normalized)
# --------------------------------------------------------------------------- #

def _norm(text: str) -> str:
    return " ".join(text.split())


def test_emit_delegation_scalar_and_complex(surface):
    r = sw.recognize(_KABS, surface)
    oset = sw.OverloadSet(surface=surface)
    oset.add(r, qualifier="ql")
    out = _norm(oset.render())
    # template param → both scalar and complex overloads, each redirecting to vendored abs
    assert ("KOKKOS_INLINE_FUNCTION auto kAbs(Kokkos::Experimental::DoubleDouble const& x) "
            "{ return Kokkos::Experimental::abs(x); }") in out
    assert ("KOKKOS_INLINE_FUNCTION auto kAbs(Kokkos::Experimental::DoubleDoubleComplex const& x) "
            "{ return Kokkos::Experimental::abs(x); }") in out


def test_emit_accessor(surface):
    r = sw.recognize(_REAL, surface)
    out = _norm(sw.emit_overload(r, surface, qualifier="ql",
                                 target=surface.complex))
    assert ("KOKKOS_INLINE_FUNCTION auto Real(Kokkos::Experimental::DoubleDoubleComplex const& z) "
            "{ return z.real(); }") in out


def test_emit_scalar_expr_widens_param_type(surface):
    r = sw.recognize(_SIGN, surface)
    out = _norm(sw.emit_overload(r, surface, qualifier="ql",
                                 target=surface.scalar))
    # the ``double(0)`` functional casts must widen to the promoted type
    assert ("return (Kokkos::Experimental::DoubleDouble(0) < x) - "
            "(x < Kokkos::Experimental::DoubleDouble(0));") in out
    assert "double(0)" not in out.replace("ddfun::DoubleDouble(0)", "")


def test_emit_transitive_keeps_inner_call(surface):
    r = sw.recognize(_ISZERO, surface, is_synth_dep=_dep_kabs)
    out = _norm(sw.emit_overload(r, surface, qualifier="ql",
                                 target=surface.scalar))
    assert "ql::kAbs(x)" in out           # inner Class-1 call preserved
    assert "auto iszero(Kokkos::Experimental::DoubleDouble const& x)" in out


def test_emit_carries_subtask_comment(surface):
    r = sw.recognize(_KABS, surface)
    out = sw.emit_overload(r, surface, qualifier="ql", target=surface.scalar)
    assert "Subtask L1'" in out and "kAbs" in out


# --------------------------------------------------------------------------- #
# emitter idempotence + dedup (STOP #Q)
# --------------------------------------------------------------------------- #

def test_emit_is_deterministic(surface):
    r = sw.recognize(_KABS, surface)
    a = sw.emit_overload(r, surface, qualifier="ql", target=surface.scalar)
    b = sw.emit_overload(r, surface, qualifier="ql", target=surface.scalar)
    assert a == b


def test_overloadset_dedup(surface):
    r = sw.recognize(_KABS, surface)
    oset = sw.OverloadSet(surface=surface)
    first = oset.render()
    oset.add(r, qualifier="ql")
    once = oset.render()
    oset.add(r, qualifier="ql")            # add the SAME wrapper again
    twice = oset.render()
    assert first == ""
    assert once == twice                   # idempotent: second add is a no-op
    # one key per (qualifier, fn, target) — kAbs has two targets (scalar+complex)
    assert len(oset.keys()) == 2


def test_overloadset_render_stable_order(surface):
    oset = sw.OverloadSet(surface=surface)
    oset.add(sw.recognize(_KABS, surface), qualifier="ql")
    oset.add(sw.recognize(_KLOG, surface), qualifier="ql")
    a = oset.render()
    oset2 = sw.OverloadSet(surface=surface)
    oset2.add(sw.recognize(_KABS, surface), qualifier="ql")
    oset2.add(sw.recognize(_KLOG, surface), qualifier="ql")
    assert a == oset2.render()


# --------------------------------------------------------------------------- #
# region-level synthesis + wire-up
# --------------------------------------------------------------------------- #

def _region_sources():
    """A source blob defining all the representative primaries in namespace ql."""
    body = "\n".join([
        "namespace ql {",
        "template<typename T> T kAbs(T const& x) { return Kokkos::abs(x); }",
        "template<typename T> T kLog(T const& x) { return Kokkos::log(x); }",
        "KOKKOS_INLINE_FUNCTION double Real(Kokkos::complex<double> const& z) "
        "{ return z.real(); }",
        "KOKKOS_INLINE_FUNCTION int Sign(double const& x) "
        "{ return (double(0) < x) - (x < double(0)); }",
        "template<typename T> "
        "KOKKOS_INLINE_FUNCTION bool iszero(T const& x) { "
        "return ql::kAbs(x) < T(1e-20) ? true : false; }",
        "}",
    ])
    return [body]


def test_synthesize_for_region_recognizes_all(surface):
    region = ("auto a = ql::kLog(ql::kAbs(x)); auto r = ql::Real(z); "
              "auto s = ql::Sign(x); bool t = ql::iszero<T>(x);")
    res = sw.synthesize_for_region(region, frozenset({"x", "z"}),
                                   _region_sources(), surface)
    recognized = {fn for _root, fn in res.recognized}
    assert recognized == {"kLog", "kAbs", "Real", "Sign", "iszero"}
    assert res.remaining == []
    # transitive iszero pulled in its kAbs dep's overload
    assert "iszero" in res.overload_set.functions()
    assert "kAbs" in res.overload_set.functions()


def test_synthesize_for_region_is_idempotent(surface):
    region = "auto a = ql::kAbs(x);"
    src = _region_sources()
    a = sw.synthesize_for_region(region, frozenset({"x"}), src, surface)
    b = sw.synthesize_for_region(region, frozenset({"x"}), src, surface)
    assert a.overload_text == b.overload_text


def test_synthesize_leaves_unrecognized_on_llm_path(surface):
    # a call whose primary is NOT in source → recognizer can't classify → remaining
    region = "auto v = ql::mystery(x);"
    res = sw.synthesize_for_region(region, frozenset({"x"}),
                                   _region_sources(), surface)
    assert res.recognized == []
    assert ("ql", "mystery", "ql") in res.remaining
    assert res.overload_text == ""


def test_comparison_not_mistaken_for_template_id(surface):
    # ``a < b`` is a comparison, not a template-id call — must not be flagged.
    calls = sw.find_qualified_app_calls("ql::x < ql::y", frozenset({"x", "y"}))
    assert calls == []


def test_math_calls_not_claimed_as_app_wrappers(surface):
    # a <cmath> qualified call stays on the Gap-A math path (find_qualified_math_calls)
    calls = sw.find_qualified_app_calls("Kokkos::abs(x)", frozenset({"x"}))
    assert calls == []


# --------------------------------------------------------------------------- #
# manifest API — is_class1_synthesizable (Step 4, consumed by L2)
# --------------------------------------------------------------------------- #

def test_is_class1_synthesizable_positives(surface):
    assert sw.is_class1_synthesizable("ql::kAbs", _KABS, surface)
    assert sw.is_class1_synthesizable("kLog", _KLOG, surface)
    assert sw.is_class1_synthesizable("ql::Real", _REAL, surface)
    assert sw.is_class1_synthesizable("Sign", _SIGN, surface)
    assert sw.is_class1_synthesizable("ql::iszero", _ISZERO, surface,
                                      is_synth_dep=_dep_kabs)


def test_is_class1_synthesizable_negatives(surface):
    multi = ("double f(double const& x) { double t = x; return t; }")
    assert not sw.is_class1_synthesizable("ql::f", multi, surface)
    # name mismatch: the primary defines kAbs, not the queried name
    assert not sw.is_class1_synthesizable("ql::WRONG", _KABS, surface)


def test_is_class1_synthesizable_pure(surface):
    # querying does not emit / mutate anything — pure predicate for L2
    before = sw.is_class1_synthesizable("ql::kAbs", _KABS, surface)
    after = sw.is_class1_synthesizable("ql::kAbs", _KABS, surface)
    assert before is after is True


# --------------------------------------------------------------------------- #
# wire-up into regional.py (recognizer-before-hint pass)
# --------------------------------------------------------------------------- #

def test_regional_surface_reads_vendored_headers():
    from agents.dd_integrator.agent import _SPEC
    from agents.integrator_base import regional
    surf = regional._build_vendored_surface(_SPEC)
    assert surf.root == "Kokkos::Experimental"
    assert "abs" in surf.scalar_ops and "log" in surf.complex_ops


def test_regional_synthesize_removes_from_llm_and_not_linted():
    from agents.dd_integrator.agent import _SPEC
    from agents.integrator_base import regional
    region = "auto a = ql::kAbs(x);"
    promoted = frozenset({"x"})
    res = regional.synthesize_shallow_wrappers(_SPEC, region, promoted,
                                               _region_sources())
    # (a) deterministic overload emitted
    assert "kAbs" in res.overload_set.functions()
    # (b) call recognized (removed from what the LLM sees)
    assert ("ql", "kAbs") in res.recognized
    # (c) the existing Gap-A bridge lint never flagged it (app wrappers are not in
    #     _MATH_FN_NAMES) — the shim carrying the synthesized overload passes clean.
    shim = "#pragma once\n" + res.overload_text + "\n"
    assert regional._lint_qualified_bridges(region, shim, promoted) is None


# --------------------------------------------------------------------------- #
# compile probe regression — the emitter output builds against the vendored dd
# surface (STOP #P: a recognizer false positive would fail to compile here)
# --------------------------------------------------------------------------- #

_needs_kokkos = pytest.mark.skipif(
    not _KOKKOS.is_dir() or not _VENDORED.is_dir(),
    reason="requires ~/kokkos-install and third_party/include")


def _module_prelude() -> str:
    from agents.build_run.agent import _module_settings
    modules, use_path = _module_settings()
    if not modules:
        return ""
    return f"module use {use_path} && module load {' '.join(modules)} && "


def _compile(cpp_path: Path, out_path: Path) -> subprocess.CompletedProcess:
    cmd = (
        f"{_module_prelude()}"
        f"g++ -std=c++20 -w "
        f"-I{_VENDORED} -I{_KOKKOS / 'include'} "
        f"{cpp_path} -L{_KOKKOS / 'lib64'} -lkokkoscore -lkokkoscontainers -ldl "
        f"-o {out_path}")
    return subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True,
                          timeout=300)


@_needs_kokkos
@pytest.mark.kokkos
def test_synthesized_overloads_compile_and_run(tmp_path):
    """Each of the four shapes' emitter output compiles + runs against vendored dd.

    Makes the §6 P2 probe (``probe_clone_synth.cpp``) a permanent regression: the
    surface here is the REAL vendored op split (scanned from the headers), the
    overloads are the emitter's ACTUAL output, and the TU exercises every shape.
    """
    from agents.integrator_base import shallow_wrapper as _sw

    ddh = (_VENDORED / "dd_math.hpp").read_text()
    ddc = (_VENDORED / "dd_complex.hpp").read_text()
    s_ops, c_ops = _sw.scan_vendored_ops([ddh, ddc], "Kokkos::Experimental::DoubleDouble",
                                         "Kokkos::Experimental::DoubleDoubleComplex")
    surf = _sw.surface_from_spelling("Kokkos::Experimental::DoubleDouble", "Kokkos::Experimental::DoubleDoubleComplex",
                                     scalar_ops=s_ops, complex_ops=c_ops)

    # a region naming all four shapes; kSqrt/kConj add complex-delegation coverage
    region = (
        "auto a = ql::kLog(ql::kAbs(x));"
        "auto r = ql::Real(z); auto i = ql::Imag(z);"
        "auto s = ql::Sign(x);"
        "auto c = ql::kConj(z); auto q = ql::kSqrt(z);"
        "bool t = ql::iszero<Kokkos::Experimental::DoubleDouble>(x);")
    res = _sw.synthesize_for_region(region, frozenset({"x", "z"}),
                                    _region_sources_full(), surf)
    overlay = res.overload_text
    assert overlay.strip(), "expected synthesized overloads"

    overlay_path = tmp_path / "synth_overlay.hpp"
    overlay_path.write_text(overlay + "\n")

    cpp = tmp_path / "probe.cpp"
    cpp.write_text(
        "#include <Kokkos_Core.hpp>\n"
        "#include <dd_math.hpp>\n"
        "#include <dd_complex.hpp>\n"
        "using Kokkos::Experimental::DoubleDouble;\n"
        "using Kokkos::Experimental::DoubleDoubleComplex;\n"
        f'#include "{overlay_path}"\n'
        "int main(int argc, char** argv){ Kokkos::initialize(argc, argv);\n"
        "  { DoubleDouble x(2.0); DoubleDoubleComplex z(1.0, 2.0);\n"
        "    DoubleDouble a = ql::kLog(ql::kAbs(x));\n"
        "    DoubleDouble r = ql::Real(z); DoubleDouble i = ql::Imag(z);\n"
        "    int s = ql::Sign(x);\n"
        "    DoubleDoubleComplex c = ql::kConj(z); DoubleDoubleComplex q = ql::kSqrt(z);\n"
        "    DoubleDouble ca = ql::kAbs(z);\n"
        "    bool t = ql::iszero(x);\n"
        "    (void)a;(void)r;(void)i;(void)s;(void)c;(void)q;(void)ca;(void)t;\n"
        "  } Kokkos::finalize(); return 0; }\n")

    exe = tmp_path / "probe"
    build = _compile(cpp, exe)
    assert build.returncode == 0, (
        f"STOP #P: synthesized overloads did NOT compile:\n{build.stderr[:4000]}\n"
        f"---overlay---\n{overlay}")
    run = subprocess.run([str(exe)], capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, f"runtime failure:\n{run.stderr[:2000]}"


def _region_sources_full():
    """Region sources including kSqrt/kConj/Imag for the compile probe coverage."""
    body = "\n".join([
        "namespace ql {",
        "template<typename T> T kAbs(T const& x) { return Kokkos::abs(x); }",
        "template<typename T> T kLog(T const& x) { return Kokkos::log(x); }",
        "template<typename T> T kSqrt(T const& x) { return Kokkos::sqrt(x); }",
        "template<typename T> T kConj(T const& x) { return Kokkos::conj(x); }",
        "KOKKOS_INLINE_FUNCTION double Real(Kokkos::complex<double> const& z) "
        "{ return z.real(); }",
        "KOKKOS_INLINE_FUNCTION double Imag(Kokkos::complex<double> const& z) "
        "{ return z.imag(); }",
        "KOKKOS_INLINE_FUNCTION int Sign(double const& x) "
        "{ return (double(0) < x) - (x < double(0)); }",
        "template<typename T> "
        "KOKKOS_INLINE_FUNCTION bool iszero(T const& x) { "
        "return ql::kAbs(x) < T(1e-20) ? true : false; }",
        "}",
    ])
    return [body]
