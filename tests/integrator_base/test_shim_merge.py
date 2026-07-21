"""Unit tests for the Wave-3 TU-global shim dedup (agents.integrator_base.shim_merge).

The failure these guard against: the Patcher emitted a *full* ``template<> struct
ql::Constants<T>`` (and full ``ql::`` free-function definitions) per region, so the
second region into a translation unit redefined a symbol the first already owned —
``error: redefinition of 'struct ql::Constants<...>'`` (72/79 of the WAVE3
residual).  The merge unifies TU-global symbols into ONE canonical per-family shim.

These tests are symbol-agnostic where it matters (a synthetic ``Widget<T>`` and a
plain ``helper`` free function), proving the mechanism is not hard-coded to
``Constants`` by name — it dedups *any* TU-global symbol.
"""

from __future__ import annotations

import re

from agents.integrator_base import shim_merge as sm


def _n_defs(text: str, name: str) -> int:
    """Count real member/function *definitions* of ``name`` (``... name(...) {``)."""
    return len(re.findall(r'\b' + re.escape(name) + r'\s*\([^;{]*\)\s*(?:const\s*)?\{', text))


def _n_specs(text: str, struct: str, targ: str) -> int:
    """Count specializations ``template<> struct <struct>< <targ> > {`` (::-agnostic)."""
    norm = re.sub(r'\s+', '', text).replace('::', '')
    key = f"structConstants<{targ}>".replace("Constants", struct) if struct != "Constants" \
        else f"struct{struct}<{targ}>"
    return norm.count(re.sub(r'\s+', '', key).replace('::', ''))


# --------------------------------------------------------------------------- #
# fixtures — minimal, well-formed shim bodies
# --------------------------------------------------------------------------- #

def _dd_constants(member: str) -> str:
    return (
        "#pragma once\n// SOURCE_HASH: PENDING\n"
        "#include <dd_math.hpp>\n#include <dd_complex.hpp>\n"
        "namespace ql {\n"
        "template <class T> struct Constants;\n"
        "template <>\n"
        "struct Constants< ::quad::ddfun::ddouble > {\n"
        f"{member}\n"
        "};\n"
        "} // namespace ql\n"
    )


_ONE = ("    static inline ::quad::ddfun::ddouble _one() {\n"
        "        return ::quad::ddfun::ddouble(1.0);\n    }")
_TWO = ("    static inline ::quad::ddfun::ddouble _two() {\n"
        "        return ::quad::ddfun::ddouble(2.0);\n    }")


# --------------------------------------------------------------------------- #
# 1. two shims, same Constants<T> → one spec, union of members
# --------------------------------------------------------------------------- #

def test_same_type_members_union_into_one_spec():
    merged = sm.merge_into_canonical(_dd_constants(_ONE), _dd_constants(_TWO))
    assert _n_specs(merged, "Constants", "quad::ddfun::ddouble") == 1
    assert _n_defs(merged, "_one") == 1
    assert _n_defs(merged, "_two") == 1
    # forward decl precedes the specialization (a spec of an undeclared primary
    # would not compile).
    assert merged.index("struct Constants;") < merged.index("struct Constants<")


def test_leading_colon_type_spelling_still_dedups():
    # The LLM sometimes writes `Constants< ::quad...>` and sometimes
    # `Constants<quad...>` — the same type to the compiler, must still collapse.
    a = _dd_constants(_ONE)
    b = _dd_constants(_TWO).replace("::quad::ddfun::ddouble >", "quad::ddfun::ddouble >")
    merged = sm.merge_into_canonical(a, b)
    assert _n_specs(merged, "Constants", "quad::ddfun::ddouble") == 1
    assert _n_defs(merged, "_one") == 1 and _n_defs(merged, "_two") == 1


# --------------------------------------------------------------------------- #
# 2. two shims, different T → both specs coexist, no interference
# --------------------------------------------------------------------------- #

def test_different_types_coexist():
    dd = _dd_constants(_ONE)
    cx = (
        "#pragma once\n// SOURCE_HASH: PENDING\n#include <dd_complex.hpp>\n"
        "namespace ql {\n"
        "template <class T> struct Constants;\n"
        "template <>\nstruct Constants< ::quad::ddfun::ddcomplex > {\n"
        "    static inline ::quad::ddfun::ddcomplex _one() {\n"
        "        return ::quad::ddfun::ddcomplex(1.0, 0.0);\n    }\n"
        "};\n} // namespace ql\n"
    )
    merged = sm.merge_into_canonical(dd, cx)
    assert _n_specs(merged, "Constants", "quad::ddfun::ddouble") == 1
    assert _n_specs(merged, "Constants", "quad::ddfun::ddcomplex") == 1
    # exactly one forward decl shared by both specializations
    assert merged.count("struct Constants;") == 1


# --------------------------------------------------------------------------- #
# 3. two shims each emitting ql::Real(T) for the same T → one definition
# --------------------------------------------------------------------------- #

def _dd_free_fn(defn: str) -> str:
    return (
        "#pragma once\n// SOURCE_HASH: PENDING\n#include <dd_math.hpp>\n"
        "namespace ql {\n" + defn + "\n} // namespace ql\n"
    )


_REAL = ("inline ::quad::ddfun::ddouble Real(const ::quad::ddfun::ddcomplex& z) {\n"
         "    return z.re;\n}")


def test_same_free_function_dedups():
    # even with different parameter *names*, the same (name, arg-types) collapses.
    a = _dd_free_fn(_REAL)
    b = _dd_free_fn(_REAL.replace("& z", "& w").replace("z.re", "w.re"))
    merged = sm.merge_into_canonical(a, b)
    assert _n_defs(merged, "Real") == 1


def test_different_free_functions_coexist():
    a = _dd_free_fn(_REAL)
    lnrat = ("inline ::quad::ddfun::ddouble Lnrat(const ::quad::ddfun::ddouble& x,\n"
             "                                     const ::quad::ddfun::ddouble& y) {\n"
             "    return quad::ddfun::log(x / y);\n}")
    merged = sm.merge_into_canonical(a, _dd_free_fn(lnrat))
    assert _n_defs(merged, "Real") == 1
    assert _n_defs(merged, "Lnrat") == 1


# --------------------------------------------------------------------------- #
# 4. redundant member (another shim already emitted it) → no dup, no error
# --------------------------------------------------------------------------- #

def test_redundant_member_no_duplicate():
    merged = sm.merge_into_canonical(_dd_constants(_ONE), _dd_constants(_ONE))
    assert _n_specs(merged, "Constants", "quad::ddfun::ddouble") == 1
    assert _n_defs(merged, "_one") == 1


def test_keep_first_on_conflicting_body():
    # If two shims define _one() with different bodies, keep the FIRST (already
    # committed + validated) one — the incoming duplicate is dropped, not appended.
    a = _dd_constants(_ONE)
    b = _dd_constants(_ONE.replace("ddouble(1.0)", "ddouble(1.0e0)"))
    merged = sm.merge_into_canonical(a, b)
    assert _n_defs(merged, "_one") == 1
    assert "ddouble(1.0)" in merged and "ddouble(1.0e0)" not in merged


# --------------------------------------------------------------------------- #
# 5. regression — a lone shim (no prior sibling) still assembles correctly
# --------------------------------------------------------------------------- #

def test_lone_shim_assembles_unchanged_semantics():
    lone = sm.merge_into_canonical(None, _dd_constants(_ONE))
    assert lone.startswith("#pragma once")
    assert _n_specs(lone, "Constants", "quad::ddfun::ddouble") == 1
    assert _n_defs(lone, "_one") == 1
    assert "#include <dd_math.hpp>" in lone
    assert "#include <dd_complex.hpp>" in lone
    # include union across the lone body is order-preserving + de-duplicated
    assert lone.count("#include <dd_math.hpp>") == 1


def test_empty_existing_is_same_as_none():
    a = sm.merge_into_canonical(None, _dd_constants(_ONE))
    b = sm.merge_into_canonical("", _dd_constants(_ONE))
    # both are first-lander renders of the same body → structurally identical
    assert a == b


# --------------------------------------------------------------------------- #
# generality — not hard-coded to Constants / ql
# --------------------------------------------------------------------------- #

def test_generalizes_to_arbitrary_specialization_and_namespace():
    def widget(member, ns="app"):
        return (
            "#pragma once\n// SOURCE_HASH: PENDING\n#include <w.hpp>\n"
            f"namespace {ns} {{\n"
            "template <class T> struct Widget;\n"
            "template <>\nstruct Widget<float> {\n" + member + "\n};\n"
            f"}} // namespace {ns}\n"
        )
    a = widget("    static float a() { return 1.0f; }")
    b = widget("    static float b() { return 2.0f; }")
    merged = sm.merge_into_canonical(a, b)
    assert re.sub(r'\s+', '', merged).count("structWidget<float>") == 1
    assert _n_defs(merged, "a") == 1 and _n_defs(merged, "b") == 1
