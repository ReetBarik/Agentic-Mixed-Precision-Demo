"""Unit tests for agents.shared.constant_derive (Gap B — source-derivable
extended-precision constants).

The load-bearing guarantees:
* the mathematical catalog reproduces the vendored dd_*/ff_* pairs bit-for-bit;
* a source ``double`` literal promotes to a ZERO low word (dd) / a split (ff) —
  the exact value the model got WRONG (0x34F0… + spurious lo) on ``_ieps50``;
* the source walk resolves the generic C++ constant-definition forms;
* the cascade classifies literal / catalog / composite RHS correctly.
"""

from __future__ import annotations

import struct

from agents.shared import constant_derive as cd


def _dbits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


# --------------------------------------------------------------------------- #
# catalog fidelity (must match third_party/include/{dd,ff}_math.hpp exactly)
# --------------------------------------------------------------------------- #

def test_catalog_matches_vendored_dd_pairs():
    vendored_dd = {
        "pi":    (0x400921fb54442d18, 0x3ca1a62633145c07),
        "e":     (0x4005bf0a8b145769, 0x3ca4d57ee2b1013a),
        "ln2":   (0x3fe62e42fefa39ef, 0x3c7abc9e3b39803f),
        "sqrt2": (0x3ff6a09e667f3bcd, 0xbc9bdd3413b26456),
    }
    for name, pair in vendored_dd.items():
        assert cd.KNOWN_CONSTANTS[name]["dd"] == pair, name


def test_catalog_matches_vendored_ff_pairs():
    vendored_ff = {
        "pi":          (0x40490fdb, 0xb3bbbd2e),
        "e":           (0x402df854, 0x33b14577),
        "ln2":         (0x3f317218, 0xb102e308),
        "sqrt2":       (0x3fb504f3, 0x32cfe77a),
        "euler_gamma": (0x3f13c468, 0xb1e4127a),
    }
    for name, pair in vendored_ff.items():
        assert cd.KNOWN_CONSTANTS[name]["ff"] == pair, name


# --------------------------------------------------------------------------- #
# literal derivation — the _ieps50 heart of Gap B
# --------------------------------------------------------------------------- #

def test_double_literal_dd_has_zero_low_word():
    # A source literal carries only double precision; lo MUST be 0.
    d = cd.derive_from_rhs("_ieps50", "TScale(1e-50)", "dd")
    assert d is not None and d.how == "literal"
    hi = _dbits(1e-50)
    assert d.expr == f"Kokkos::Experimental::DoubleDouble::from_bits(0x{hi:016x}ULL, 0x0000000000000000ULL)"
    # this is the value the model previously got wrong (0x34F0… + spurious lo)
    assert "358dee7a4ad4b81f" in d.expr
    assert "0x0000000000000000ULL" in d.expr


def test_double_literal_ff_splits_across_limbs():
    d = cd.derive_from_rhs("MY_TINY", "0.125", "ff")
    assert d is not None and d.expr.startswith("Kokkos::Experimental::FloatFloat::from_bits(0x")


# --------------------------------------------------------------------------- #
# complex-container derivation (Wave 2) — the _ieps50 imaginary iε regulator
#
# _ieps50 = TOutput{_zero(), TScale(1e-50)} is 0 + 1e-50·i.  Wave-1 surfaced only
# the bare literal 1e-50 and the model collapsed the container to a REAL
# DoubleDouble(1e-50) — dropping the imaginary axis the iε prescription lives on, which
# left B0m.h:68/69 & friends stuck at dd_untested.  The container is now derived
# WHOLE (both limbs), bit-exact against the ddfun_enabled reference where
#   DoubleDouble(double h) : hi(h), lo(0.0)   →   dd_real(1e-50) == {1e-50, 0}.
# --------------------------------------------------------------------------- #

_IEPS50_RHS = "TOutput{Constants<TScale>::_zero(), TScale(1e-50)}"
# minimal synthetic sources reproducing the B0m.h:68 pattern (no app headers).
_IEPS50_SOURCES = [
    "template<class A,class B,class C>\n"
    "static TOutput _ieps50() { return " + _IEPS50_RHS + "; }\n",
    "static constexpr T _zero() { return T(0.0); }\n",
]


def test_ieps50_dd_complex_is_bit_exact_imaginary():
    d = cd.derive_complex_from_rhs("_ieps50", _IEPS50_RHS, "dd", _IEPS50_SOURCES)
    assert d is not None and d.how == "complex"
    # imaginary part == dd_real(1e-50) == {bits(1e-50), 0} (bit-exact vs ddfun ref)
    hi = _dbits(1e-50)
    assert hi == 0x358DEE7A4AD4B81F                       # the correct hi word
    assert d.imag == f"Kokkos::Experimental::DoubleDouble::from_bits(0x{hi:016x}ULL, 0x0000000000000000ULL)"
    # real part is exactly zero (the _zero() accessor resolved + derived)
    assert d.real == "Kokkos::Experimental::DoubleDouble::from_bits(0x0000000000000000ULL, 0x0000000000000000ULL)"


def test_ieps50_scalar_derive_returns_none_container_needs_complex_path():
    # the scalar cascade cannot derive a 2-element container (it is not a literal
    # or catalog closed form) — that is why the complex path exists.
    assert cd.derive_from_rhs("_ieps50", _IEPS50_RHS, "dd") is None


def test_complex_container_paren_form():
    # `Type(re, im)` constructor form is recognized too, not only braces.
    d = cd.derive_complex_from_rhs("c", "cx(0.0, 0.5)", "dd", [])
    assert d is not None
    assert d.real == cd._make_call("dd", _dbits(0.0), 0)
    assert d.imag == cd._make_call("dd", _dbits(0.5), 0)


def test_complex_container_requires_two_derivable_parts():
    # 1 part (a plain cast) is NOT a container.
    assert cd.derive_complex_from_rhs("x", "TScale(1e-50)", "dd", []) is None
    # a part that is not derivable (opaque runtime value) → None (falls to R4).
    assert cd.derive_complex_from_rhs(
        "x", "TOutput{runtime_val(), TScale(1e-50)}", "dd", []) is None


def test_parse_float_literal_strips_casts_and_suffixes():
    assert cd.parse_float_literal("TScale(1e-50)") == 1e-50
    assert cd.parse_float_literal("static_cast<double>(0.5)") == 0.5
    assert cd.parse_float_literal("1.5f") == 1.5
    assert cd.parse_float_literal("2.0") == 2.0
    assert cd.parse_float_literal("double(2)") == 2.0
    # not a plain literal -> None (falls to catalog / R4)
    assert cd.parse_float_literal("a + b") is None
    assert cd.parse_float_literal("foo(a, b)") is None


# --------------------------------------------------------------------------- #
# source walk (resolve_constant_rhs)
# --------------------------------------------------------------------------- #

def test_resolve_define():
    assert cd.resolve_constant_rhs("EPS", ["#define EPS 1e-30\n"]) == "1e-30"


def test_resolve_constexpr_assignment():
    src = "namespace app {\n  constexpr double MY_TINY = 1e-40;\n}\n"
    assert cd.resolve_constant_rhs("MY_TINY", [src]) == "1e-40"


def test_resolve_static_accessor():
    src = "static constexpr T _half() { return T(0.5); }\n"
    assert cd.resolve_constant_rhs("_half", [src]) == "T(0.5)"


def test_resolve_template_accessor():
    src = ("template<class A,class B,class C>\n"
           "static TOutput _ieps50() { return TOutput{Constants<TScale>::_zero(), "
           "TScale(1e-50)}; }\n")
    rhs = cd.resolve_constant_rhs("_ieps50", [src])
    assert rhs == "TOutput{Constants<TScale>::_zero(), TScale(1e-50)}"


def test_resolve_not_found_returns_none():
    assert cd.resolve_constant_rhs("NOPE", ["int x = 3;\n"]) is None


def test_resolve_ignores_function_like_macro():
    # #define MAX(a,b) ... is not a constant; must not match MAX.
    assert cd.resolve_constant_rhs("MAX", ["#define MAX(a,b) ((a)>(b)?(a):(b))\n"]) is None


def test_resolve_searches_multiple_sources_region_first():
    region = "double z = OTHER;\n"
    header = "#define OTHER 3.5\n"
    assert cd.resolve_constant_rhs("OTHER", [region, header]) == "3.5"


# --------------------------------------------------------------------------- #
# catalog closed forms
# --------------------------------------------------------------------------- #

def test_catalog_bare_alias():
    d = cd.derive_from_rhs("mypi", "M_PI", "dd")
    assert d is not None and d.how == "catalog:pi"
    assert d.expr == cd._make_call("dd", *cd.KNOWN_CONSTANTS["pi"]["dd"])


def test_catalog_two_pi_closed_form():
    d = cd.derive_from_rhs("twopi", "2.0*M_PI", "dd")
    assert d is not None and d.how == "catalog:two_pi"


def test_catalog_half_pi_closed_form():
    d = cd.derive_from_rhs("halfpi", "M_PI*0.5", "dd")
    assert d is not None and d.how == "catalog:half_pi"


def test_std_numbers_pi_alias():
    d = cd.derive_from_rhs("p", "std::numbers::pi_v", "ff")
    assert d is not None and d.how == "catalog:pi"


# --------------------------------------------------------------------------- #
# composite RHS — literal enumeration
# --------------------------------------------------------------------------- #

def test_literals_in_composite_complex():
    # the qcdloop _ieps50 shape: a complex {0, 1e-50}
    lits = cd.derive_literals_in("TOutput{Constants<TScale>::_zero(), TScale(1e-50)}", "dd")
    names = {l.name for l in lits}
    assert "1e-50" in names
    # a bare index / integer count is NOT surfaced as a constant
    assert not any(l.name in {"0", "2"} for l in lits)


def test_derive_from_rhs_gives_up_on_opaque():
    # runtime expression with no literal / catalog term -> None (falls to R4)
    assert cd.derive_from_rhs("x", "some_runtime_call(a, b)", "dd") is None


# --------------------------------------------------------------------------- #
# π-family catalog extension (Subtask 3) — the _pi2o6 R4-escape fix
#
# Upstream Constants<T> defines the π family COMPOSITIONALLY (kokkosMaths.h:95-127):
#   _pi2()   = _pi() * _pi()                    // π²
#   _pio3()  = _pi() / TScale(3)                // π/3
#   _pio6()  = _pi() / TScale(6)                // π/6
#   _pi2o3() = _pi() * _pio3<...>()             // π²/3
#   _pi2o6() = _pi() * _pio6<...>()             // π²/6   <- B10's #error
#   _pi2o12()= _pi2() / TScale(12)              // π²/12
# The catalog now carries each of these, DERIVED from the canonical `pi` entry at
# prec=80 (not transcribed), and derive_from_catalog recognizes the RHS shapes.
# --------------------------------------------------------------------------- #

_ALIAS = cd.PI_FAMILY_ACCESSOR_ALIASES


def _decimal_bailey_dd(value):
    """Independent reference dd split of a high-precision Decimal."""
    import struct
    from decimal import Decimal

    hi = float(value)
    lo = float(value - Decimal(hi))
    b = lambda x: struct.unpack("<Q", struct.pack("<d", x))[0]
    return b(hi), b(lo)


def _decimal_bailey_ff(value):
    import struct
    from decimal import Decimal

    f32 = lambda x: struct.unpack("<f", struct.pack("<f", x))[0]
    hi = f32(float(value))
    lo = f32(float(value - Decimal(hi)))
    b = lambda x: struct.unpack("<I", struct.pack("<f", x))[0]
    return b(hi), b(lo)


def test_pi_family_catalog_entries_present():
    for name in ("pi_squared", "pi_over_3", "pi_over_6",
                 "pi_squared_over_3", "pi_squared_over_6", "pi_squared_over_12"):
        assert name in cd.KNOWN_CONSTANTS, name
        assert "dd" in cd.KNOWN_CONSTANTS[name] and "ff" in cd.KNOWN_CONSTANTS[name]


def test_pi_family_dd_bit_exact_vs_independent_reference():
    # Each entry round-trips to Decimal within one ULP of the low limb — computed
    # from an INDEPENDENT high-precision π string (STOP #C guard).
    from decimal import Decimal

    pi = Decimal("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899")
    refs = {
        "pi_squared":         pi * pi,
        "pi_over_3":          pi / 3,
        "pi_over_6":          pi / 6,
        "pi_squared_over_3":  pi * pi / 3,
        "pi_squared_over_6":  pi * pi / 6,
        "pi_squared_over_12": pi * pi / 12,
    }
    for name, val in refs.items():
        assert cd.KNOWN_CONSTANTS[name]["dd"] == _decimal_bailey_dd(val), name


def test_pi_family_ff_correctly_rounded_vs_independent_reference():
    from decimal import Decimal

    pi = Decimal("3.14159265358979323846264338327950288419716939937510582097494459230781640628620899")
    refs = {
        "pi_squared":         pi * pi,
        "pi_over_6":          pi / 6,
        "pi_squared_over_6":  pi * pi / 6,
        "pi_squared_over_12": pi * pi / 12,
    }
    for name, val in refs.items():
        assert cd.KNOWN_CONSTANTS[name]["ff"] == _decimal_bailey_ff(val), name


def test_pi_squared_over_6_numeric_value():
    # sanity: π²/6 ≈ 1.6449340668482264 (Basel constant)
    import struct

    hi = cd.KNOWN_CONSTANTS["pi_squared_over_6"]["dd"][0]
    val = struct.unpack("<d", struct.pack("<Q", hi))[0]
    assert abs(val - 1.6449340668482264) < 1e-15


# --- RHS shape resolution (the upstream composition forms) ------------------ #

def test_compose_pi_squared_from_product():
    d = cd.derive_from_rhs("_pi2", "_pi() * _pi()", "dd", _ALIAS)
    assert d is not None and d.how == "catalog:pi_squared"


def test_compose_pi_over_6_from_division():
    d = cd.derive_from_rhs("_pio6", "_pi() / TScale(6)", "dd", _ALIAS)
    assert d is not None and d.how == "catalog:pi_over_6"


def test_compose_pi_squared_over_6_from_accessor_product():
    # the B10 case: _pi() * _pio6<TOutput, TMass, TScale>() → π²/6
    d = cd.derive_from_rhs("_pi2o6", "_pi() * _pio6<TOutput, TMass, TScale>()", "dd", _ALIAS)
    assert d is not None and d.how == "catalog:pi_squared_over_6"


def test_compose_pi_squared_over_12_from_pi2_division():
    d = cd.derive_from_rhs("_pi2o12", "_pi2() / TScale(12)", "dd", _ALIAS)
    assert d is not None and d.how == "catalog:pi_squared_over_12"


def test_compose_pi_squared_over_3_from_accessor_product():
    d = cd.derive_from_rhs("_pi2o3", "_pi() * _pio3<TOutput, TMass, TScale>()", "dd", _ALIAS)
    assert d is not None and d.how == "catalog:pi_squared_over_3"


def test_compose_cast_wrapped_variant():
    # T(_pi() * _pio6<...>()) — the whole composition wrapped in a functional cast
    d = cd.derive_from_rhs("_pi2o6", "T(_pi() * _pio6<A, B, C>())", "dd", _ALIAS)
    assert d is not None and d.how == "catalog:pi_squared_over_6"


def test_compose_template_arg_tolerance_on_accessor():
    # _pio6<...>() must resolve to _pio6 (template args stripped in the recognizer)
    d = cd.derive_from_rhs("x", "_pio6<TOutput, TMass, TScale>()", "dd", _ALIAS)
    assert d is not None and d.how == "catalog:pi_over_6"


# --- negatives (never invent a value) --------------------------------------- #

def test_compose_unknown_operand_returns_none():
    assert cd.derive_from_rhs("x", "_pi() * SomeOther()", "dd", _ALIAS) is None


def test_compose_non_catalog_divisor_returns_none():
    # π/7 is not a catalog entry → None (do not invent)
    assert cd.derive_from_rhs("x", "_pi() / TScale(7)", "dd", _ALIAS) is None


def test_compose_unknown_accessor_returns_none_no_crash():
    assert cd.derive_from_rhs("x", "_unknown() * _pi()", "dd", _ALIAS) is None


def test_compose_requires_alias_map_for_accessors():
    # without the caller-supplied alias map, the app-spelled accessors are opaque
    assert cd.derive_from_rhs("_pi2o6", "_pi() * _pio6<A,B,C>()", "dd") is None


def test_strip_casts_leaves_product_intact():
    # regression: _strip_casts must NOT corrupt a product-of-calls
    assert cd._strip_casts("_pi() * _pio6()") == "_pi() * _pio6()"
    assert cd._strip_casts("T(_pi() * _pio6())") == "_pi() * _pio6()"
    assert cd._strip_casts("TScale(1e-50)") == "1e-50"
