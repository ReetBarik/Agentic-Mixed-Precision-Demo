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
    assert d.expr == f"quad::ddfun::make_dd(0x{hi:016x}ULL, 0x0000000000000000ULL)"
    # this is the value the model previously got wrong (0x34F0… + spurious lo)
    assert "358dee7a4ad4b81f" in d.expr
    assert "0x0000000000000000ULL" in d.expr


def test_double_literal_ff_splits_across_limbs():
    d = cd.derive_from_rhs("MY_TINY", "0.125", "ff")
    assert d is not None and d.expr.startswith("quad::ffun::make_ff(0x")


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
