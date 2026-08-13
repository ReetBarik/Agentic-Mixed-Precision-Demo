# Region-core element-level promotion (2026-07-28).
#
# Unit coverage for the STOP #CC fix: element occurrences ``base[k]`` of a FIXED-SIZE
# complex aggregate are promoted at the read site (and demoted on element store) so a
# promoted dd operand no longer multiplies a caller-precision ``Kokkos::complex<double>``
# element — without ever retyping the array declaration (the d1 whole-array failure mode
# stays impossible by construction).
#
# The detector keys on decl SHAPE only; app identifiers (``cxs`` / ``x4``) appear here in
# TESTS solely as fixtures, never baked into production code (no-placeholder-patterns).
from __future__ import annotations

from agents.integrator_base import boundary
from agents.shared import region_scan
from agents.shared import type_resolve

_DD = "Kokkos::Experimental::DoubleDouble"
_DDC = "Kokkos::Experimental::DoubleDoubleComplex"
_KC = "Kokkos::complex<double>"
_TOK = frozenset({"TOutput", "complex"})


# --------------------------------------------------------------------------- #
# region_scan.region_element_bases — fixed-size complex aggregate detection
# --------------------------------------------------------------------------- #

def test_detects_kokkos_array_of_complex():
    src = (
        "TOutput foo(const TMass& s) {\n"
        "    Kokkos::Array<TOutput, 3> cxs;\n"
        "    return cxs[0];\n"
        "}\n"
    )
    bases = region_scan.region_element_bases(src, _TOK)
    assert bases == {"cxs": "TOutput"}


def test_detects_c_array_of_complex():
    src = (
        "TOutput foo() {\n"
        "    TOutput cxs[3];\n"
        "    return cxs[0];\n"
        "}\n"
    )
    bases = region_scan.region_element_bases(src, _TOK)
    assert bases == {"cxs": "TOutput"}


def test_detects_std_array_of_literal_complex():
    src = "void f(){ std::array<complex, 4> zs; }\n"
    assert region_scan.region_element_bases(src, _TOK) == {"zs": "complex"}


def test_ignores_dynamic_view_container():
    # A Kokkos::View is a DYNAMIC container (design non-goal) — never an element base,
    # so B10's res(i,1) accessor stays at caller precision.
    src = "void f(Kokkos::View<TOutput**> res){ auto z = res(0,1); }\n"
    assert region_scan.region_element_bases(src, _TOK) == {}


def test_ignores_std_vector_container():
    src = "void f(){ std::vector<TOutput> v; }\n"
    assert region_scan.region_element_bases(src, _TOK) == {}


def test_ignores_non_complex_element_array():
    # Array of a real scalar — element promotion is a complex-container concern only.
    src = "void f(){ Kokkos::Array<double, 3> ds; }\n"
    assert region_scan.region_element_bases(src, _TOK) == {}


def test_ignores_non_literal_extent():
    # A non-literal extent is not a proven fixed-size aggregate.
    src = "void f(){ Kokkos::Array<TOutput, N> cxs; }\n"
    assert region_scan.region_element_bases(src, _TOK) == {}


def test_ignores_nested_aggregate_element():
    # A nested aggregate element is deferred (non-goal).
    src = "void f(){ Kokkos::Array<Kokkos::Array<TOutput,2>, 3> m; }\n"
    assert region_scan.region_element_bases(src, _TOK) == {}


# --------------------------------------------------------------------------- #
# type_resolve.array_element_type — shape-only element resolver
# --------------------------------------------------------------------------- #

def test_array_element_type_fixed_size():
    assert type_resolve.array_element_type("Kokkos::Array<TOutput, 3>") == "TOutput"
    assert type_resolve.array_element_type("std::array<complex, 4>") == "complex"


def test_array_element_type_rejects_dynamic_and_nonliteral():
    assert type_resolve.array_element_type("Kokkos::View<TOutput*>") is None
    assert type_resolve.array_element_type("std::vector<TOutput>") is None
    assert type_resolve.array_element_type("Kokkos::Array<TOutput, N>") is None


# --------------------------------------------------------------------------- #
# promote_region_block — element read wrap
# --------------------------------------------------------------------------- #

def test_element_read_is_wrapped_at_complex_type():
    # B14's shape: fac = <promoted> * cxs[k].  cxs[k] must enter the dd arithmetic as a
    # DoubleDoubleComplex so no complex<DoubleDoubleComplex> forms; the array decl is left untouched.
    region = "    TOutput fac = si * cxs[k];"
    block, promoted = boundary.promote_region_block(
        region, reads=["si"], writes=[], scalar_type=_DD,
        caller_type="double", two_limb=True,
        complex_type=_DDC, complex_tokens=_TOK, complex_names={"si"},
        caller_complex=_KC, element_bases={"cxs": "TOutput"},
    )
    assert promoted is True
    body = "\n".join(block)
    # element read wrapped component-wise, full precision preserved.
    assert f"{_DDC}({_DD}(cxs[k].real()), {_DD}(cxs[k].imag()))" in body
    # the array declaration is NEVER retyped (no ``DoubleDouble cxs`` / ``DoubleDoubleComplex cxs``).
    assert "cxs;" not in body.replace(" ", "")  # no naked retyped decl slipped in


def test_element_base_not_renamed_or_aliased():
    # d1 preserved: the base name never becomes a promoted/aliased name.
    region = "    TOutput fac = si * cxs[k];"
    block, _ = boundary.promote_region_block(
        region, reads=["si"], writes=[], scalar_type=_DD,
        caller_type="double", two_limb=True,
        complex_type=_DDC, complex_tokens=_TOK, complex_names={"si"},
        caller_complex=_KC, element_bases={"cxs": "TOutput"},
    )
    body = "\n".join(block)
    assert "cxs__ff" not in body and "cxs__ext" not in body
    # public promoted-names set must not include the base (Gap-A lint safety).
    names = boundary.compute_promoted_names(region, ["si"], [])
    assert "cxs" not in names


def test_decl_init_element_read_chains_promotion():
    # B15's shape: TOutput xs = cxs[0];  xs promotes (Rule R2) because it consumes a
    # wrapped element, while cxs stays an untouched array.
    region = (
        "    TOutput xs = cxs[0];\n"
        "    TOutput r = xs * two;"
    )
    block, promoted = boundary.promote_region_block(
        region, reads=["two"], writes=[], scalar_type=_DD,
        caller_type="double", two_limb=True,
        complex_type=_DDC, complex_tokens=_TOK, complex_names={"two"},
        caller_complex=_KC, element_bases={"cxs": "TOutput"},
    )
    assert promoted is True
    body = "\n".join(block)
    assert "xs__ext" in body                       # xs chained to promotion
    assert f"{_DDC}({_DD}(cxs[0].real())" in body   # its element read wrapped


def test_whole_array_pass_untouched_without_element_bases():
    # No element_bases → byte-identical to the pre-2d/pre-element transform.
    region = "    TOutput fac = si * cxs[k];"
    with_eb, _ = boundary.promote_region_block(
        region, reads=["si"], writes=[], scalar_type=_DD, caller_type="double",
        two_limb=True, complex_type=_DDC, complex_tokens=_TOK,
        complex_names={"si"}, caller_complex=_KC, element_bases={},
    )
    body = "\n".join(with_eb)
    assert "cxs[k]" in body                          # left verbatim
    assert "cxs[k].real()" not in body               # element itself never wrapped
    # This IS the pre-fix STOP #CC shape: a dd operand multiplies the caller-precision
    # element (``si__ff * cxs[k]``).  The fix path (element_bases set) wraps cxs[k].
    assert "si__ff * cxs[k]" in body


# --------------------------------------------------------------------------- #
# promote_region_block — element store demote
# --------------------------------------------------------------------------- #

def test_element_store_of_dd_value_is_demoted():
    # A bare store of a promoted (dd) value into a caller-precision aggregate element
    # must reconstruct the caller complex value (array stays at caller precision).
    region = (
        "    TOutput acc = si * two;\n"
        "    cxs[k] = acc;"
    )
    block, promoted = boundary.promote_region_block(
        region, reads=["si", "two"], writes=[], scalar_type=_DD,
        caller_type="double", two_limb=True,
        complex_type=_DDC, complex_tokens=_TOK, complex_names={"si", "two"},
        caller_complex=_KC, element_bases={"cxs": _KC},
    )
    assert promoted is True
    body = "\n".join(block)
    # RHS demoted to the caller complex spelling via two-limb reconstruction.
    assert "cxs[k] = Kokkos::complex<double>(static_cast<double>(" in body
    assert ".re.hi)" in body and ".im.lo)" in body


def test_element_store_of_plain_value_left_alone():
    # A store whose RHS carries no promoted value is NOT rewritten (guarded by _looks_dd).
    region = (
        "    int k = 0;\n"
        "    cxs[k] = other;"
    )
    block, _ = boundary.promote_region_block(
        region, reads=[], writes=["other"], scalar_type=_DD,
        caller_type="double", two_limb=True,
        complex_type=_DDC, complex_tokens=_TOK, complex_names={"other"},
        caller_complex=_KC, element_bases={"cxs": _KC},
    )
    body = "\n".join(block)
    # ``other`` is a Case-B write → its own boundary handling, but the element store
    # target is not double-demoted into a nested reconstruction of a caller value.
    assert "cxs[k] = Kokkos::complex<double>(static_cast<double>(cxs" not in body


# --------------------------------------------------------------------------- #
# widen_carrier_assign_line — deliverable (c) receiving-local widen (B14:754)
# --------------------------------------------------------------------------- #

def test_carrier_sibling_cast_assign_widened():
    # B14's sibling branch: ``fac`` is a widened complex carrier; the non-region
    # assignment's ``TOutput(...)`` cast must become the dd complex container.
    line = "        fac = TOutput(-xs / (m2 * m4 * ta));"
    out = boundary.widen_carrier_assign_line(
        line, frozenset({"fac"}), _DDC, _DD)
    assert out == "        fac = Kokkos::Experimental::DoubleDoubleComplex(-xs / (m2 * m4 * ta));"


def test_carrier_noncast_assign_reconstructed():
    # A non-cast complex RHS is reconstructed component-wise into the dd container.
    line = "    fac = zval;"
    out = boundary.widen_carrier_assign_line(line, frozenset({"fac"}), _DDC, _DD)
    assert out == (f"    fac = {_DDC}({_DD}((zval).real()), {_DD}((zval).imag()));")


def test_carrier_assign_already_dd_left_alone():
    # RHS already produces a dd value (names the extended scalar) → no widen.
    line = "    fac = si__ext * two;"
    assert boundary.widen_carrier_assign_line(
        line, frozenset({"fac"}), _DDC, _DD) is None


def test_carrier_assign_referencing_carrier_left_alone():
    # RHS reads another carrier → already dd-producing → no widen.
    line = "    fac = gac + one;"
    assert boundary.widen_carrier_assign_line(
        line, frozenset({"fac", "gac"}), _DDC, _DD) is None


def test_carrier_decl_and_compound_assign_not_widened():
    # A decl (leading type token) and a compound assign are NOT plain carrier stores.
    assert boundary.widen_carrier_assign_line(
        "    TOutput fac = TOutput(x);", frozenset({"fac"}), _DDC, _DD) is None
    assert boundary.widen_carrier_assign_line(
        "    fac += TOutput(x);", frozenset({"fac"}), _DDC, _DD) is None


def test_carrier_widen_noop_without_carriers():
    # Empty carrier set → strict no-op (byte-identical for non-element variants).
    line = "    fac = TOutput(x);"
    assert boundary.widen_carrier_assign_line(line, frozenset(), _DDC, _DD) is None


# --------------------------------------------------------------------------- #
# demote_exit_carriers_line — deliverable (b) designed-exit narrow (B14:764/765)
# --------------------------------------------------------------------------- #

def test_designed_exit_store_demotes_carrier():
    # ``res(i,1) = fac`` — the carrier read is reconstructed to the caller complex value.
    line = "        res(i,1) = fac;"
    out = boundary.demote_exit_carriers_line(
        line, frozenset({"fac"}), _KC, "double", True)
    assert out is not None
    assert "res(i,1) = Kokkos::complex<double>(static_cast<double>((fac).re.hi)" in out
    assert ".im.lo)" in out


def test_designed_exit_mixed_operand_only_carrier_demoted():
    # ``res(i,0) = fac * wlogtmu`` — only ``fac`` (the carrier) demotes; the caller-
    # precision co-operand ``wlogtmu`` is left untouched so the product is caller-precision.
    line = "        res(i,0) = fac * wlogtmu;"
    out = boundary.demote_exit_carriers_line(
        line, frozenset({"fac"}), _KC, "double", True)
    assert out is not None
    assert "(fac).re.hi" in out
    assert "* wlogtmu;" in out
    assert "wlogtmu).re" not in out          # wlogtmu never wrapped


def test_designed_exit_no_carrier_read_is_noop():
    # A store with no carrier read is unchanged.
    assert boundary.demote_exit_carriers_line(
        "    res(i,0) = a * b;", frozenset({"fac"}), _KC, "double", True) is None


def test_designed_exit_skips_store_target():
    # A store *to* the carrier is not a read occurrence → no demotion.
    assert boundary.demote_exit_carriers_line(
        "    fac = other;", frozenset({"fac"}), _KC, "double", True) is None
