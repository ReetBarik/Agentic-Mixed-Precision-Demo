"""Phase 2d — complex-container promotion + supporting fixes.

Covers the four 2d-A fixes and their integration:
  * type_resolve — template-parameter → concrete-type binding + complex classification;
  * boundary.promote_region_block — complex read/decl promotion, functional-cast
    rewrite, complex demote, and the "no local write landing → no-op" guard;
  * region_scan — subscript exclusion + complex-read classification;
  * shim_merge.dedup_inline — the duplicate-``inline`` sanitizer;
  * regional._lint_complex_antipattern — the Kokkos/std complex-on-extended lint;
  * fanout.Promote — complex-field serialization round-trip.

Fixtures mirror the real Phase-2c ``llm_gen_failed`` regions (B2m.h:188/193,
B0m.h:405, boxGPU.h:140) so a regression re-surfaces the exact failing shape.
"""

from __future__ import annotations

from agents.integrator_base import boundary
from agents.integrator_base import shim_merge
from agents.integrator_base.regional import _lint_complex_antipattern
from agents.patcher.fanout import Promote, VariantSpec
from agents.shared import region_scan, type_resolve


# --------------------------------------------------------------------------- #
# type_resolve
# --------------------------------------------------------------------------- #

def _write_app(tmp_path):
    """A minimal app: an entry template + a vanilla and a dd concrete instantiation."""
    (tmp_path / "recipes.hpp").write_text(
        "template <class TOutput, class TMass, class TScale, class Printer>\n"
        "int run_app(int argc, char** argv) { return 0; }\n")
    (tmp_path / "vanilla.cpp").write_text(
        "int main(int c, char** v){ return ns::run_app<Kokkos::complex<double>, "
        "double, double, ns::VanillaPrinter>(c, v); }\n")
    (tmp_path / "dd.cpp").write_text(
        "int main(int c, char** v){ return ns::run_app<ql::ddfun::ddcomplex, "
        "ql::ddfun::ddouble, ql::ddfun::ddouble, ns::DDPrinter>(c, v); }\n")
    return [str(tmp_path)]


def test_classify_concrete_type():
    c = type_resolve.classify_concrete_type
    assert c("Kokkos::complex<double>") == "complex"
    assert c("std::complex<float>") == "complex"
    assert c("const Kokkos::complex<double>&") == "complex"
    assert c("double") == "scalar"
    assert c("TMass") == "scalar"
    assert c("Kokkos::View<TOutput* [3]>") == "aggregate"
    assert c("Kokkos::Array<double, 4>") == "aggregate"
    assert c("double*") == "aggregate"


def test_resolve_bindings_picks_vanilla(tmp_path):
    roots = _write_app(tmp_path)
    b = type_resolve.resolve_bindings(roots, caller_type="double")
    assert b["TOutput"] == "Kokkos::complex<double>"
    assert b["TMass"] == "double" and b["TScale"] == "double"
    assert type_resolve.complex_param_names(b) == frozenset({"TOutput"})
    assert "TOutput" in type_resolve.complex_type_tokens(b)
    assert "complex" in type_resolve.complex_type_tokens(b)


def test_resolve_bindings_picks_dd_for_dd_caller(tmp_path):
    roots = _write_app(tmp_path)
    b = type_resolve.resolve_bindings(roots, caller_type="ql::ddfun::ddouble")
    # the dd caller selects the dd instantiation (its scalar args are DoubleDouble)
    assert b["TMass"] == "ql::ddfun::ddouble"


def test_resolve_bindings_empty_without_roots():
    assert type_resolve.resolve_bindings([], "double") == {}
    assert type_resolve.resolve_bindings(None, "double") == {}


# --------------------------------------------------------------------------- #
# region_scan — subscript exclusion + complex classification
# --------------------------------------------------------------------------- #

_FUNC = """\
KOKKOS_INLINE_FUNCTION
void f(const TMass sibar, const TMass tabar, const TOutput wlogsmu) {
    Kokkos::Array<TMass, 13> xpi_in;
    const TOutput fac = TOutput(sibar * tabar);
    Y[0][1] = ql::Constants<TMass>::_half() * (xpi_in[0] + xpi_in[1]);
}
"""


def test_region_scan_excludes_subscripted():
    # line 5 (Y = ... xpi_in[..]) — xpi_in is used as a subscript base → excluded.
    reads = region_scan.region_reads_from_function(_FUNC, 1, 5, 5)
    assert "xpi_in" not in reads


def test_region_complex_read_names():
    tokens = {"TOutput", "complex"}
    names = region_scan.region_complex_read_names(_FUNC, tokens)
    assert "wlogsmu" in names      # param declared const TOutput
    assert "sibar" not in names    # TMass is real scalar
    assert "tabar" not in names


def test_name_core_types():
    m = region_scan.name_core_types(_FUNC)
    assert m.get("sibar") == "TMass"
    assert m.get("wlogsmu") == "TOutput"
    assert m.get("fac") == "TOutput"


# --------------------------------------------------------------------------- #
# boundary — complex promotion
# --------------------------------------------------------------------------- #

FF = "Kokkos::Experimental::FloatFloat"
FFC = "Kokkos::Experimental::FloatFloatComplex"
TOK = frozenset({"TOutput", "complex"})


def test_complex_decl_promotes_to_container():
    # B2m.h:188 shape: `const TOutput fac = TOutput(sibar * tabar);`
    region = "        const TOutput fac = TOutput(sibar * tabar);"
    block, promoted = boundary.promote_region_block(
        region, ["sibar", "tabar"], [], FF, "double", True,
        complex_type=FFC, complex_tokens=TOK, complex_names=frozenset(),
        caller_complex="Kokkos::complex<double>")
    text = "\n".join(block)
    assert promoted
    # real reads promote to the scalar
    assert f"{FF} sibar__ff = {FF}(sibar);" in text
    # the complex-typed decl promotes to the container, and the functional cast is
    # rewritten to the container ctor (fed by the promoted scalar product)
    assert f"const {FFC} fac__ext = {FFC}(sibar__ff * tabar__ff);" in text
    # exit reconstructs the caller complex from the two limbs of each component
    assert "TOutput fac = TOutput(static_cast<double>(fac__ext.re.hi)" in text
    assert "fac__ext.im.hi" in text
    # NEVER a scalar cast on a complex, NEVER Kokkos::complex<FloatFloat>
    assert f"{FF}(TOutput" not in text
    assert f"complex<{FF}" not in text and f"complex< {FF}" not in text


def test_complex_read_promotes_to_container():
    # B2m.h:193 shape: `const TOutput wlog = wlogsmu + wlogtmu - wlog4mu;`
    region = "        const TOutput wlog = wlogsmu + wlogtmu - wlog4mu;"
    reads = ["wlogsmu", "wlogtmu", "wlog4mu"]
    block, promoted = boundary.promote_region_block(
        region, reads, [], FF, "double", True,
        complex_type=FFC, complex_tokens=TOK, complex_names=frozenset(reads),
        caller_complex="Kokkos::complex<double>")
    text = "\n".join(block)
    assert promoted
    # each complex read promotes via component-wise scalar wrap (precision-safe)
    assert (f"{FFC} wlogsmu__ff = {FFC}({FF}(wlogsmu.real()), "
            f"{FF}(wlogsmu.imag()));") in text
    assert f"const {FFC} wlog__ext = wlogsmu__ff + wlogtmu__ff - wlog4mu__ff;" in text


def test_no_local_write_landing_is_noop():
    # boxGPU.h:140 shape: `res(i, 0) /= scalefac2;` — only a subscripted aggregate
    # store, no promotable local.  UPCAST (ff, two_limb=True): the widened read is
    # truncated back / unconvertible at the sink → honest no-op (promotion_no_op).
    region = "        res(i, 0) /= scalefac2;"
    block, promoted = boundary.promote_region_block(
        region, ["scalefac2"], [], FF, "double", True,
        complex_type=FFC, complex_tokens=TOK)
    assert not promoted
    assert block == [region]


def test_downcast_read_only_aggregate_sink_promotes():
    # Same boxGPU.h:140 shape under the FLOAT rung (two_limb=False).  Demoting the read
    # to float and feeding it into the (double) aggregate sink genuinely loses precision
    # (2c measured de≈5.8e-8 ≫ baseline) — a real, discriminating measurement, NOT a
    # no-op.  Regression guard: the first 2d-A cut's no-local-write guard fired here too
    # and killed boxGPU.h:140-142's float measurements on B1.
    region = "        res(i, 0) /= scalefac2;"
    block, promoted = boundary.promote_region_block(
        region, ["scalefac2"], [], "float", "double", False,
        complex_type="Kokkos::complex<float>", complex_tokens=TOK)
    text = "\n".join(block)
    assert promoted
    assert "float scalefac2__ff = float(scalefac2);" in text
    assert "res(i, 0) /= scalefac2__ff;" in text


def test_scalar_only_region_still_promotes():
    # boxGPU.h:139 shape: `const TScale scalefac2 = scalefac * scalefac;` — a real
    # scalar local decl still promotes to the scalar (regression guard).
    region = "        const TScale scalefac2 = scalefac * scalefac;"
    block, promoted = boundary.promote_region_block(
        region, ["scalefac"], [], FF, "double", True,
        complex_type=FFC, complex_tokens=TOK)
    text = "\n".join(block)
    assert promoted
    assert f"{FF} scalefac2__ext = scalefac__ff * scalefac__ff;" in text
    assert FFC not in text          # no complex anywhere — it's a real-scalar region


def test_float_rung_complex_uses_native_container():
    region = "        const TOutput fac = TOutput(sibar * tabar);"
    block, promoted = boundary.promote_region_block(
        region, ["sibar", "tabar"], [], "float", "double", False,
        complex_type="Kokkos::complex<float>", complex_tokens=TOK,
        caller_complex="Kokkos::complex<double>")
    text = "\n".join(block)
    assert promoted
    assert "const Kokkos::complex<float> fac__ext = Kokkos::complex<float>(" in text
    # native (single-limb) complex demotes via real()/imag(), not .re.hi
    assert "static_cast<double>(fac__ext.real())" in text
    assert ".re.hi" not in text


def test_pre2d_scalar_behavior_unchanged():
    # complex_type=None → the exact pre-2d transform.
    region = "        const TScale x = a + b;"
    block, promoted = boundary.promote_region_block(region, ["a", "b"], [], FF)
    text = "\n".join(block)
    assert promoted
    assert f"{FF} a__ff = {FF}(a);" in text


# --------------------------------------------------------------------------- #
# boundary.write_truncation_inert (Phase 2d-B)
# --------------------------------------------------------------------------- #

TOK_CX = frozenset({"TOutput", "complex"})
CALLER_CX = "Kokkos::complex<double>"


def test_write_truncation_complex_decl_landing_ff():
    # B2m.h:188 shape under the ff UPCAST: `const TOutput fac = TOutput(sibar*tabar);`.
    # The extended product lands in `fac` (TOutput = caller complex) and is truncated
    # back at the boundary → provably inert → write_truncation.
    region = "        const TOutput fac = TOutput(sibar * tabar);"
    assert boundary.write_truncation_inert(
        region, ["sibar", "tabar"], [], True,
        caller_type="double", complex_tokens=TOK_CX, caller_complex=CALLER_CX) is True


def test_write_truncation_complex_add_chain_ff():
    # B2m.h:193 shape under ff: `const TOutput wlog = wlogsmu + wlogtmu - wlog4mu;`
    region = "        const TOutput wlog = wlogsmu + wlogtmu - wlog4mu;"
    assert boundary.write_truncation_inert(
        region, ["wlogsmu", "wlogtmu", "wlog4mu"], [], True,
        caller_type="double", complex_tokens=TOK_CX, caller_complex=CALLER_CX) is True


def test_write_truncation_not_flagged_for_float_downcast():
    # Same B2m.h:188 region under the native float DOWNCAST — truncating to a narrower
    # target is real precision loss, NOT a no-op; must still promote (build+measure).
    region = "        const TOutput fac = TOutput(sibar * tabar);"
    assert boundary.write_truncation_inert(
        region, ["sibar", "tabar"], [], False,
        caller_type="double", complex_tokens=TOK_CX, caller_complex=CALLER_CX) is False


def test_write_truncation_caseB_store_dd():
    # kokkosUtils.h:183 shape under dd: `A = TMass(... T ...);` — a pre-declared (Case-B)
    # write to the caller-precision `A`.  The Case-B store is always demoted to caller
    # precision on exit → provably inert → write_truncation.  (Real run: `A` is recovered
    # by region_writes_from_source since it is template-typed; here it is passed in.)
    region = ("            A = TMass(TMass(ql::Constants<TMass>::_half()) * "
              "ql::Real(ql::kPow<TOutput, TMass, TScale>(ql::kLog(TMass("
              "ql::Constants<TMass>::_one()) + T),2)));")
    assert boundary.write_truncation_inert(
        region, ["T"], ["A"], True, caller_type="double",
        complex_tokens=TOK_CX, caller_complex=CALLER_CX) is True


def test_write_truncation_spares_unrecognized_scalar_decl():
    # boxGPU.h:139 shape under ff: `const TScale scalefac2 = scalefac * scalefac;`.
    # TScale is an unrecognized template type — treated as a possibly-wider persistent
    # sink, so the region is left to honest build+measure (it is a real, if tiny,
    # measurement — must NOT be gated).
    region = "        const TScale scalefac2 = scalefac * scalefac;"
    assert boundary.write_truncation_inert(
        region, ["scalefac"], [], True,
        caller_type="double", complex_tokens=TOK_CX, caller_complex=CALLER_CX) is False


def test_write_truncation_bare_return_not_flagged():
    # kokkosUtils.h:212 shape: `return -(S * (B0 - H * B2) + A);` — no store landing.
    # A bare return of a multi-op reduction is not provably inert (extended precision
    # rounded once at the return could discriminate) → not flagged (conservative).
    region = "        return -(S * (B0 - H * B2) + A);"
    assert boundary.write_truncation_inert(
        region, ["S", "B0", "H", "B2", "A"], [], True,
        caller_type="double", complex_tokens=TOK_CX, caller_complex=CALLER_CX) is False


def test_write_truncation_empty_payload_not_flagged():
    # Nothing promotes (no reads / writes / promotable decls) → that is the empty-payload
    # promotion_no_op class, not write-truncation.
    region = "        T c = T(k);"   # sole operand is an int index -> no promotion
    assert boundary.write_truncation_inert(
        region, [], [], True, caller_type="double") is False


def test_region_writes_from_source_recovers_template_write():
    # kok:183-style template write is invisible to extract_region_writes(double) but
    # recovered region-locally; a subscripted/aggregate store is NOT a Case-B write.
    assert region_scan.region_writes_from_source("A = TMass(x + T);") == ["A"]
    assert region_scan.region_writes_from_source(
        "const TOutput fac = TOutput(a * b);") == []          # a decl, not Case-B
    assert region_scan.region_writes_from_source("res(i, 0) /= scalefac2;") == []
    assert region_scan.region_writes_from_source("if (a == b) { c = d; }") == ["c"]


# --------------------------------------------------------------------------- #
# shim_merge.dedup_inline
# --------------------------------------------------------------------------- #

def test_dedup_inline():
    d = shim_merge.dedup_inline
    assert d("KOKKOS_INLINE_FUNCTION inline int f()") == "KOKKOS_INLINE_FUNCTION int f()"
    assert d("inline KOKKOS_INLINE_FUNCTION int f()") == "KOKKOS_INLINE_FUNCTION int f()"
    assert d("inline inline int f()") == "inline int f()"
    assert d("inline int f()") == "inline int f()"


# --------------------------------------------------------------------------- #
# regional._lint_complex_antipattern
# --------------------------------------------------------------------------- #

def test_complex_antipattern_lint_flags_extended():
    bad = "Kokkos::complex<Kokkos::Experimental::FloatFloat> z;"
    assert _lint_complex_antipattern(bad, FF) is not None
    bad_dd = "std::complex<Kokkos::Experimental::DoubleDouble> z;"
    assert _lint_complex_antipattern(bad_dd, "Kokkos::Experimental::DoubleDouble") is not None


def test_complex_antipattern_lint_allows_ffcomplex_and_float():
    assert _lint_complex_antipattern("Kokkos::Experimental::FloatFloatComplex z;", FF) is None
    # float rung: Kokkos::complex<float> is legal and must NOT be flagged
    assert _lint_complex_antipattern("Kokkos::complex<float> z;", "float") is None


# --------------------------------------------------------------------------- #
# fanout.Promote serialization
# --------------------------------------------------------------------------- #

def test_promote_complex_fields_roundtrip():
    spec = VariantSpec(variant_name="f_B1", orig_name="f", file="x.h",
                       orig_start=1, orig_end=9)
    spec.promotes.append(Promote(
        region_start=3, region_end=3, reads=["sibar"], writes=[],
        scalar_type=FF, two_limb=True, caller_type="double",
        complex_type=FFC, complex_tokens=["TOutput", "complex"],
        complex_names=["wlog"], caller_complex="Kokkos::complex<double>"))
    back = VariantSpec.from_json(spec.to_json())
    p = back.promotes[0]
    assert p.complex_type == FFC
    assert p.complex_tokens == ["TOutput", "complex"]
    assert p.complex_names == ["wlog"]
    assert p.caller_complex == "Kokkos::complex<double>"


def test_promote_defaults_backward_compatible():
    # an old manifest Promote (no complex fields) deserializes to scalar-only defaults.
    old = {"region_start": 3, "region_end": 3, "reads": ["a"], "writes": [],
           "scalar_type": FF, "two_limb": True, "caller_type": "double"}
    spec = VariantSpec.from_json({"variant_name": "f_B1", "orig_name": "f",
                                  "file": "x.h", "orig_start": 1, "orig_end": 9,
                                  "promotes": [old]})
    p = spec.promotes[0]
    assert p.complex_type is None and p.complex_tokens == [] and p.complex_names == []
