"""Instantiation-gate classifier tests — deterministic bucketing of dd binding errors.

The classifier is the STOP #BB guard: every g++ error in the emitted dd variant
tree must bucket into one of the four known emission-binding shapes, and any error
it cannot bucket must surface as ``unknown`` (a hard STOP), never a silent fallback.

The error strings here are verbatim samples from the REAL B10 (89) / B14 (5) build
logs (``lmeasure_run/*/patcher_runs/logs/iter_0_build.log``), so the test pins the
classifier to the actual toolchain output, not a paraphrase.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.patcher import instantiation_gate as ig

# Real B10/B14 build logs — the ground-truth corpus (present only in a full run tree).
_ROOT = Path(__file__).resolve().parents[2]
_B10_LOG = (_ROOT / "runs" / "qcdloop" / "tier_b_stage2_leaf_promotion"
            / "lmeasure_run" / "B10" / "patcher_runs" / "logs" / "iter_0_build.log")
_B14_LOG = (_ROOT / "runs" / "qcdloop" / "tier_b_stage2_leaf_promotion"
            / "lmeasure_run" / "B14" / "patcher_runs" / "logs" / "iter_0_build.log")


# --------------------------------------------------------------------------- #
# Per-shape classification (verbatim g++ messages, curly-quoted as g++ prints) #
# --------------------------------------------------------------------------- #

def test_shape1_construct():
    msg = ("no matching function for call to "
           "‘Kokkos::complex<double>::complex(quad::ddfun::ddouble)’")
    assert ig.classify_error(msg) == ig.SHAPE_1_EXIT_NARROW


def test_shape1_construct_ddcomplex():
    msg = ("no matching function for call to "
           "‘Kokkos::complex<double>::complex(quad::ddfun::ddcomplex)’")
    assert ig.classify_error(msg) == ig.SHAPE_1_EXIT_NARROW


def test_shape1_decl_conversion():
    msg = ("conversion from ‘quad::ddfun::ddcomplex’ to non-scalar type "
           "‘const Kokkos::complex<double>’ requested")
    assert ig.classify_error(msg) == ig.SHAPE_1_EXIT_NARROW


def test_shape1_store_assign():
    msg = ("no match for ‘operator=’ (operand types are "
           "‘quad::ddfun::ddcomplex’ and ‘Kokkos::complex<double>’)")
    assert ig.classify_error(msg) == ig.SHAPE_1_EXIT_NARROW


def test_shape1_could_not_convert():
    msg = ("could not convert 'Kokkos::operator-<double, double>(...)' from "
           "'Kokkos::complex<double>' to 'quad::ddfun::ddcomplex'")
    assert ig.classify_error(msg) == ig.SHAPE_1_EXIT_NARROW


def test_shape2_invalid_cast():
    msg = "invalid cast from type ‘quad::ddfun::ddouble’ to type ‘double’"
    assert ig.classify_error(msg) == ig.SHAPE_2_INTERIOR_WIDEN


def test_shape2_const_double_init():
    msg = ("cannot convert ‘quad::ddfun::ddouble’ to ‘const double’ "
           "in initialization")
    assert ig.classify_error(msg) == ig.SHAPE_2_INTERIOR_WIDEN


def test_shape2_const_ref_init():
    msg = ("invalid initialization of reference of type ‘const double&’ from "
           "expression of type ‘quad::ddfun::ddouble’")
    assert ig.classify_error(msg) == ig.SHAPE_2_INTERIOR_WIDEN


def test_shape3_nested_complex():
    msg = ("no match for ‘operator=’ (operand types are "
           "‘quad::ddfun::ddcomplex’ and "
           "‘Kokkos::complex<quad::ddfun::ddcomplex>’)")
    assert ig.classify_error(msg) == ig.SHAPE_3_NESTED_COMPLEX


def test_shape3_static_assert():
    msg = ("static assertion failed: Kokkos::complex can only be instantiated for a "
           "cv-unqualified floating point type")
    assert ig.classify_error(msg) == ig.SHAPE_3_NESTED_COMPLEX


def test_shape4_shim():
    msg = ('#error "DD Chain Integrator: ql::ddilog(ddouble) requires manual '
           'classification"')
    assert ig.classify_error(msg) == ig.SHAPE_4_SHIM


def test_ascii_quotes_also_match():
    # g++ can emit ASCII quotes (-fno-diagnostics-fancy / older locales).
    msg = "invalid cast from type 'quad::ddfun::ddouble' to type 'double'"
    assert ig.classify_error(msg) == ig.SHAPE_2_INTERIOR_WIDEN


# --------------------------------------------------------------------------- #
# The STOP #BB guard — an unrecognised error must NOT be silently bucketed     #
# --------------------------------------------------------------------------- #

def test_unknown_error_is_unknown():
    msg = "use of undeclared identifier ‘frobnicate’"
    assert ig.classify_error(msg) == ig.SHAPE_UNKNOWN


def test_unrelated_dd_shape_is_unknown():
    # A dd/dd binding the four shapes do not cover — must STOP, not degrade.
    msg = ("no match for ‘operator+’ (operand types are "
           "‘quad::ddfun::ddouble’ and ‘std::vector<int>’)")
    assert ig.classify_error(msg) == ig.SHAPE_UNKNOWN


# --------------------------------------------------------------------------- #
# Whole-log classification                                                     #
# --------------------------------------------------------------------------- #

def test_classify_synthetic_log():
    log = (
        "foo.h:1:1: note: some note\n"
        "foo.h:10:5: error: invalid cast from type "
        "‘quad::ddfun::ddouble’ to type ‘double’\n"
        "foo.h:20:5: error: no matching function for call to "
        "‘Kokkos::complex<double>::complex(quad::ddfun::ddouble)’\n"
    )
    r = ig.classify_build_log(log)
    assert r.total == 2
    assert not r.has_unknown
    assert r.counts()[ig.SHAPE_2_INTERIOR_WIDEN] == 1
    assert r.counts()[ig.SHAPE_1_EXIT_NARROW] == 1
    assert not r.ok


def test_empty_log_is_ok():
    r = ig.classify_build_log("[100%] Built target boxGPU_app\n")
    assert r.ok
    assert r.total == 0
    assert not r.has_unknown


def test_unknown_in_log_flags_stop_bb():
    log = "foo.h:1:1: error: use of undeclared identifier ‘frobnicate’\n"
    r = ig.classify_build_log(log)
    assert r.total == 1
    assert r.has_unknown
    assert len(r.unknown) == 1


# --------------------------------------------------------------------------- #
# Against the REAL build logs (the 89 / 5 corpus) — zero unknowns is the bar   #
# --------------------------------------------------------------------------- #

@pytest.mark.skipif(not _B10_LOG.is_file(), reason="B10 run log not present")
def test_real_b10_log_fully_classified():
    r = ig.classify_build_log_file(_B10_LOG)
    assert r.total == 89, r.summary()
    assert not r.has_unknown, f"STOP #BB — unclassified: {r.unknown}"
    c = r.counts()
    # Every one of the four shapes is represented in the B10 corpus.
    assert c[ig.SHAPE_1_EXIT_NARROW] > 0
    assert c[ig.SHAPE_2_INTERIOR_WIDEN] > 0
    assert c[ig.SHAPE_3_NESTED_COMPLEX] > 0
    assert c[ig.SHAPE_4_SHIM] == 1


@pytest.mark.skipif(not _B14_LOG.is_file(), reason="B14 run log not present")
def test_real_b14_log_fully_classified():
    r = ig.classify_build_log_file(_B14_LOG)
    assert r.total == 5, r.summary()
    assert not r.has_unknown, f"STOP #BB — unclassified: {r.unknown}"
    # B14 opts OUT of leaf promotion; its errors are Shapes 1 & 3 only (base emission).
    c = r.counts()
    assert c.get(ig.SHAPE_1_EXIT_NARROW, 0) > 0
    assert c.get(ig.SHAPE_3_NESTED_COMPLEX, 0) > 0
