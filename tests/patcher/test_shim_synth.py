"""Phase-2 deliverable 2 — pipeline-authored leaf-shim synthesis.

Structural inventory extraction, precision-parameterized sibling rendering, and the
sha256 invalidation stamp.  Exercised against the real reference header (the 13 §1.1
leaves) and a synthetic header (parameterization + edge cases).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.patcher.shim_synth import (
    ShimSynthError, extract_inventory, inventory_sha256, read_embedded_sha, render_shim)

_REF = Path(__file__).resolve().parents[2] / "runs" / "qcdloop_headers_full" / "kokkosMaths.h"


# --------------------------------------------------------------------------- #
# inventory extraction (structural, against the real reference header)
# --------------------------------------------------------------------------- #

def test_inventory_finds_the_13_non_template_leaves():
    inv = extract_inventory(_REF.read_text(), reference_scalar="double")
    names = [l.name for l in inv]
    assert len(inv) == 13
    # Exactly the design §1.1 set, with the right overload multiplicities.
    assert names.count("kAbs") == 2
    assert names.count("Imag") == 2
    assert names.count("Real") == 2
    assert names.count("Sign") == 2
    assert names.count("Max") == 2
    assert names.count("Min") == 2
    assert names.count("Htheta") == 1


def test_inventory_excludes_templates_and_struct_members():
    inv = extract_inventory(_REF.read_text(), reference_scalar="double")
    sigs = " ".join(l.signature for l in inv)
    # The generic kAbs<T>/kLog/kSqrt/kConj/kPow/iszero templates auto-instantiate — not leaves.
    assert "template" not in sigs
    # Constants<T> members (_pi/_C/_ieps50/...) live inside the struct brace — never surface.
    assert "_pi" not in sigs
    assert "_ieps50" not in sigs
    assert "Constants" not in sigs


def test_inventory_source_order_kabs_before_its_callers():
    # Sign/Max/Min/Htheta call ql::kAbs — the shim must declare kAbs first (source order).
    inv = extract_inventory(_REF.read_text(), reference_scalar="double")
    order = [l.name for l in inv]
    first_kabs = order.index("kAbs")
    for caller in ("Sign", "Max", "Min", "Htheta"):
        assert first_kabs < order.index(caller)


# --------------------------------------------------------------------------- #
# sibling rendering (token rewrite, library-native binding)
# --------------------------------------------------------------------------- #

def test_render_shim_emits_float_siblings():
    shim = render_shim(_REF.read_text(), reference_scalar="double", target_scalar="float")
    # Real-precision overloads become float-typed, bodies preserved modulo the scalar token.
    assert "float kAbs(float const& x) { return Kokkos::abs(x); }" in shim
    assert "float kAbs(Kokkos::complex<float> const& x) { return Kokkos::abs(x); }" in shim
    assert "float Real(Kokkos::complex<float> const& x) { return x.real(); }" in shim
    assert "Kokkos::complex<float> Sign(Kokkos::complex<float> const& x)" in shim
    # Sign(double) returns int — the return type is NOT the scalar and must stay int.
    assert "int Sign(float const& x)" in shim
    # No double signature leaks into the float shim (would shadow / ODR-collide).
    assert "double kAbs(double" not in shim
    assert "complex<double>" not in shim


def test_render_shim_is_namespace_scoped_and_guarded():
    shim = render_shim(_REF.read_text(), reference_scalar="double", target_scalar="float")
    assert "#pragma once" in shim
    assert "namespace ql {" in shim
    assert "}  // namespace ql" in shim


def test_render_shim_parameterized_not_float_hardcoded():
    # STOP #SS: the generator rewrites reference->target tokens; it never keys on "float".
    # A synthetic library-native precision selects the same path with its own tokens.
    src = (
        "#pragma once\n"
        "namespace ql {\n"
        "    KOKKOS_INLINE_FUNCTION double kAbs(double const& x) { return Kokkos::abs(x); }\n"
        "}\n")
    shim = render_shim(src, reference_scalar="double", target_scalar="myhalf",
                       precision_label="half")
    assert "myhalf kAbs(myhalf const& x) { return Kokkos::abs(x); }" in shim
    assert "precision=half" in shim


def test_render_shim_empty_inventory_fails_loud():
    src = "#pragma once\nnamespace ql {\n}\n"
    with pytest.raises(ShimSynthError):
        render_shim(src, reference_scalar="double", target_scalar="float")


def test_missing_namespace_fails_loud():
    with pytest.raises(ShimSynthError):
        extract_inventory("int foo(double x){return 0;}", reference_scalar="double",
                          namespace="ql")


# --------------------------------------------------------------------------- #
# sha256 invalidation
# --------------------------------------------------------------------------- #

def test_inventory_sha_is_stamped_and_readable():
    shim = render_shim(_REF.read_text(), reference_scalar="double", target_scalar="float")
    inv = extract_inventory(_REF.read_text(), reference_scalar="double")
    assert read_embedded_sha(shim) == inventory_sha256(inv)


def test_sha_changes_when_a_leaf_body_changes():
    inv = extract_inventory(_REF.read_text(), reference_scalar="double")
    sha0 = inventory_sha256(inv)
    edited = _REF.read_text().replace("return 0.5 * (1 + ql::Sign(x));",
                                      "return 0.25 * (1 + ql::Sign(x));")
    sha1 = inventory_sha256(extract_inventory(edited, reference_scalar="double"))
    assert sha0 != sha1


def test_sha_order_independent():
    # Reordering the leaves in the header does not change the sha (sorted internally).
    inv = extract_inventory(_REF.read_text(), reference_scalar="double")
    reordered = list(reversed(inv))
    assert inventory_sha256(inv) == inventory_sha256(reordered)
