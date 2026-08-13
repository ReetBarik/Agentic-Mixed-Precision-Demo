"""Unit tests for the regional SOURCE_HASH key (cache.compute_region_hash)."""

from __future__ import annotations

from agents.integrator_base import cache

_SRC = "double r = a + b;\n"
_RULES = "ruleset v1"
_SCALAR = "Kokkos::Experimental::FloatFloat"


def test_stable_for_identical_inputs():
    a = cache.compute_region_hash(_SRC, _RULES, _SCALAR, ["r"])
    b = cache.compute_region_hash(_SRC, _RULES, _SCALAR, ["r"])
    assert a == b
    assert len(a) == 64


def test_writes_order_independent():
    a = cache.compute_region_hash(_SRC, _RULES, _SCALAR, ["r", "s"])
    b = cache.compute_region_hash(_SRC, _RULES, _SCALAR, ["s", "r"])
    assert a == b


def test_varies_with_region_source():
    a = cache.compute_region_hash(_SRC, _RULES, _SCALAR, ["r"])
    b = cache.compute_region_hash("double r = a - b;\n", _RULES, _SCALAR, ["r"])
    assert a != b


def test_varies_with_ruleset_bytes():
    a = cache.compute_region_hash(_SRC, _RULES, _SCALAR, ["r"])
    b = cache.compute_region_hash(_SRC, "ruleset v2", _SCALAR, ["r"])
    assert a != b


def test_varies_with_scalar_type():
    a = cache.compute_region_hash(_SRC, _RULES, _SCALAR, ["r"])
    b = cache.compute_region_hash(_SRC, _RULES, "Kokkos::Experimental::DoubleDouble", ["r"])
    assert a != b


def test_varies_with_write_set():
    a = cache.compute_region_hash(_SRC, _RULES, _SCALAR, ["r"])
    b = cache.compute_region_hash(_SRC, _RULES, _SCALAR, ["r", "s"])
    assert a != b
