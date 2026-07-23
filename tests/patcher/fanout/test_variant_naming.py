"""Variant naming — determinism + collision-freeness (pure, no libclang)."""

from __future__ import annotations

import pytest

from agents.patcher.variant_naming import (
    VariantNameError, assert_no_collisions, variant_name, variant_names_for_path,
)


def test_bottom_up_name():
    assert variant_name("f", ["g", "h"], "B1") == "f_g_h_B1"


def test_first_below_root_has_only_integral_suffix():
    assert variant_name("h", [], "B1") == "h_B1"


def test_names_for_path():
    m = variant_names_for_path(["entry", "h", "g", "f"], "B1")
    assert m == {"h": "h_B1", "g": "g_h_B1", "f": "f_g_h_B1"}
    # the root (entry) is never renamed
    assert "entry" not in m


def test_names_for_path_root_only():
    assert variant_names_for_path(["entry"], "B1") == {}


def test_determinism():
    a = variant_names_for_path(["entry", "h", "g", "f"], "B1")
    b = variant_names_for_path(["entry", "h", "g", "f"], "B1")
    assert a == b


def test_distinct_paths_distinct_names():
    # two paths to f through different intermediates -> distinct names, no collision
    p1 = variant_names_for_path(["entry", "h", "g", "f"], "B1")
    p2 = variant_names_for_path(["entry", "h2", "g", "f"], "B1")
    assert p1["f"] == "f_g_h_B1"
    assert p2["f"] == "f_g_h2_B1"
    assert_no_collisions([p1, p2])          # must not raise


def test_same_function_same_name_is_not_a_collision():
    # the SAME original function keeping the SAME variant name across two intents
    # (shared prefix / byte-identical over-generation) is allowed.
    m = variant_names_for_path(["entry", "h", "g", "f"], "B1")
    assert_no_collisions([m, dict(m)])      # must not raise


def test_collision_detected():
    # fabricate a genuine collision: same variant name for two different originals
    bad = [{"f": "X_B1"}, {"g": "X_B1"}]
    with pytest.raises(VariantNameError):
        assert_no_collisions(bad)


def test_illegal_identifier_rejected():
    with pytest.raises(VariantNameError):
        variant_name("f", ["bad-name"], "B1")
    with pytest.raises(VariantNameError):
        variant_name("f", ["g"], "B1 ")
