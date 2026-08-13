"""Unit tests for agents.patcher.shim_normalise (Subtask 3 deterministic sweep).

The three defect classes ("Blocker B", design §8 point 1):
  N1 — redeclaration of an already-promoted local (the T__ff class);
  N2 — malformed / redundant unary operator+;
  N3 — decimal-literal extended-scalar constructor (_ieps50 residual).

Each transform is checked for a POSITIVE rewrite, a NEGATIVE non-rewrite,
idempotence (twice == once), and semantic-nullity on clean input.
"""

from __future__ import annotations

import struct

from agents.patcher import shim_normalise as sn

_DD = "Kokkos::Experimental::DoubleDouble"
_FF = "Kokkos::Experimental::FloatFloat"


def _dbits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


# --------------------------------------------------------------------------- #
# N1 — drop same-scope redeclaration of a promoted local
# --------------------------------------------------------------------------- #

def test_n1_drops_same_scope_redeclaration():
    src = (
        "void f() {\n"
        f"    {_DD} T__ff = {_DD}(T);  // Rule R1\n"
        "    use(T__ff);\n"
        f"    {_DD} T__ff = {_DD}(T);  // Rule R1\n"
        "    use(T__ff);\n"
        "}\n"
    )
    out = sn.normalise_source(src)
    # first decl kept, second demoted to an assignment (no type prefix)
    assert out.count(f"{_DD} T__ff =") == 1
    assert "    T__ff = Kokkos::Experimental::DoubleDouble(T);  // Rule R1" in out


def test_n1_keeps_deeper_scope_shadow():
    # a re-declaration at a DEEPER brace scope is legal C++ shadowing — leave intact
    src = (
        "void f() {\n"
        f"    {_DD} x = {_DD}(a);\n"
        "    {\n"
        f"        {_DD} x = {_DD}(b);\n"   # legal shadow, deeper scope
        "    }\n"
        "}\n"
    )
    assert sn.normalise_source(src) == src


def test_n1_distinct_names_untouched():
    src = (
        "void f() {\n"
        f"    {_DD} a = {_DD}(x);\n"
        f"    {_DD} b = {_DD}(y);\n"
        "}\n"
    )
    assert sn.normalise_source(src) == src


def test_n1_idempotent():
    src = (
        "void f() {\n"
        f"    {_DD} T__ff = {_DD}(T);\n"
        f"    {_DD} T__ff = {_DD}(T);\n"
        "}\n"
    )
    once = sn.normalise_source(src)
    assert sn.normalise_source(once) == once


def test_n1_reopened_sibling_scope_redeclares_legally():
    # two SIBLING blocks each declaring the same name is legal (separate scopes)
    src = (
        "void f() {\n"
        "    {\n"
        f"        {_DD} x = {_DD}(a);\n"
        "    }\n"
        "    {\n"
        f"        {_DD} x = {_DD}(b);\n"
        "    }\n"
        "}\n"
    )
    assert sn.normalise_source(src) == src


# --------------------------------------------------------------------------- #
# N2 — redundant unary operator+
# --------------------------------------------------------------------------- #

def test_n2_removes_unary_plus_after_assignment():
    assert sn.normalise_source("x = + y;\n") == "x =  y;\n"


def test_n2_removes_unary_plus_after_return():
    assert sn.normalise_source("return + expr;\n") == "return  expr;\n"


def test_n2_leaves_binary_plus():
    assert sn.normalise_source("a = b + c;\n") == "a = b + c;\n"


def test_n2_leaves_increment_and_plus_assign():
    for s in ("i++;\n", "x += 3;\n", "z = ++k;\n", "a = b++ + c;\n"):
        assert sn.normalise_source(s) == s


def test_n2_idempotent():
    once = sn.normalise_source("y = + x;\n")
    assert once == "y =  x;\n"
    assert sn.normalise_source(once) == once


# --------------------------------------------------------------------------- #
# N3 — decimal-literal extended-scalar constructor
# --------------------------------------------------------------------------- #

def test_n3_rewrites_literal_ctor_to_bit_pair():
    out = sn.normalise_source(f"{_DD} e = {_DD}(1e-50);\n")
    hi = _dbits(1e-50)
    assert f"Kokkos::Experimental::DoubleDouble::from_bits(0x{hi:016x}ULL, 0x0000000000000000ULL)" in out
    assert "(1e-50)" not in out


def test_n3_is_bit_identical_value():
    # DoubleDouble(h) == {h, 0}; DoubleDouble::from_bits(bits(h), 0) reconstructs the same value
    out = sn.normalise_source(f"{_DD} e = {_DD}(0.125);\n")
    hi = _dbits(0.125)
    assert f"0x{hi:016x}ULL, 0x0000000000000000ULL" in out


def test_n3_leaves_identifier_ctor():
    # ctor over an identifier / expression is a real promotion, not a literal — keep
    src = f"{_DD} a = {_DD}(x);\n"
    assert sn.normalise_source(src) == src


def test_n3_leaves_nonliteral_expression_ctor():
    src = f"{_DD} a = {_DD}(x + y);\n"
    assert sn.normalise_source(src) == src


def test_n3_ff_family_uses_make_ff():
    out = sn.normalise_source(f"{_FF} e = {_FF}(0.5);\n")
    assert "Kokkos::Experimental::FloatFloat::from_bits(0x" in out


def test_n3_idempotent():
    once = sn.normalise_source(f"{_DD} e = {_DD}(1e-50);\n")
    assert sn.normalise_source(once) == once


# --------------------------------------------------------------------------- #
# whole-file semantic nullity on clean input
# --------------------------------------------------------------------------- #

def test_clean_shim_unchanged():
    clean = (
        "#pragma once\n"
        "namespace ql {\n"
        f"inline {_DD} kLog({_DD} x) {{\n"
        "    return Kokkos::Experimental::log(x);\n"
        "}\n"
        f"{_DD} a = {_DD}(x);\n"
        "auto s = a + b;\n"
        "}\n"
    )
    assert sn.normalise_source(clean) == clean


def test_all_three_together_idempotent():
    src = (
        "void f() {\n"
        f"    {_DD} T__ff = {_DD}(T);\n"
        f"    {_DD} T__ff = {_DD}(T);\n"
        f"    {_DD} e = {_DD}(1e-50);\n"
        "    x = + T__ff;\n"
        "}\n"
    )
    once = sn.normalise_source(src)
    assert sn.normalise_source(once) == once
    # all three fired
    assert once.count(f"{_DD} T__ff =") == 1
    assert "DoubleDouble::from_bits(0x" in once
    assert "= + " not in once


# --------------------------------------------------------------------------- #
# normalise_file
# --------------------------------------------------------------------------- #

def test_normalise_file_reports_change(tmp_path):
    p = tmp_path / "shim.h"
    p.write_text(f"{_DD} e = {_DD}(1e-50);\n", encoding="utf-8")
    assert sn.normalise_file(p) is True
    assert "DoubleDouble::from_bits(0x" in p.read_text(encoding="utf-8")


def test_normalise_file_noop_on_clean(tmp_path):
    p = tmp_path / "shim.h"
    clean = f"{_DD} a = {_DD}(x);\n"
    p.write_text(clean, encoding="utf-8")
    assert sn.normalise_file(p) is False
    assert p.read_text(encoding="utf-8") == clean
