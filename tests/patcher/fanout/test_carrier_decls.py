"""Blocker A, Subtask 4 — VariantSpec carrier decl-widening (design §7).

Unit tests for the emission layer that widens a chain *carrier*'s declaration to
the chain's dd type as part of :func:`agents.patcher.fanout.render_variant`.  A
carrier is a variable declared outside the chain's line set but written by one
chain link and read by another; leaving its decl at caller precision truncates the
widened value at the interior write (the 2d-B ``chain_write_truncation`` bug).

These tests are self-contained: a synthetic header is written to a tmp file and a
:class:`VariantSpec` with synthetic ``closure_decls`` is rendered directly, so no
call graph / libclang is required.  They cover the single-declarator rewrite, the
multi-declarator sibling inheritance (§2 conservative policy), the shared
descending-line-order pass with promotes, and json/merge round-trip of the field.
"""

from __future__ import annotations

from pathlib import Path

from agents.patcher.fanout import (
    ClosureDecl, Promote, VariantSpec, render_variant,
)

DD = "Kokkos::Experimental::DoubleDouble"

# A function whose body declares carriers outside a promotable region.  Lines are
# 1-based file-absolute; the function spans lines 2..12 here.
HEADER = """\
#pragma once
template<class TMass>
TMass ddilog(TMass T) {
    TMass Y, S, A;
    const TMass one = TMass(1);
    Y = (TMass(-1) - T) / T;
    A = TMass(2) + A * T;
    S = Y + one;
    const TMass H = Y + Y - one;
    return -(S - H + A);
}
"""


def _write(tmp_path: Path, text: str = HEADER) -> Path:
    p = tmp_path / "kokkosUtils.h"
    p.write_text(text, encoding="utf-8")
    return p


def _spec(path: Path, **kw) -> VariantSpec:
    # ddilog spans lines 2..11 (template line .. closing brace).
    return VariantSpec(
        variant_name="ddilog_B10", orig_name="ddilog", file=str(path),
        orig_start=2, orig_end=11, **kw)


# --------------------------------------------------------------------------- #
# single-declarator carrier
# --------------------------------------------------------------------------- #

def test_single_declarator_type_token_rewritten(tmp_path):
    # A carrier declared alone on its line: the type token is swapped for dd.
    text = """\
#pragma once
template<class TMass>
TMass f(TMass T) {
    TMass Y;
    Y = T + T;
    return Y;
}
"""
    p = _write(tmp_path, text)
    spec = VariantSpec(variant_name="f_B10", orig_name="f", file=str(p),
                       orig_start=2, orig_end=7,
                       closure_decls=[ClosureDecl(decl_line=4, orig_type="TMass",
                                                  dd_type=DD, name="Y")])
    out = render_variant(spec)
    assert f"{DD} Y;" in out
    # untouched TMass tokens elsewhere (the parameter, the return type) remain
    assert "TMass T" in out


# --------------------------------------------------------------------------- #
# multi-declarator sibling inheritance (§2)
# --------------------------------------------------------------------------- #

def test_multi_declarator_leading_type_widens_all_siblings(tmp_path):
    p = _write(tmp_path)
    # Only Y and A are strict carriers, but widening the leading type token of
    # ``TMass Y, S, A;`` widens S too (conservative, safe — a wider sibling never
    # truncates).
    spec = _spec(p, closure_decls=[
        ClosureDecl(decl_line=4, orig_type="TMass", dd_type=DD, name="Y"),
    ])
    out = render_variant(spec)
    assert f"{DD} Y, S, A;" in out
    # the const decl on the next line keeps its caller type (not a carrier record)
    assert "const TMass one = TMass(1);" in out


def test_no_matching_type_leaves_line_verbatim(tmp_path):
    # A ClosureDecl whose orig_type does not match the decl line is a no-op (defensive
    # idempotence: a re-render of an already-widened decl, or a stale coordinate).
    p = _write(tmp_path)
    spec = _spec(p, closure_decls=[
        ClosureDecl(decl_line=4, orig_type="TOutput", dd_type=DD, name="Y"),
    ])
    out = render_variant(spec)
    assert "TMass Y, S, A;" in out
    assert DD not in out


# --------------------------------------------------------------------------- #
# shared descending-line-order pass (promotes + carrier decls)
# --------------------------------------------------------------------------- #

def test_decl_widen_and_promote_apply_in_correct_order(tmp_path):
    # A carrier decl (line 4) ABOVE a promoted region (lines 6..8, a block that the
    # promotion replaces with a different number of lines) must still be located
    # correctly: because the region (higher line) is edited first, the decl line
    # never shifts.  Assert both edits land.
    p = _write(tmp_path)
    spec = _spec(p,
                 closure_decls=[ClosureDecl(decl_line=4, orig_type="TMass",
                                            dd_type=DD, name="Y")],
                 promotes=[Promote(region_start=6, region_end=8,
                                   reads=["T", "A"], writes=["Y", "A", "S"],
                                   scalar_type=DD, two_limb=True,
                                   caller_type="TMass")])
    out = render_variant(spec)
    # carrier decl widened
    assert f"{DD} Y, S, A;" in out
    # region body was retyped (the promotion introduced dd typed locals / casts);
    # the const decl below the region is untouched.
    assert "const TMass H = Y + Y - one;" in out
    assert out.count(f"{DD} Y, S, A;") == 1


def test_multiple_closure_decls_all_widen(tmp_path):
    # Two carrier decls on different lines both widen; descending order preserves both.
    text = """\
#pragma once
template<class TMass>
TMass f(TMass T) {
    TMass a;
    TMass b, c;
    a = T + T;
    b = a + T;
    c = b - a;
    return c;
}
"""
    p = _write(tmp_path, text)
    spec = VariantSpec(variant_name="f_B10", orig_name="f", file=str(p),
                       orig_start=2, orig_end=10, closure_decls=[
        ClosureDecl(decl_line=4, orig_type="TMass", dd_type=DD, name="a"),
        ClosureDecl(decl_line=5, orig_type="TMass", dd_type=DD, name="b"),
    ])
    out = render_variant(spec)
    assert f"{DD} a;" in out
    assert f"{DD} b, c;" in out


# --------------------------------------------------------------------------- #
# json / merge round-trip
# --------------------------------------------------------------------------- #

def test_json_round_trip_preserves_closure_decls(tmp_path):
    p = _write(tmp_path)
    spec = _spec(p, closure_decls=[
        ClosureDecl(decl_line=4, orig_type="TMass", dd_type=DD, name="Y"),
        ClosureDecl(decl_line=9, orig_type="TMass", dd_type=DD, name="H"),
    ])
    d = spec.to_json()
    back = VariantSpec.from_json(d)
    assert back.closure_decls == spec.closure_decls
    assert all(isinstance(c, ClosureDecl) for c in back.closure_decls)


def test_from_json_defaults_empty_when_field_absent(tmp_path):
    # A pre-Blocker-A manifest has no closure_decls key; from_json must default it.
    d = {"variant_name": "v", "orig_name": "f", "file": "x.h",
         "orig_start": 1, "orig_end": 3}
    back = VariantSpec.from_json(d)
    assert back.closure_decls == []


def test_merge_unions_closure_decls_dedup(tmp_path):
    p = _write(tmp_path)
    a = _spec(p, closure_decls=[
        ClosureDecl(decl_line=4, orig_type="TMass", dd_type=DD, name="Y")])
    b = _spec(p, closure_decls=[
        ClosureDecl(decl_line=4, orig_type="TMass", dd_type=DD, name="Y"),  # dup
        ClosureDecl(decl_line=9, orig_type="TMass", dd_type=DD, name="H")])
    a.merge(b)
    lines = sorted(c.decl_line for c in a.closure_decls)
    assert lines == [4, 9]  # dup collapsed, new one added


def test_closure_decls_default_empty_no_regression(tmp_path):
    # A spec with no closure_decls renders exactly as before (the field is inert).
    p = _write(tmp_path)
    spec = _spec(p)
    out = render_variant(spec)
    assert "TMass Y, S, A;" in out
    assert DD not in out
