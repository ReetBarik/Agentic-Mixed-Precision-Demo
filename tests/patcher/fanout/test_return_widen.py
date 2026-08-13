"""Closure Subtask 2a — return-type widening emission machinery (design §7, rule c).

Unit + integration tests for the machinery Subtask 2b's rule-(c) algorithm will
consume:

* :class:`agents.patcher.fanout.ReturnWiden` — shape, merge (all cases), json
  round-trip;
* :func:`agents.integrator_base.boundary.widen_return_type_line` — the return-type
  token rewrite on synthetic frames and on the real ``Li2omx2`` fixture;
* :func:`agents.patcher.fanout.render_variant` — the return-widen emission pass,
  end-to-end on Li2omx2 and interacting with a :class:`ClosureDecl` on the same file;
* :func:`agents.patcher.chain_promote._attach_return_widens` — the wiring that binds a
  :class:`ReturnWiden` to its variant(s), including the nonexistent-variant fail-loud.

Self-contained: synthetic headers are written to tmp files; the real fixture is the
committed ``src/kokkosUtils.h`` (the same source conftest copies for the call-graph
fixtures).  No call graph / libclang is required for any of these.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from agents.integrator_base import boundary
from agents.integrator_base.boundary import BoundaryError, widen_return_type_line
from agents.patcher.chain_promote import _attach_return_widens
from agents.patcher.fanout import (
    ClosureDecl, FanoutError, ReturnWiden, VariantSpec, render_variant,
)

DD = "Kokkos::Experimental::DoubleDouble"
DDC = "DoubleDoubleComplex"

_KOKKOS_UTILS = Path(__file__).resolve().parents[3] / "src" / "kokkosUtils.h"

# Li2omx2 (the TScale-arg overload) in src/kokkosUtils.h:
#   691  template<typename TOutput, typename TMass, typename TScale>
#   692  KOKKOS_INLINE_FUNCTION TOutput Li2omx2(TScale const& v, ...) {
#   ...
#   711      return Li2omx2;                  # returned LOCAL, not the return type
#   712  }
_LI2_START = 691
_LI2_END = 712
_LI2_RET_LINE = 692


# --------------------------------------------------------------------------- #
# (A) ReturnWiden shape + serialisation
# --------------------------------------------------------------------------- #

def test_return_widen_shape():
    rw = ReturnWiden(return_line=692, orig_type="TOutput", dd_type=DDC,
                     function_name="Li2omx2_B10")
    assert (rw.return_line, rw.orig_type, rw.dd_type, rw.function_name) == (
        692, "TOutput", DDC, "Li2omx2_B10")


def _spec_with_rw(rw):
    return VariantSpec(variant_name="Li2omx2_B10", orig_name="Li2omx2",
                       file="kokkosUtils.h", orig_start=1, orig_end=3,
                       return_widen=rw)


def test_merge_both_none():
    a, b = _spec_with_rw(None), _spec_with_rw(None)
    a.merge(b)
    assert a.return_widen is None


def test_merge_one_none_one_set():
    rw = ReturnWiden(692, "TOutput", DDC, "Li2omx2_B10")
    a, b = _spec_with_rw(None), _spec_with_rw(rw)
    a.merge(b)
    assert a.return_widen == rw
    # symmetric: set + None keeps the set one
    c, d = _spec_with_rw(rw), _spec_with_rw(None)
    c.merge(d)
    assert c.return_widen == rw


def test_merge_both_set_equal_dedup():
    a = _spec_with_rw(ReturnWiden(692, "TOutput", DDC, "Li2omx2_B10"))
    b = _spec_with_rw(ReturnWiden(692, "TOutput", DDC, "Li2omx2_B10"))
    a.merge(b)
    assert a.return_widen == ReturnWiden(692, "TOutput", DDC, "Li2omx2_B10")


def test_merge_both_set_conflict_raises():
    a = _spec_with_rw(ReturnWiden(692, "TOutput", DDC, "Li2omx2_B10"))
    b = _spec_with_rw(ReturnWiden(692, "TOutput", DD, "Li2omx2_B10"))  # different dd_type
    with pytest.raises(FanoutError) as exc:
        a.merge(b)
    msg = str(exc.value)
    assert "Li2omx2_B10" in msg and DDC in msg and DD in msg


def test_json_round_trip_return_widen():
    spec = _spec_with_rw(ReturnWiden(692, "TOutput", DDC, "Li2omx2_B10"))
    d = spec.to_json()
    assert d["return_widen"] == {
        "return_line": 692, "orig_type": "TOutput", "dd_type": DDC,
        "function_name": "Li2omx2_B10"}
    back = VariantSpec.from_json(d)
    assert isinstance(back.return_widen, ReturnWiden)
    assert back.return_widen == spec.return_widen


def test_json_round_trip_none_serialises_null():
    spec = _spec_with_rw(None)
    d = spec.to_json()
    assert d["return_widen"] is None
    back = VariantSpec.from_json(d)
    assert back.return_widen is None


def test_from_json_absent_field_defaults_none():
    # A pre-2a manifest has no return_widen key.
    d = {"variant_name": "v", "orig_name": "f", "file": "x.h",
         "orig_start": 1, "orig_end": 3}
    assert VariantSpec.from_json(d).return_widen is None


# --------------------------------------------------------------------------- #
# (B) widen_return_type_line on synthetic frames
# --------------------------------------------------------------------------- #

def test_single_line_template_return():
    src = "    KOKKOS_INLINE_FUNCTION TOutput f(TScale const& v) {\n        return v;\n    }"
    out = widen_return_type_line(src, return_line=1, orig_type="TOutput",
                                 dd_type=DDC, function_name="f_B10")
    first = out.split("\n")[0]
    assert first == "    KOKKOS_INLINE_FUNCTION DoubleDoubleComplex f(TScale const& v) {"
    # macro keyword and body untouched
    assert "return v;" in out


def test_multi_line_template_return():
    # return type on line 1, function name on line 2 (a long template return type).
    src = ("    typename std::conditional<B, TOutput, TScale>::type\n"
           "    f(TScale const& v) {\n"
           "        return v;\n"
           "    }")
    out = widen_return_type_line(src, return_line=1, orig_type="TOutput",
                                 dd_type=DDC, function_name="f_B10")
    lines = out.split("\n")
    assert lines[0] == "    typename std::conditional<B, DoubleDoubleComplex, TScale>::type"
    assert lines[1] == "    f(TScale const& v) {"


def test_const_reference_qualified_return():
    src = "    const TOutput& f(int i) {\n        return r;\n    }"
    out = widen_return_type_line(src, return_line=1, orig_type="TOutput",
                                 dd_type=DDC, function_name="f_B10")
    assert out.split("\n")[0] == "    const DoubleDoubleComplex& f(int i) {"


def test_static_qualified_return():
    src = "    static inline TOutput f() {\n        return r;\n    }"
    out = widen_return_type_line(src, return_line=1, orig_type="TOutput",
                                 dd_type=DDC, function_name="f_B10")
    assert out.split("\n")[0] == "    static inline DoubleDoubleComplex f() {"


def test_namespaced_return_last_segment():
    # std::complex<TScale> — orig_type is the last segment 'complex'.
    src = "    std::complex<TScale> f(int i) {\n        return r;\n    }"
    out = widen_return_type_line(src, return_line=1, orig_type="complex",
                                 dd_type=DDC, function_name="f_B10")
    assert out.split("\n")[0] == "    std::DoubleDoubleComplex<TScale> f(int i) {"


def test_orig_type_not_found_raises_with_diagnostic():
    src = "    TScale f(int i) {\n        return r;\n    }"
    with pytest.raises(BoundaryError) as exc:
        widen_return_type_line(src, return_line=1, orig_type="TOutput",
                               dd_type=DDC, function_name="f_B10")
    msg = str(exc.value)
    assert "TOutput" in msg and "f_B10" in msg and "line 1" in msg


def test_idempotent_reapply():
    src = "    TOutput f(int i) {\n        return r;\n    }"
    once = widen_return_type_line(src, return_line=1, orig_type="TOutput",
                                  dd_type=DDC, function_name="f_B10")
    twice = widen_return_type_line(once, return_line=1, orig_type="TOutput",
                                   dd_type=DDC, function_name="f_B10")
    assert once == twice
    assert once.split("\n")[0] == "    DoubleDoubleComplex f(int i) {"


# --------------------------------------------------------------------------- #
# (C) widen_return_type_line on the real Li2omx2 fixture
# --------------------------------------------------------------------------- #

def _li2_source() -> tuple[str, list[str]]:
    text = _KOKKOS_UTILS.read_text(encoding="utf-8")
    return text, text.split("\n")


def test_real_li2omx2_return_type_widened_body_untouched():
    text, lines = _li2_source()
    out = widen_return_type_line(text, return_line=_LI2_RET_LINE,
                                 orig_type="TOutput", dd_type=DDC,
                                 function_name="Li2omx2_B10")
    out_lines = out.split("\n")
    # signature line: TOutput -> DoubleDoubleComplex, everything else on the line preserved
    assert out_lines[_LI2_RET_LINE - 1] == lines[_LI2_RET_LINE - 1].replace(
        "TOutput Li2omx2", "DoubleDoubleComplex Li2omx2", 1)
    assert "KOKKOS_INLINE_FUNCTION DoubleDoubleComplex Li2omx2(" in out_lines[_LI2_RET_LINE - 1]
    # the returned LOCAL statement `return Li2omx2;` (:711) is a local, not the return
    # type declaration — untouched.
    assert out_lines[711 - 1] == lines[711 - 1]
    assert "return Li2omx2;" in out_lines[711 - 1]
    # exactly one line changed
    diff = [i for i in range(len(lines)) if lines[i] != out_lines[i]]
    assert diff == [_LI2_RET_LINE - 1]


def test_real_li2omx2_shared_original_not_edited():
    before = _KOKKOS_UTILS.read_text(encoding="utf-8")
    _ = widen_return_type_line(before, return_line=_LI2_RET_LINE,
                               orig_type="TOutput", dd_type=DDC,
                               function_name="Li2omx2_B10")
    # widen_return_type_line is pure (returns a new string) — the fixture on disk is
    # provably not touched.
    after = _KOKKOS_UTILS.read_text(encoding="utf-8")
    assert before == after


# --------------------------------------------------------------------------- #
# (D) render_variant end-to-end on Li2omx2
# --------------------------------------------------------------------------- #

def _li2_spec(**kw) -> VariantSpec:
    return VariantSpec(variant_name="Li2omx2_B10", orig_name="Li2omx2",
                       file=str(_KOKKOS_UTILS), orig_start=_LI2_START,
                       orig_end=_LI2_END, **kw)


def test_render_variant_return_widen_only_on_li2omx2():
    # A ReturnWiden and no other edits: the produced variant has
    # `DoubleDoubleComplex Li2omx2_B10(...)` in the signature; everything else is the original
    # body verbatim (modulo the orig_name -> variant_name rename the rewriter does).
    spec = _li2_spec(return_widen=ReturnWiden(
        return_line=_LI2_RET_LINE, orig_type="TOutput", dd_type=DDC,
        function_name="Li2omx2_B10"))
    out = render_variant(spec)
    out_lines = out.split("\n")

    # baseline: render with NO return_widen (rename only) to isolate the one change.
    base = render_variant(_li2_spec()).split("\n")
    assert len(out_lines) == len(base)
    diff = [i for i in range(len(base)) if base[i] != out_lines[i]]
    # exactly the signature line differs, and it differs only by TOutput->DoubleDoubleComplex.
    assert len(diff) == 1
    d = diff[0]
    assert out_lines[d] == base[d].replace("TOutput Li2omx2_B10", "DoubleDoubleComplex Li2omx2_B10", 1)
    assert "DoubleDoubleComplex Li2omx2_B10(" in out_lines[d]
    # the returned local statement is byte-identical to the rename-only baseline.
    ret_idx = next(i for i, ln in enumerate(base) if "return Li2omx2;" in ln)
    assert out_lines[ret_idx] == base[ret_idx]


def test_render_variant_return_widen_and_closure_decl_no_drift(tmp_path):
    # A ReturnWiden AND a ClosureDecl on the same file: both edits land correctly and
    # no line-number drift corrupts either.  Synthetic so the decl line is a known
    # carrier.
    text = """\
#pragma once
template<class T>
T f(T v) {
    T carry;
    carry = v + v;
    T other = carry * v;
    return other;
}
"""
    p = tmp_path / "h.h"
    p.write_text(text, encoding="utf-8")
    spec = VariantSpec(
        variant_name="f_B10", orig_name="f", file=str(p),
        orig_start=2, orig_end=8,
        closure_decls=[ClosureDecl(decl_line=4, orig_type="T", dd_type=DD, name="carry")],
        return_widen=ReturnWiden(return_line=3, orig_type="T", dd_type=DD,
                                 function_name="f_B10"))
    out = render_variant(spec)
    lines = out.split("\n")
    # return type widened on the signature line (line 3 -> index 1 of the copied slice)
    assert f"{DD} f_B10(T v) {{" in out
    # carrier decl widened
    assert f"{DD} carry;" in out
    # both landed; the parameter `T v` on the signature keeps its type (only the
    # leading return-type token changed).
    assert "f_B10(T v)" in out


# --------------------------------------------------------------------------- #
# (E) compat: empty return_widen renders byte-identically
# --------------------------------------------------------------------------- #

def test_render_variant_no_return_widen_unchanged(tmp_path):
    text = """\
#pragma once
template<class T>
T f(T v) {
    T a = v + v;
    return a;
}
"""
    p = tmp_path / "h.h"
    p.write_text(text, encoding="utf-8")
    spec = VariantSpec(variant_name="f_B10", orig_name="f", file=str(p),
                       orig_start=2, orig_end=6)
    out = render_variant(spec)
    # return type NOT widened (field is None); the rename is the only transform.
    assert "T f_B10(T v)" in out
    assert DD not in out and DDC not in out


# --------------------------------------------------------------------------- #
# (F) wiring: _attach_return_widens
# --------------------------------------------------------------------------- #

def _mk_specs():
    # Rule (c) records ReturnWiden at FRAME level: function_name = the ORIGINAL name
    # (Li2omx2), return_line = the signature line.  The attach binds by orig_name +
    # line-containment, so a single record rides EVERY per-caller-path variant of that
    # function (Li2omx2_B10_B1m_B10 via one path, Li2omx2_B13_... via another).
    a = VariantSpec(variant_name="Li2omx2_B10_B1m_B10", orig_name="Li2omx2",
                    file="kokkosUtils.h", orig_start=688, orig_end=708)
    # same original reached via a second caller path -> a distinct variant name and
    # spec object, placed under a different file key to exercise the cross-file attach.
    b = VariantSpec(variant_name="Li2omx2_B13_B2ma_B2m_B10", orig_name="Li2omx2",
                    file="kokkosUtils.h", orig_start=688, orig_end=708)
    return ({"kokkosUtils.h": {"Li2omx2_B10_B1m_B10": a},
             "other.h": {"Li2omx2_B13_B2ma_B2m_B10": b}}, a, b)


def test_attach_empty_is_noop():
    specs, a, b = _mk_specs()
    _attach_return_widens([], specs)
    assert a.return_widen is None and b.return_widen is None


def test_attach_binds_to_every_matching_variant():
    specs, a, b = _mk_specs()
    rw = ReturnWiden(688, "TOutput", DDC, "Li2omx2")   # frame-level: orig name
    _attach_return_widens([rw], specs)
    assert a.return_widen == rw and b.return_widen == rw


def test_attach_nonexistent_variant_raises():
    specs, _a, _b = _mk_specs()
    rw = ReturnWiden(688, "TOutput", DDC, "NoSuchFn")
    with pytest.raises(FanoutError) as exc:
        _attach_return_widens([rw], specs)
    assert "NoSuchFn" in str(exc.value)


def test_attach_line_outside_extent_raises():
    # A record whose return_line is outside every candidate variant's original extent
    # is a wiring bug (STOP #5) — the closure demanded a widen on a line no variant of
    # that function clones.
    specs, _a, _b = _mk_specs()
    rw = ReturnWiden(9999, "TOutput", DDC, "Li2omx2")
    with pytest.raises(FanoutError):
        _attach_return_widens([rw], specs)


def test_attach_conflict_raises():
    specs, _a, _b = _mk_specs()
    rw1 = ReturnWiden(688, "TOutput", DDC, "Li2omx2")
    rw2 = ReturnWiden(688, "TOutput", DD, "Li2omx2")   # different dd_type
    with pytest.raises(FanoutError):
        _attach_return_widens([rw1, rw2], specs)
