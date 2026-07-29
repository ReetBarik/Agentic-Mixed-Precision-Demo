"""Deliverable 4 — app-output boundary narrowing reuses the acc1482 transform (STOP #TT).

The flip driver's Printer is the app-output boundary: it narrows the dd-computed result
to caller precision via the SHARED ``narrow_two_limb_scalar`` primitive (the same one the
element-promotion designed exit uses), emitting a single caller-precision token so the
Validator scorer ingests a flip candidate exactly like a double candidate.
"""

from __future__ import annotations

from agents.integrator_base import boundary
from agents.integrator_base.boundary import narrow_two_limb_scalar
from agents.patcher.precision_flip import TargetPrecision
from agents.patcher.tu_emit import render_group_driver


# --------------------------------------------------------------------------- #
# the shared reconstruction primitive
# --------------------------------------------------------------------------- #

def test_shared_primitive_two_limb_idiom():
    assert (narrow_two_limb_scalar("v", "double", two_limb=True)
            == "static_cast<double>(v.hi) + static_cast<double>(v.lo)")


def test_shared_primitive_native_is_plain_cast():
    assert narrow_two_limb_scalar("v", "float", two_limb=False) == "static_cast<float>(v)"


def test_element_demote_delegates_to_shared_primitive():
    # _demote_expr (acc1482 designed-exit scalar demote) must produce the identical
    # reconstruction the flip boundary uses — proving one source of truth (STOP #TT).
    demoted = boundary._demote_expr("w", "double", two_limb=True)
    frag = narrow_two_limb_scalar("w__ext.hi", "double", two_limb=True)  # sanity of pieces
    assert "static_cast<double>(w__ext.hi) + static_cast<double>(w__ext.lo)" == demoted


def test_complex_demote_reconstructs_both_components_via_shared():
    out = boundary._demote_complex_expr("w", "TOutput", "double", two_limb=True)
    assert narrow_two_limb_scalar("w__ext.re", "double", True) in out
    assert narrow_two_limb_scalar("w__ext.im", "double", True) in out


# --------------------------------------------------------------------------- #
# the flip driver printer IS the boundary
# --------------------------------------------------------------------------- #

def test_flip_printer_narrows_via_shared_primitive():
    d = render_group_driver("box/B1m.h", TargetPrecision.DD)
    # The emitted printer body must contain the shared reconstruction (dd -> double),
    # NOT the oracle's hi|lo dump (which keeps full dd and would false-positive).
    recon = narrow_two_limb_scalar("v", "double", two_limb=True)
    assert f"dhex({recon})" in d


def test_flip_printer_emits_single_token_not_hi_lo():
    d = render_group_driver("box/B1m.h", TargetPrecision.DD)
    printer = d.split("namespace ql_app")[1].split("int main")[0]
    # No hi|lo separator emission in the flip printer (that is the oracle's format).
    assert "'|'" not in printer
    assert "out += dhex(v.hi)" not in printer


def test_flip_printer_targets_caller_type_double():
    d = render_group_driver("box/B1m.h", TargetPrecision.DD)
    assert "static_cast<double>(v.hi)" in d
