"""Deliverable 2 — per-integral TU emission + precision-parameterized wrapper.

Unit-level: renders + group discovery + snapshot guard + precision parameterization.
The end-to-end g++ compile of the emitted TU against a snapshot clone is exercised by
the harness (it needs the module env + Kokkos); here we assert the *shape* the compile
depends on (fork-shape wrapper arm, pruned-group include, profile-driven template args).
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from agents.patcher.precision_flip import TargetPrecision
from agents.patcher.tu_emit import (
    PROFILES, TUEmitError, emit_flip_tu, group_header_for_files,
    profile_for, render_group_driver, render_wrapper)


# --------------------------------------------------------------------------- #
# wrapper generator
# --------------------------------------------------------------------------- #

def test_wrapper_has_dd_arm_the_snapshot_lacks():
    w = render_wrapper(TargetPrecision.DD)
    # The load-bearing §5.5 arm: USE_DD_COMPLEX -> kokkosMaths_dd.h.
    assert "#if defined(USE_DD_COMPLEX)" in w
    assert '#include "kokkosMaths_dd.h"' in w
    # Preserves the quad arm + double default (fork shape).
    assert "USE_QUAD_COMPLEX" in w
    assert '#include "kokkosMaths.h"' in w
    assert w.count("#pragma once") == 1


def test_wrapper_ladder_is_wellformed():
    w = render_wrapper(TargetPrecision.DD)
    # Exactly one opening #if, and the dd arm precedes the quad arm as #elif.
    assert w.index("#if defined(USE_DD_COMPLEX)") < w.index("#elif defined(USE_QUAD_COMPLEX)")
    # Balanced #if/#endif (2 opens: dd-ladder head + inner CUDA guard -> matches design).
    assert w.count("#endif") == 2


def test_wrapper_has_ff_arm_after_enrichment():
    # The ff enrichment (kokkosMaths_ff.h) makes ff a static-header ladder arm, table-driven
    # alongside dd (STOP #SS: selected by the profile's maths_header, not a precision name).
    w = render_wrapper(TargetPrecision.DD)
    assert "#elif defined(USE_FF_COMPLEX)" in w
    assert '#include "kokkosMaths_ff.h"' in w
    # ff arm sits in the ladder between the dd arm and the quad arm.
    assert (w.index("#if defined(USE_DD_COMPLEX)")
            < w.index("#elif defined(USE_FF_COMPLEX)")
            < w.index("#elif defined(USE_QUAD_COMPLEX)"))


# --------------------------------------------------------------------------- #
# group driver generator
# --------------------------------------------------------------------------- #

def test_group_driver_includes_group_header_not_meta_header():
    d = render_group_driver("box/B1m.h", TargetPrecision.DD)
    assert '#include "box/B1m.h"' in d
    # Must NOT pull the meta-header (that would define QCDLOOP_BOX_FULL_DISPATCH and
    # instantiate every group -> loses per-group isolation).
    assert '#include "boxGPU.h"' not in d
    # The full-dispatch macro must never be DEFINED in the driver (a comment naming it
    # is fine); defining it would activate the meta-dispatch and lose isolation.
    assert "#define QCDLOOP_BOX_FULL_DISPATCH" not in d


def test_group_driver_dd_template_args_and_define():
    d = render_group_driver("box/B2m.h", TargetPrecision.DD)
    assert "#define USE_DD_COMPLEX" in d
    assert "ql::ddfun::ddcomplex" in d
    assert "ql::ddfun::ddouble" in d
    assert "DDPrinter" in d
    assert "run_app<" in d


def test_group_driver_reuses_shared_recipe():
    d = render_group_driver("box/B1m.h", TargetPrecision.DD)
    assert '#include "boxGPU_app_recipes.hpp"' in d


# --------------------------------------------------------------------------- #
# group discovery (structural, no integral->group table)
# --------------------------------------------------------------------------- #

def test_group_discovery_selects_box_group_header():
    assert group_header_for_files(["B1m.h", "kokkosUtils.h"]) == "box/B1m.h"
    assert group_header_for_files(["/x/y/B2m.h"]) == "box/B2m.h"


def test_group_discovery_ignores_non_group_headers():
    # kokkosMaths.h / kokkosUtils.h are not B<k>m.h group headers.
    assert group_header_for_files(["B3m.h", "kokkosMaths.h", "timer.h"]) == "box/B3m.h"


def test_group_discovery_rejects_no_group():
    with pytest.raises(TUEmitError):
        group_header_for_files(["kokkosUtils.h", "timer.h"])


def test_group_discovery_rejects_multi_group():
    with pytest.raises(TUEmitError):
        group_header_for_files(["B1m.h", "B2m.h"])


# --------------------------------------------------------------------------- #
# precision parameterization (STOP #SS)
# --------------------------------------------------------------------------- #

def test_dd_profile_available():
    assert profile_for(TargetPrecision.DD).precision is TargetPrecision.DD


def test_unavailable_precision_fails_loud_not_dd_fallback(monkeypatch):
    # An unavailable precision must fail loud, never silently degrade to dd (reverse-STOP
    # #SS).  All three shipped precisions are now available (dd/quad static, float shim, ff
    # enrichment), so we assert the fail-loud path on a profile forced unavailable — the
    # mechanism, not any one precision.
    forced = PROFILES[TargetPrecision.FF]
    monkeypatch.setitem(
        PROFILES, TargetPrecision.FF,
        replace(forced, available=False))
    with pytest.raises(TUEmitError):
        profile_for(TargetPrecision.FF)
    with pytest.raises(TUEmitError):
        render_group_driver("box/B1m.h", TargetPrecision.FF)


def test_ff_profile_available_via_enrichment():
    # Phase-2: FF is served by its static enrichment header kokkosMaths_ff.h (commit
    # d0f5b35) — a static-header profile like dd (NOT shim synthesis), with the custom
    # ql::ffun::ffcomplex container that clears STOP #EEE.
    prof = PROFILES[TargetPrecision.FF]
    assert prof.available
    assert not prof.shim_synthesis
    assert prof.maths_header == "kokkosMaths_ff.h"
    assert prof.cpp_output == "ql::ffun::ffcomplex"
    assert prof.cpp_scalar == "ql::ffun::ffloat"
    assert prof.two_limb
    assert prof.define_macro == "USE_FF_COMPLEX"
    assert profile_for(TargetPrecision.FF).precision is TargetPrecision.FF


def test_ff_group_driver_shape():
    d = render_group_driver("box/B1m.h", TargetPrecision.FF)
    assert '#include "box/B1m.h"' in d
    assert "#define USE_FF_COMPLEX" in d
    assert "ql::ffun::ffcomplex" in d
    assert "ql::ffun::ffloat" in d
    assert "FFPrinter" in d
    assert "run_app<" in d


def test_float_profile_available_via_shim_synthesis():
    # Phase-2: FLOAT is served by shim synthesis (no static header, no enrichment).
    prof = PROFILES[TargetPrecision.FLOAT]
    assert prof.available
    assert prof.shim_synthesis
    assert prof.maths_reference_header == "kokkosMaths.h"
    assert prof.reference_scalar == "double"
    assert prof.cpp_scalar == "float"
    assert not prof.two_limb
    # profile_for must return it (not fail loud) now that it is available.
    assert profile_for(TargetPrecision.FLOAT).precision is TargetPrecision.FLOAT


def test_float_wrapper_is_two_line_shim_shape():
    # A shim-synthesis wrapper includes the double reference then the generated shim, and
    # carries NO precision #define ladder (the shim supplies the leaves directly).
    w = render_wrapper(TargetPrecision.FLOAT)
    assert '#include "kokkosMaths.h"' in w
    assert '#include "kokkosMaths_float_shim.hpp"' in w
    assert "USE_DD_COMPLEX" not in w
    assert "#elif" not in w
    assert w.count("#pragma once") == 1


def test_float_group_driver_shape():
    d = render_group_driver("box/B1m.h", TargetPrecision.FLOAT)
    assert '#include "box/B1m.h"' in d
    assert "Kokkos::complex<float>" in d
    assert "FloatPrinter" in d
    # native single-limb float: no USE_*_COMPLEX define, includes Kokkos_Complex for the container.
    assert "#define USE_" not in d
    assert "#include <Kokkos_Complex.hpp>" in d


def test_profiles_declare_every_ladder_target():
    # The table is parameterized across every precision the ladder can target.  QF
    # joined at the QF integration; it is declared here even while unavailable, which
    # is what lets profile_for fail loud instead of KeyError-ing.
    assert set(PROFILES) == {TargetPrecision.DD, TargetPrecision.QF,
                             TargetPrecision.FF, TargetPrecision.FLOAT}


# --------------------------------------------------------------------------- #
# emission + snapshot guard (STOP #Z)
# --------------------------------------------------------------------------- #

_FAKE_MATHS = """\
#pragma once
namespace ql {
    template<typename T>
    KOKKOS_INLINE_FUNCTION T kAbs(T const& x) { return Kokkos::abs(x); }

    KOKKOS_INLINE_FUNCTION double kAbs(double const& x) { return Kokkos::abs(x); }

    KOKKOS_INLINE_FUNCTION double Real(Kokkos::complex<double> const& x) { return x.real(); }
}
"""


def _fake_tree(root: Path) -> Path:
    (root / "box").mkdir(parents=True)
    (root / "boxGPU.h").write_text("// meta\n")
    (root / "box" / "B1m.h").write_text("// group\n")
    (root / "kokkosMaths_wrapper.h").write_text("// old wrapper\n")
    (root / "kokkosMaths.h").write_text(_FAKE_MATHS)
    return root


def test_emit_writes_wrapper_and_driver_into_clone(tmp_path):
    clone = _fake_tree(tmp_path / "tree")
    drv = tmp_path / "drv"
    tu = emit_flip_tu(clone, "box/B1m.h", drv, TargetPrecision.DD)
    assert tu.wrapper_path == clone / "kokkosMaths_wrapper.h"
    assert "USE_DD_COMPLEX" in tu.wrapper_path.read_text()
    assert tu.driver_path.exists()
    assert '#include "box/B1m.h"' in tu.driver_path.read_text()


def test_emit_float_writes_shim_into_clone(tmp_path):
    clone = _fake_tree(tmp_path / "tree")
    drv = tmp_path / "drv"
    tu = emit_flip_tu(clone, "box/B1m.h", drv, TargetPrecision.FLOAT)
    # The shim is written into the clone, alongside (not over) the reference header.
    assert tu.shim_path == clone / "kokkosMaths_float_shim.hpp"
    assert tu.shim_path.exists()
    shim = tu.shim_path.read_text()
    assert "@shim-inventory-sha256:" in shim
    # Only the non-template double overloads get float siblings (kAbs(double), Real(complex)).
    assert "float kAbs(float const& x)" in shim
    assert "Real(Kokkos::complex<float> const& x)" in shim
    # The generic template kAbs<T> is NOT re-emitted (it auto-instantiates at float).
    body = shim[shim.index("namespace ql {"):]
    assert "template" not in body
    # The wrapper points at the reference + the shim.
    w = tu.wrapper_path.read_text()
    assert '#include "kokkosMaths.h"' in w
    assert '#include "kokkosMaths_float_shim.hpp"' in w
    # The reference header itself is untouched.
    assert (clone / "kokkosMaths.h").read_text() == _FAKE_MATHS


def test_emit_float_shim_reused_when_reference_unchanged(tmp_path):
    clone = _fake_tree(tmp_path / "tree")
    drv = tmp_path / "drv"
    tu = emit_flip_tu(clone, "box/B1m.h", drv, TargetPrecision.FLOAT)
    first = tu.shim_path.read_text()
    # Re-emit against the same (unchanged) reference: sha-keyed no-op, byte-identical shim.
    tu2 = emit_flip_tu(clone, "box/B1m.h", drv, TargetPrecision.FLOAT)
    assert tu2.shim_path.read_text() == first


def test_emit_float_shim_regenerated_when_reference_changes(tmp_path):
    clone = _fake_tree(tmp_path / "tree")
    drv = tmp_path / "drv"
    emit_flip_tu(clone, "box/B1m.h", drv, TargetPrecision.FLOAT)
    # Add a new non-template leaf to the reference -> inventory sha changes -> regenerate.
    ref = clone / "kokkosMaths.h"
    ref.write_text(ref.read_text().replace(
        "    KOKKOS_INLINE_FUNCTION double Real(Kokkos::complex<double> const& x) "
        "{ return x.real(); }\n",
        "    KOKKOS_INLINE_FUNCTION double Real(Kokkos::complex<double> const& x) "
        "{ return x.real(); }\n"
        "    KOKKOS_INLINE_FUNCTION double Imag(double const& x) { return 0.0; }\n"))
    tu2 = emit_flip_tu(clone, "box/B1m.h", drv, TargetPrecision.FLOAT)
    assert "Imag(float const& x)" in tu2.shim_path.read_text()


def test_emit_refuses_snapshot_write():
    # Point at the real pristine snapshot: the guard must fire before any write.
    snap = Path(__file__).resolve().parents[2] / "runs" / "qcdloop_headers_full"
    with pytest.raises(TUEmitError):
        emit_flip_tu(snap, "box/B1m.h", Path("/tmp/should_not_matter"))


def test_emit_rejects_non_tree(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(TUEmitError):
        emit_flip_tu(empty, "box/B1m.h", tmp_path / "drv")


# --------------------------------------------------------------------------- #
# QF (quad-float, 4xFP32) — the rung between double and dd
# --------------------------------------------------------------------------- #

def test_qf_profile_available_via_enrichment():
    # QF is served by its static enrichment header kokkosMaths_qf.h — a static-header
    # profile like dd/ff (NOT shim synthesis), on the custom ql::qfun::qfcomplex
    # container (QuadFloatComplex), which is QF's answer to the same STOP #EEE that
    # forced ff off Kokkos::complex.
    prof = PROFILES[TargetPrecision.QF]
    assert prof.available
    assert not prof.shim_synthesis
    assert prof.maths_header == "kokkosMaths_qf.h"
    assert prof.define_macro == "USE_QF_COMPLEX"
    assert prof.cpp_output == "ql::qfun::qfcomplex"
    assert prof.cpp_scalar == "ql::qfun::qfloat"
    assert prof.printer_name == "QFPrinter"
    assert profile_for(TargetPrecision.QF).precision is TargetPrecision.QF


def test_qf_group_driver_shape():
    d = render_group_driver("box/B1m.h", TargetPrecision.QF)
    assert '#include "box/B1m.h"' in d
    assert "#define USE_QF_COMPLEX" in d
    assert "ql::qfun::qfcomplex" in d
    assert "ql::qfun::qfloat" in d
    assert "QFPrinter" in d
    assert "run_app<" in d
    # QF is an extended aggregate, so the driver must NOT pull in Kokkos_Complex.hpp
    # (that include is for the native-scalar profiles' Kokkos::complex<float>).
    assert "#include <Kokkos_Complex.hpp>" not in d


def test_unavailable_target_still_fails_loud():
    # Reverse-STOP #SS: the fail-loud path must survive QF becoming available — an
    # undeclared/unavailable target raises rather than silently degrading to dd.
    from dataclasses import replace as _replace
    prof = PROFILES[TargetPrecision.QF]
    PROFILES[TargetPrecision.QF] = _replace(prof, available=False)
    try:
        with pytest.raises(TUEmitError):
            profile_for(TargetPrecision.QF)
        with pytest.raises(TUEmitError):
            render_group_driver("box/B1m.h", TargetPrecision.QF)
    finally:
        PROFILES[TargetPrecision.QF] = prof


def test_qf_is_four_limb_not_two():
    # The crux of the profile: QF IS an extended aggregate (two_limb=True, meaning
    # "not a native scalar"), but its limbs are f0..f3 — it has no .hi/.lo at all, so
    # the two-limb spelling would not compile against it.
    prof = PROFILES[TargetPrecision.QF]
    assert prof.two_limb
    assert prof.limbs == ("f0", "f1", "f2", "f3")
    assert PROFILES[TargetPrecision.DD].limbs == ("hi", "lo")
    assert PROFILES[TargetPrecision.FF].limbs == ("hi", "lo")
    assert PROFILES[TargetPrecision.FLOAT].limbs == ()


def test_qf_printer_sums_every_limb():
    # The QF narrowing printer must sum ALL FOUR limbs.  Truncating to f0 would deliver
    # ~7 digits from a type that carries ~29, and would do so silently.
    from agents.patcher.tu_emit import _printer_struct
    body = _printer_struct(PROFILES[TargetPrecision.QF])
    for limb in ("f0", "f1", "f2", "f3"):
        assert f"static_cast<double>(v.{limb})" in body
    assert body.count("static_cast<double>") == 4
    assert ".hi" not in body and ".lo" not in body


def test_dd_and_ff_printers_unchanged_by_the_limb_generalization():
    # Regression guard: generalizing the narrowing primitive to a limb list must leave
    # the dd/ff drivers byte-identical, or the routing baseline moves under us.
    from agents.patcher.tu_emit import _printer_struct
    assert ("out += dhex(static_cast<double>(v.hi) + static_cast<double>(v.lo));"
            in _printer_struct(PROFILES[TargetPrecision.DD]))
    assert ("out += dhex(static_cast<double>(v.hi) + static_cast<double>(v.lo));"
            in _printer_struct(PROFILES[TargetPrecision.FF]))
    # FLOAT narrows to caller_type "double" too — a plain cast, no limb members.
    assert ("out += dhex(static_cast<double>(v));"
            in _printer_struct(PROFILES[TargetPrecision.FLOAT]))


def test_qf_arm_present_in_the_wrapper_ladder():
    # Now that QF is available it must contribute its own arm — and the arm must be
    # distinct from the CUDA __nv_fp128 USE_QUAD_COMPLEX arm, which is a different
    # precision that merely reads as "quad".
    w = render_wrapper(TargetPrecision.DD)
    assert "defined(USE_QF_COMPLEX)" in w
    assert '#include "kokkosMaths_qf.h"' in w
    assert "defined(USE_QUAD_COMPLEX)" in w
    assert '#include "kokkosMaths_quad.h"' in w
    assert w.index("USE_QF_COMPLEX") != w.index("USE_QUAD_COMPLEX")


def test_wrapper_arm_doc_is_generated_from_available_profiles():
    # The doc block is generated, so it cannot drift from the ladder beneath it:
    # every available macro profile gets exactly one "-> <scalar>" line.
    w = render_wrapper(TargetPrecision.DD)
    for prof in PROFILES.values():
        if prof.available and prof.define_macro is not None:
            assert f"-> {prof.cpp_scalar} ({prof.maths_header})" in w


def test_unavailable_arm_absent_from_the_wrapper_ladder():
    # The converse: an unavailable profile contributes neither an arm nor a doc line.
    from dataclasses import replace as _replace
    prof = PROFILES[TargetPrecision.QF]
    PROFILES[TargetPrecision.QF] = _replace(prof, available=False)
    try:
        w = render_wrapper(TargetPrecision.DD)
        assert "#if defined(USE_QF_COMPLEX)" not in w
        assert "#elif defined(USE_QF_COMPLEX)" not in w
        assert '#include "kokkosMaths_qf.h"' not in w
        assert "-> ql::qfun::qfloat" not in w
    finally:
        PROFILES[TargetPrecision.QF] = prof
