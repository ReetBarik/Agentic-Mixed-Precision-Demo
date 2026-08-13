"""Subtask L3 — the leaf-clone COMPILE gate (permanent P2 / P5 probes).

These are the design's *verdict gate* probes, promoted from throwaway `/tmp` scripts
(``runs/qcdloop/tier_b_stage2_leaf_promotion/probe_evidence/``) to permanent regression
tests.  They are the falsifier the dispatch reserved:

> *"If the L1′ P2 compile test fails when adapted, STOP at L3 and hand the scope call
>  back."*

The gate encodes **Resolution A** — source-provided dd Class-1 wrappers
(``kAbs``/``kLog``/``Sign``/``Constants``…) are a **dd boundary**, so the pipeline does
NOT synthesize them.  Three facts are pinned against the *real* toolchain (gcc/13.3.0 +
Kokkos + the vendored dd headers + the enriched ``kokkosMaths_dd.h``):

* **P2 (Resolution A).** The clone :func:`render_variant` actually emits — byte-exact
  ``Lnrat_B10`` — compiles against the enriched source with **no synth overlay** and
  computes ``log(1.5/2.5) = log(0.6) = -0.51082562376599072`` in dd.  This is the leaf
  body that STOP #K said could not be synthesized; under Resolution A it comes from
  source and builds.
* **P2-negative.** The *pre-*Resolution-A synth overlay (a redundant ``ql::kLog``/
  ``kAbs`` at dd) is what makes it fail — an ambiguating redeclaration against the
  source wrapper (STOP #K, report §2.1).  This proves Resolution A is load-bearing, not
  incidental: the fix is *removing* the synthesis, and only that.
* **P5.** The enriched source ships the 43-coeff dd ``Constants<T>`` table bit-exactly
  (discharges STOP #E), so Class-2 is source-resident too — the twin of the Class-1
  dissolution P2 relies on.

Heavyweight and environment-specific: skipped unless gcc/13.3.0 (via ``module load`` or
a ``g++ >= 13`` on PATH), a Kokkos install (``$KOKKOS_INSTALL`` or ``~/kokkos-install``),
the vendored dd headers, and the enriched vendored snapshot are all present.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from agents.patcher.fanout import ReturnWiden, VariantSpec, render_variant
from agents.patcher.instantiation_gate import classify_build_log, SHAPE_1_EXIT_NARROW
from tests.patcher.fanout.conftest import requires_qcdloop_full

_ROOT = Path(__file__).resolve().parents[3]
_QCDLOOP_FULL = _ROOT / "runs" / "qcdloop_headers_full"
_DD_INCLUDE = _ROOT / "third_party" / "include"
_PROBE_EVIDENCE = (_ROOT / "runs" / "qcdloop" / "tier_b_stage2_leaf_promotion"
                   / "probe_evidence")
_KOKKOS = Path(os.environ.get("KOKKOS_INSTALL", str(Path.home() / "kokkos-install")))

# The @138 shallow ``Lnrat(TScale const&, TScale const&)`` overload the B10 chain reaches
# (graph-confirmed extent; the control-flow @126 overload is NOT the one Li2omx2/B10 call).
_LNRAT_TSCALE_EXTENT = (138, 141)


def _kokkos_ok() -> bool:
    return ((_KOKKOS / "include" / "Kokkos_Core.hpp").is_file()
            and (_KOKKOS / "lib64").is_dir())


def _have_dd_headers() -> bool:
    return ((_DD_INCLUDE / "dd_math.hpp").is_file()
            and (_DD_INCLUDE / "dd_complex.hpp").is_file())


requires_kokkos = pytest.mark.skipif(
    not _kokkos_ok(),
    reason=f"Kokkos install not found at {_KOKKOS} (set $KOKKOS_INSTALL)")
requires_dd_headers = pytest.mark.skipif(
    not _have_dd_headers(), reason=f"vendored dd headers not found in {_DD_INCLUDE}")


def _nvcc_env_compile(src: Path, out: Path) -> subprocess.CompletedProcess:
    """Compile ``src`` with gcc/13.3.0 + Kokkos + dd + enriched source (Resolution A).

    Runs through a login shell so ``module load gcc/13.3.0`` is honored exactly as the
    pipeline's build chain does (``module use /soft/modulefiles``); falls back to a
    PATH ``g++`` when the module system is absent.  ``-Iruns/qcdloop_headers_full`` puts
    the ENRICHED ``kokkosMaths_dd.h`` (source-provided dd wrappers + 43-coeff Constants)
    ahead of everything — that is the Resolution-A boundary the clone builds against.
    """
    cmd = (
        "module use /soft/modulefiles >/dev/null 2>&1; "
        "module load gcc/13.3.0 >/dev/null 2>&1; "
        f"g++ -std=c++20 -w "
        f"-I{_DD_INCLUDE} -I{_KOKKOS / 'include'} -I{_QCDLOOP_FULL} "
        f"{src} -L{_KOKKOS / 'lib64'} -lkokkoscore -lkokkoscontainers -ldl "
        f"-o {out}"
    )
    return subprocess.run(["bash", "-lc", cmd], capture_output=True, text=True,
                          timeout=300)


def _emit_lnrat_b10_clone() -> str:
    """The byte-exact ``Lnrat_B10`` the L3 emission pass renders (report §1.1).

    Built exactly as :func:`agents.patcher.chain_promote._materialize_leaf_variants`
    builds it: a verbatim clone of the @138 ``Lnrat`` overload with a rule-(c)-style
    return widen to the box binding's dd complex container (``TOutput == DoubleDoubleComplex``).
    """
    spec = VariantSpec(
        variant_name="Lnrat_B10", orig_name="Lnrat",
        file=str(_QCDLOOP_FULL / "kokkosUtils.h"),
        orig_start=_LNRAT_TSCALE_EXTENT[0], orig_end=_LNRAT_TSCALE_EXTENT[1])
    spec.return_widen = ReturnWiden(
        return_line=139, orig_type="TOutput",
        dd_type="Kokkos::Experimental::DoubleDoubleComplex", function_name="Lnrat_B10")
    return render_variant(spec)


# --------------------------------------------------------------------------- #
# P2 — Resolution A: the emitted clone compiles + runs against source alone
# --------------------------------------------------------------------------- #

@requires_qcdloop_full
@requires_dd_headers
@requires_kokkos
def test_p2_resolution_a_leaf_clone_compiles_and_runs(tmp_path):
    # The clone render_variant ACTUALLY emits, compiled against the enriched source with
    # NO synth overlay — the exact Resolution-A configuration (Class-1 = source boundary).
    clone = _emit_lnrat_b10_clone()
    # sanity: the rendered clone is the @138 shallow body with a widened return.
    assert "Kokkos::Experimental::DoubleDoubleComplex Lnrat_B10(TScale const& x, TScale const& y)" in clone
    assert "ql::Lnrat" not in clone            # renamed: no self-recursion (Q1)

    src = tmp_path / "probe_resA.cpp"
    src.write_text(
        '#include <Kokkos_Core.hpp>\n'
        '#include "kokkosMaths_dd.h"\n'
        '#include <dd_math.hpp>\n'
        '#include <dd_complex.hpp>\n'
        'using Kokkos::Experimental::DoubleDouble;\n'
        'using Kokkos::Experimental::DoubleDoubleComplex;\n'
        'namespace ql {\n'
        f'{clone}\n'
        '}\n'
        'int main(int argc, char** argv) {\n'
        '    Kokkos::initialize(argc, argv);\n'
        '    int bad = 0;\n'
        '    {\n'
        '        DoubleDouble v(1.5), x(2.5);\n'
        '        auto r = ql::Lnrat_B10<DoubleDoubleComplex, double, DoubleDouble>(v, x);\n'
        '        Kokkos::printf("re=%.17g im=%.17g\\n", r.real().hi, r.imag().hi);\n'
        '    }\n'
        '    Kokkos::finalize();\n'
        '    return bad;\n'
        '}\n')
    out = tmp_path / "resA"
    res = _nvcc_env_compile(src, out)
    assert res.returncode == 0, (
        "Resolution-A leaf clone must compile against the enriched source "
        f"(STOP #K verdict gate). stderr:\n{res.stderr}")
    run = subprocess.run([str(out)], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    # log(1.5/2.5) = log(0.6) in dd; the TScale branch has no imaginary part.
    assert "re=-0.51082562376599072" in run.stdout, run.stdout
    assert "im=0" in run.stdout, run.stdout


# --------------------------------------------------------------------------- #
# RETROSPECTIVE INSTANTIATION AUDIT (STOP #A dispatch fix, 2026-07-28)         #
#                                                                             #
# The P2 test above instantiates the clone at the IDEALISED all-dd binding     #
# ``<DoubleDoubleComplex, double, DoubleDouble>`` — which is NOT the binding the real build    #
# uses.  The box instantiates ``ql::BO`` (and every clone it reaches) at the   #
# REAL binding ``TOutput = Kokkos::complex<double>``, ``TScale = double``.  A   #
# clone whose return is widened to ``DoubleDoubleComplex`` then lands in a               #
# ``const TOutput`` (== ``Kokkos::complex<double>``) receiver with no          #
# narrowing — the Shape-1 exit-boundary defect that the STOP #A dispatch fix    #
# uncovered.  These probes force the REAL binding so that false positive can    #
# never recur in a unit test: an uninstantiated / idealised-binding compile is  #
# not evidence the emission is type-correct.                                    #
# --------------------------------------------------------------------------- #

def _compile_clone_at_binding(tmp_path, toutput: str, main_body: str,
                              out_name: str) -> subprocess.CompletedProcess:
    """Render the real ``Lnrat_B10`` clone and compile a TU instantiating it at
    ``toutput`` (the box's ``TOutput`` binding)."""
    clone = _emit_lnrat_b10_clone()
    src = tmp_path / f"{out_name}.cpp"
    # Include the base ``kokkosMaths.h`` (the double primary + Sign(double)/
    # Constants<double>) exactly as the real build's wrapper chain does — the box is
    # instantiated at ``TScale = double``, so the clone body resolves against the
    # double overloads, and ONLY the widened return (DoubleDoubleComplex) vs the double-computed
    # body value is the type conflict.  Including the all-dd ``kokkosMaths_dd.h`` here
    # instead would ODR-collide with the base and produce artifact errors the real
    # build never sees (it is the *reference* dd header, not layered on top).
    src.write_text(
        '#include <Kokkos_Core.hpp>\n'
        '#include "kokkosMaths.h"\n'
        '#include <dd_math.hpp>\n'
        '#include <dd_complex.hpp>\n'
        'using Kokkos::Experimental::DoubleDouble;\n'
        'using Kokkos::Experimental::DoubleDoubleComplex;\n'
        'namespace ql {\n'
        f'{clone}\n'
        '}\n'
        'int main(int argc, char** argv) {\n'
        '    Kokkos::initialize(argc, argv);\n'
        f'    {main_body}\n'
        '    Kokkos::finalize();\n'
        '    return 0;\n'
        '}\n')
    return _nvcc_env_compile(src, tmp_path / out_name)


@requires_qcdloop_full
@requires_dd_headers
@requires_kokkos
def test_audit_real_box_binding_exposes_shape1_truncation(tmp_path):
    """AUDIT: at the REAL box binding the widened clone truncates into ``const TOutput``.

    This is the false positive the idealised-binding P2 test hid.  We assert the
    failure IS the Shape-1 exit-boundary defect (a dd value into a
    ``Kokkos::complex<double>`` receiver) — pinning that the emission, as-is, is not
    type-correct under the real instantiation, and that the classifier buckets it.
    Once the Shape-1 emission fix lands, the companion narrowing probe below compiles.
    """
    # The exact real-build shape: clone return (DoubleDoubleComplex) into a const TOutput local.
    body = (
        'double v = 1.5, x = 2.5;\n'
        '    const Kokkos::complex<double> r = '
        'ql::Lnrat_B10<Kokkos::complex<double>, double, double>(v, x);\n'
        '    (void)r;')
    res = _compile_clone_at_binding(tmp_path, "Kokkos::complex<double>", body,
                                    "audit_real_bind")
    assert res.returncode != 0, (
        "the widened clone must NOT silently compile into a const "
        "Kokkos::complex<double> at the real box binding — if it does, the emission "
        "is either fixed (update this audit) or the test lost its teeth")
    # The failure must be the Shape-1 exit-boundary defect the classifier recognises.
    report = classify_build_log(res.stderr)
    assert not report.has_unknown, f"STOP #BB — unclassified: {report.unknown}"
    assert report.counts().get(SHAPE_1_EXIT_NARROW, 0) >= 1, res.stderr


@requires_qcdloop_full
@requires_dd_headers
@requires_kokkos
def test_audit_explicit_narrowing_compiles_at_real_binding(tmp_path):
    """AUDIT (positive): an explicit dd->caller narrowing at the exit compiles.

    This is the SHAPE-1 FIX in miniature, self-contained (independent of the leaf
    clone, whose verbatim body has its own rule-(d) instantiation issue): a
    ``DoubleDoubleComplex`` value narrowed to the caller ``Kokkos::complex<double>`` via
    component ``.hi`` reconstruction is exactly what the boundary transform must emit
    at a designed exit.  Proves the fix DIRECTION is sound at the real binding (the
    same binding the box uses) — a dd value CAN be delivered into a
    ``Kokkos::complex<double>`` sink when the narrowing is emitted, and CANNOT when it
    is a raw assignment (the negative audit above).
    """
    src = tmp_path / "audit_narrow.cpp"
    src.write_text(
        '#include <Kokkos_Core.hpp>\n'
        '#include "kokkosMaths.h"\n'
        '#include <dd_math.hpp>\n'
        '#include <dd_complex.hpp>\n'
        'using Kokkos::Experimental::DoubleDouble;\n'
        'using Kokkos::Experimental::DoubleDoubleComplex;\n'
        'int main(int argc, char** argv) {\n'
        '    Kokkos::initialize(argc, argv);\n'
        '    {\n'
        '        DoubleDoubleComplex dd(DoubleDouble(-0.51082562376599072), DoubleDouble(0.0));\n'
        # the exact narrowing the boundary transform must emit at a designed exit:
        '        const Kokkos::complex<double> r(dd.real().hi, dd.imag().hi);\n'
        '        Kokkos::printf("re=%.17g\\n", r.real());\n'
        '    }\n'
        '    Kokkos::finalize();\n'
        '    return 0;\n'
        '}\n')
    res = _nvcc_env_compile(src, tmp_path / "audit_narrow")
    assert res.returncode == 0, (
        "an explicit dd->caller narrowing at the exit boundary MUST compile at the "
        f"real box binding (the Shape-1 fix direction). stderr:\n{res.stderr}")
    run = subprocess.run([str(tmp_path / 'audit_narrow')], capture_output=True,
                         text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    assert "re=-0.51082562376599072" in run.stdout, run.stdout


# --------------------------------------------------------------------------- #
# P2-negative — the pre-Resolution-A synth overlay is what STOP #K rejected
# --------------------------------------------------------------------------- #

@requires_qcdloop_full
@requires_dd_headers
@requires_kokkos
def test_p2_synth_overlay_collides_with_source(tmp_path):
    # Re-introducing the synthesized Class-1 dd wrappers (what L1′ emitted BEFORE
    # Resolution A) is exactly what breaks: an ambiguating redeclaration against the
    # source wrappers in kokkosMaths_dd.h (report §2.1, STOP #K).  This pins that
    # NOT-synthesizing is the fix — the collision is real, not hypothetical.
    src = tmp_path / "probe_synth.cpp"
    src.write_text(
        '#include <Kokkos_Core.hpp>\n'
        '#include "kokkosMaths_dd.h"\n'
        '#include <dd_math.hpp>\n'
        '#include <dd_complex.hpp>\n'
        'using Kokkos::Experimental::DoubleDouble;\n'
        'using Kokkos::Experimental::DoubleDoubleComplex;\n'
        'namespace ql {\n'
        '  KOKKOS_INLINE_FUNCTION auto kLog(DoubleDoubleComplex const& z){ return Kokkos::Experimental::log(z); }\n'
        '  KOKKOS_INLINE_FUNCTION auto kAbs(DoubleDoubleComplex const& z){ return Kokkos::Experimental::abs(z); }\n'
        '}\n'
        'int main(){ return 0; }\n')
    res = _nvcc_env_compile(src, tmp_path / "synth")
    assert res.returncode != 0, (
        "the pre-Resolution-A synth overlay MUST collide with the source dd wrappers "
        "(if this ever compiles, the enriched source stopped providing them and "
        "Resolution A's premise is void)")
    assert "ambiguating" in res.stderr or "redeclaration" in res.stderr, res.stderr


# --------------------------------------------------------------------------- #
# P5 — the enriched source ships the 43-coeff dd Constants table (STOP #E)
# --------------------------------------------------------------------------- #

@requires_qcdloop_full
@requires_dd_headers
@requires_kokkos
def test_p5_enriched_source_provides_dd_constants(tmp_path):
    probe = _PROBE_EVIDENCE / "probe_constants_dd43.cpp"
    if not probe.is_file():
        pytest.skip(f"P5 probe source not found at {probe}")
    out = tmp_path / "pC43"
    res = _nvcc_env_compile(probe, out)
    assert res.returncode == 0, res.stderr
    run = subprocess.run([str(out)], capture_output=True, text=True, timeout=120)
    assert run.returncode == 0, run.stderr
    # 43-coeff dd table, bit-exact _pi, and the π²/12 sum (Class-2 is source-resident).
    assert "P5 PASS: enriched source provides 43-coeff dd table" in run.stdout, run.stdout
    assert "sum_C(43) hi=0.8224670334241132" in run.stdout, run.stdout
