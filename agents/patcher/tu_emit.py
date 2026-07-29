"""Phase-1 template-argument promotion — per-integral TU emission (deliverable 2).

The Phase-1 correctness mechanism is a per-integral **whole-TU precision flip**: a
flagged integral is compiled in its own translation unit at its own precision, then its
output is narrowed at the app boundary (deliverable 4).  This module emits the two build
artifacts that flip needs, both **precision-parameterized** so Phase 2/3 can target
``ff`` / ``float`` without a rewrite (hard-coding to dd is STOP #SS):

1. **The precision-parameterized wrapper** (:func:`render_wrapper`).  The master
   snapshot's ``kokkosMaths_wrapper.h`` only branches ``USE_QUAD_COMPLEX`` vs the double
   ``kokkosMaths.h`` — it has **no** ``USE_DD_COMPLEX`` arm (design §5.5), so building the
   snapshot at dd would silently fall through to the double ``Constants<T>`` (19-term Cheb,
   ~16 digits) — a numerical defect, not a compile error.  The generator emits the
   fork-shape wrapper (one arm per *available* precision profile, selected by the driver's
   ``#define``).  It is written into the **cloned** working tree only; the snapshot stays
   pristine (STOP #Z — :func:`emit_flip_tu` refuses to write under the snapshot dir).

2. **The per-group driver TU** (:func:`render_group_driver`).  A clean single-precision
   compile identical in shape to the ``boxGPU_dd.cpp`` oracle, except it includes only the
   integral's **group header** (``box/B1m.h``) instead of the meta-header ``boxGPU.h``.
   Omitting ``boxGPU.h`` means ``QCDLOOP_BOX_FULL_DISPATCH`` is never defined, so the
   *pruned per-group* ``BO`` is active — the TU pulls in only the integral's own mass-group
   (B1m for B6-B10), isolating it from the B2m/B3m/B4m gaps.  This is exactly the PoC that
   built B10 clean at dd against the unmodified vendored surface.

The group is discovered **structurally** — :func:`group_header_for_files` reads the
integral's characterization region file names and returns the ``box/<group>.h`` among them.
No integral→group table is baked in (feedback_no_placeholder_patterns); the only shape
assumed is the qcdloop convention that a box group header lives under ``box/`` — which the
caller confirms by passing region files, not names.

The emitter never mutates ``runs/qcdloop_headers_full`` or ``third_party/`` and never reads
from ``ddfun_enabled`` (Decision 2: the pipeline builds against its own snapshot + vendored
primitives, not the Validator's oracle tree).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from agents.integrator_base.boundary import narrow_two_limb_scalar
from agents.patcher.precision_flip import TargetPrecision

# The master snapshot header tree — must stay pristine (STOP #Z).  Any emit target that
# resolves to a path under here is refused.
_SNAPSHOT_DIR = (Path(__file__).resolve().parents[2]
                 / "runs" / "qcdloop_headers_full").resolve()


class TUEmitError(RuntimeError):
    """An emission precondition failed (unavailable precision, snapshot write, etc.)."""


@dataclass(frozen=True)
class PrecisionProfile:
    """Everything precision-specific the wrapper + driver generators need for one target.

    A profile is *available* iff its ``maths_header`` is a header the snapshot vendors
    (dd is; ff/float are not yet — their profiles are declared so the generator branches
    on a table, never on a hard-coded ``dd``, but selecting an unavailable one fails
    loud rather than inventing a header).
    """

    precision: TargetPrecision
    define_macro: str | None      # driver #define selecting this arm; None = default arm
    maths_header: str             # precision maths header the wrapper's arm includes
    cpp_output: str               # TOutput template arg (the complex container)
    cpp_scalar: str               # TMass / TScale template arg (the real scalar)
    printer_name: str             # ql_app Printer struct name emitted into the driver
    two_limb: bool                # component printer emits hi|lo (dd/ff) vs one token
    available: bool               # the maths_header is vendored in the snapshot
    caller_type: str = "double"   # the app-boundary caller precision the flip narrows to


# Precision profile table.  DD is fully wired for Phase-1; FF/FLOAT are declared (so the
# stack is parameterized — STOP #SS) but marked unavailable until their maths headers are
# vendored (a Phase-2/3 prerequisite, not a Phase-1 emission).
PROFILES: dict[TargetPrecision, PrecisionProfile] = {
    TargetPrecision.DD: PrecisionProfile(
        precision=TargetPrecision.DD,
        define_macro="USE_DD_COMPLEX",
        maths_header="kokkosMaths_dd.h",
        cpp_output="ql::ddfun::ddcomplex",
        cpp_scalar="ql::ddfun::ddouble",
        printer_name="DDPrinter",
        two_limb=True,
        available=True),
    TargetPrecision.FF: PrecisionProfile(
        precision=TargetPrecision.FF,
        define_macro="USE_FF_COMPLEX",
        maths_header="kokkosMaths_ff.h",
        cpp_output="ql::ffun::ffcomplex",
        cpp_scalar="ql::ffun::ffloat",
        printer_name="FFPrinter",
        two_limb=True,
        available=False),
    TargetPrecision.FLOAT: PrecisionProfile(
        precision=TargetPrecision.FLOAT,
        define_macro=None,                 # default arm (kokkosMaths.h), Kokkos::complex
        maths_header="kokkosMaths.h",
        cpp_output="Kokkos::complex<float>",
        cpp_scalar="float",
        printer_name="FloatPrinter",
        two_limb=False,
        available=False),
}


def profile_for(target: TargetPrecision) -> PrecisionProfile:
    """The :class:`PrecisionProfile` for ``target``, or fail loud if unavailable.

    Fails (rather than degrading to dd) when the target's maths header is not vendored —
    a Phase-2/3 target reaching here without its prerequisite header is a wiring bug the
    caller must see, not a silent dd fallback (which would be STOP #SS in reverse)."""
    prof = PROFILES.get(target)
    if prof is None:
        raise TUEmitError(f"no precision profile for target {target!r}")
    if not prof.available:
        raise TUEmitError(
            f"precision {target.value!r} profile is declared but its maths header "
            f"{prof.maths_header!r} is not vendored in the snapshot — wire the header "
            f"(a Phase-2/3 prerequisite) before targeting it; Phase-1 targets dd only")
    return prof


# --------------------------------------------------------------------------- #
# group discovery (structural, no integral→group table)
# --------------------------------------------------------------------------- #

def group_header_for_files(region_files) -> str:
    """The ``box/<group>.h`` group header among an integral's region files.

    ``region_files`` are the file names carried by the integral's characterization
    regions (bare basenames like ``B1m.h`` or paths).  Returns the group header path
    relative to the tree root (``box/B1m.h``) — the include the per-group driver uses so
    only that mass-group's pruned ``BO`` is active.  Structural, not name-mapped: it
    selects the file that lives under ``box/`` (the qcdloop group-header convention),
    raising if zero or several distinct groups are present (a chain spanning two mass
    groups is out of the per-group TU's scope — hand back rather than guess).
    """
    groups: set[str] = set()
    for f in region_files:
        name = Path(f).name
        # A box group header is one of the five mass-group files under box/.  We accept
        # any header whose basename matches the group-header shape (``B<k>m.h``) — a
        # structural pattern (k internal masses), not an enumerated integral list.
        stem = name[:-2] if name.endswith(".h") else name
        if len(stem) >= 3 and stem[0] == "B" and stem.endswith("m") and stem[1:-1].isdigit():
            groups.add(name)
    if not groups:
        raise TUEmitError(
            f"no box group header (B<k>m.h) among region files {list(region_files)!r} — "
            f"cannot select a per-group TU")
    if len(groups) > 1:
        raise TUEmitError(
            f"regions span multiple mass groups {sorted(groups)!r} — a per-group TU "
            f"covers one group; multi-group promotion is out of Phase-1 scope")
    return f"box/{groups.pop()}"


# --------------------------------------------------------------------------- #
# wrapper generator (precision-parameterized)
# --------------------------------------------------------------------------- #

def render_wrapper(target: TargetPrecision = TargetPrecision.DD) -> str:
    """Render the precision-parameterized ``kokkosMaths_wrapper.h``.

    Emits one ``#if defined(<macro>) -> #include <header>`` arm per *available* precision
    profile (so adding a vendored ff/float header automatically adds its arm), then the
    ``USE_QUAD_COMPLEX`` CUDA-quad arm (preserved from the snapshot), then the default
    double ``kokkosMaths.h``.  ``target`` is validated (its arm must be emittable) but the
    wrapper itself carries every arm — the *driver's* ``#define`` selects one.  This is the
    shape the fork's wrapper has and the snapshot's lacks (design §5.5).
    """
    profile_for(target)   # validate the requested target is emittable
    arms: list[str] = []
    for prof in PROFILES.values():
        if prof.available and prof.define_macro is not None:
            arms.append(f'#if defined({prof.define_macro})\n'
                        f'#include "{prof.maths_header}"')
    # The macro-guarded available arms come first as an #if / #elif ladder.
    ladder = ""
    for i, arm in enumerate(arms):
        body = arm.split("\n", 1)[1]
        cond = arm.split("\n", 1)[0]
        cond = cond.replace("#if ", "#elif " if i else "#if ")
        ladder += f"{cond}\n{body}\n"
    if not ladder:
        ladder = ""  # no macro arms available: fall straight to the quad/default ladder
    quad_default = (
        ("#elif defined(USE_QUAD_COMPLEX)\n" if ladder else "#if defined(USE_QUAD_COMPLEX)\n")
        + "#ifdef KOKKOS_ENABLE_CUDA\n"
        + '#include "kokkosMaths_quad.h"\n'
        + "#else\n"
        + '#error "USE_QUAD_COMPLEX requires KOKKOS_ENABLE_CUDA to be defined"\n'
        + "#endif\n"
        + "#else\n"
        + '#include "kokkosMaths.h"\n'
        + "#endif\n")
    return (
        "//\n"
        "// QCDLoop + Kokkos 2025 — precision wrapper (pipeline-generated, Phase-1 flip)\n"
        "//\n"
        "// Selects the precision maths header from the driver's build define:\n"
        "//   USE_DD_COMPLEX   -> ql::ddfun double-double (all backends)\n"
        "//   USE_QUAD_COMPLEX -> CUDA __nv_fp128 quad (CUDA only)\n"
        "//   neither          -> Kokkos::complex<double> (kokkosMaths.h)\n"
        "//\n"
        "// Generated into the CLONED working tree only; the snapshot wrapper is pristine\n"
        "// (STOP #Z).  Do not hand-edit — regenerate via agents.patcher.tu_emit.\n"
        "\n"
        "#pragma once\n"
        "\n"
        f"{ladder}{quad_default}")


# --------------------------------------------------------------------------- #
# per-group driver generator (precision-parameterized)
# --------------------------------------------------------------------------- #

def _printer_struct(prof: PrecisionProfile) -> str:
    """The ql_app **narrowing** Printer struct for ``prof`` (the app-output boundary).

    The flip computes the integral at ``prof`` precision internally, but the caller
    contract is ``prof.caller_type`` (double).  This printer is the app-output boundary
    (design §2.2.3: "the point where res_dd(i,k) is read back for the caller"): it
    reconstructs the caller-precision value from the extended value via the SHARED
    :func:`agents.integrator_base.boundary.narrow_two_limb_scalar` primitive — the same
    reconstruction the element-promotion designed-exit uses (STOP #TT: no one-off
    boundary narrowing) — then emits it as a single caller-precision ``dhex`` token,
    byte-format-identical to the vanilla driver so the Validator scorer ingests the flip
    candidate exactly like any double candidate.  The measured lift is therefore the
    honest caller-precision accuracy of a dd-computed result vs the dd reference — capped
    at double's ~15.9-digit floor, NOT the dd oracle's ~30 (which would be a false
    positive: the candidate does not deliver dd to the caller).
    """
    if prof.two_limb:
        recon = narrow_two_limb_scalar("v", prof.caller_type, two_limb=True)
        return (
            f"struct {prof.printer_name} {{\n"
            f"    // app-output boundary narrow: extended -> caller precision (STOP #TT)\n"
            f"    static void emit(std::string& out, const {prof.cpp_scalar}& v) {{\n"
            f"        out += dhex({recon});\n"
            f"    }}\n"
            f"}};")
    # Native single-limb extended scalar (float): a plain cast is the reconstruction.
    recon = narrow_two_limb_scalar("v", prof.caller_type, two_limb=False)
    return (
        f"struct {prof.printer_name} {{\n"
        f"    static void emit(std::string& out, {prof.cpp_scalar} v) {{ "
        f"out += dhex({recon}); }}\n"
        f"}};")


def render_group_driver(group_header: str,
                        target: TargetPrecision = TargetPrecision.DD) -> str:
    """Render the per-group driver ``.cpp`` for ``group_header`` at ``target`` precision.

    Identical in shape to ``boxGPU_dd.cpp`` (the proven oracle recipe) except it includes
    the single ``group_header`` (``box/B1m.h``) instead of ``boxGPU.h`` — so the pruned
    per-group ``BO`` is active and the TU is isolated to that mass group.  The precision
    ``#define`` (if any), the ``run_app`` template args, and the Printer all come from the
    profile, so retargeting to ff/float is a table change, not a code change (STOP #SS).
    """
    prof = profile_for(target)
    define_line = (f"#define {prof.define_macro}   "
                   f"// selects the precision arm in kokkosMaths_wrapper.h\n"
                   if prof.define_macro else "")
    complex_inc = ("#include <Kokkos_Complex.hpp>\n" if not prof.two_limb else "")
    return (
        f"// Pipeline-generated per-group precision-flip driver (Phase-1 template-arg "
        f"promotion).\n"
        f"// group={group_header}  precision={prof.precision.value}\n"
        f"//\n"
        f"// Whole-TU precision flip: includes ONLY the integral's mass-group header so the\n"
        f"// pruned per-group BO is active (QCDLOOP_BOX_FULL_DISPATCH stays undefined),\n"
        f"// isolating this group from the other groups' dd gaps.  Same mt19937(12345)\n"
        f"// recipes as the vanilla/dd drivers (shared boxGPU_app_recipes.hpp).\n"
        f"\n"
        f"{define_line}"
        f"#include <Kokkos_Core.hpp>\n"
        f"{complex_inc}"
        f"#include <string>\n"
        f"\n"
        f'#include "{group_header}"              // pruned per-group BO (no full dispatch)\n'
        f'#include "boxGPU_app_recipes.hpp"      // ql_app::dhex, ql_app::run_app\n'
        f"\n"
        f"namespace ql_app {{\n"
        f"{_printer_struct(prof)}\n"
        f"}}  // namespace ql_app\n"
        f"\n"
        f"int main(int argc, char* argv[]) {{\n"
        f"    return ql_app::run_app<{prof.cpp_output}, {prof.cpp_scalar},\n"
        f"                           {prof.cpp_scalar}, ql_app::{prof.printer_name}>("
        f"argc, argv);\n"
        f"}}\n")


# --------------------------------------------------------------------------- #
# emission (writes into the clone; refuses the snapshot)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class FlipTU:
    """Paths of the artifacts :func:`emit_flip_tu` wrote."""

    wrapper_path: Path
    driver_path: Path
    group_header: str
    target: TargetPrecision


def _refuse_snapshot(path: Path) -> None:
    """Fail loud if ``path`` is inside the pristine snapshot tree (STOP #Z)."""
    p = path.resolve()
    if p == _SNAPSHOT_DIR or _SNAPSHOT_DIR in p.parents:
        raise TUEmitError(
            f"refusing to write {p} under the pristine snapshot {_SNAPSHOT_DIR} "
            f"(STOP #Z); emit into a cloned tree instead")


def emit_flip_tu(clone_tree: Path, group_header: str, driver_dir: Path,
                 target: TargetPrecision = TargetPrecision.DD) -> FlipTU:
    """Emit the precision-flip build artifacts for one group into a cloned tree.

    Writes the precision-parameterized wrapper into ``clone_tree`` (overwriting the
    clone's copy, never the snapshot — guarded) and the per-group driver ``.cpp`` into
    ``driver_dir``.  Returns the paths.  ``clone_tree`` must be a real clone (contains
    ``boxGPU.h`` + ``box/``) and must NOT be the snapshot.
    """
    prof = profile_for(target)   # fail loud before any write if target unavailable
    clone_tree = Path(clone_tree).resolve()
    driver_dir = Path(driver_dir).resolve()
    wrapper_path = clone_tree / "kokkosMaths_wrapper.h"
    _refuse_snapshot(wrapper_path)
    if not (clone_tree / "boxGPU.h").is_file():
        raise TUEmitError(f"{clone_tree} is not a qcdloop header tree (no boxGPU.h)")
    if not (clone_tree / group_header).is_file():
        raise TUEmitError(f"group header {group_header} not found under {clone_tree}")

    wrapper_path.write_text(render_wrapper(target))
    driver_dir.mkdir(parents=True, exist_ok=True)
    grp_stem = Path(group_header).stem
    driver_path = driver_dir / f"boxGPU_flip_{grp_stem}_{prof.precision.value}.cpp"
    driver_path.write_text(render_group_driver(group_header, target))
    return FlipTU(wrapper_path=wrapper_path, driver_path=driver_path,
                  group_header=group_header, target=target)
