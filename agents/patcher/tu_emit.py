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

from agents.integrator_base.boundary import narrow_extended_scalar
from agents.patcher.precision_flip import TargetPrecision
from agents.patcher.shim_synth import read_embedded_sha, render_shim

# The master snapshot header tree — must stay pristine (STOP #Z).  Any emit target that
# resolves to a path under here is refused.
_SNAPSHOT_DIR = (Path(__file__).resolve().parents[2]
                 / "runs" / "qcdloop_headers_full").resolve()


class TUEmitError(RuntimeError):
    """An emission precondition failed (unavailable precision, snapshot write, etc.)."""


@dataclass(frozen=True)
class PrecisionProfile:
    """Everything precision-specific the wrapper + driver generators need for one target.

    A profile is *available* iff it can be served without enrichment.  Two mechanisms make
    a profile available:

    * a **static maths header** the snapshot vendors (``dd`` — ``kokkosMaths_dd.h``), or
    * **shim synthesis** (``shim_synthesis=True``): the target precision has no static
      wrapper but its leaf overloads have library-native instantiations, so the pipeline
      generates the missing non-template siblings as ``kokkosMaths_<precision>_shim.hpp`` on
      top of ``maths_reference_header`` (the double reference).  This is the Phase-2 float
      path; see :mod:`agents.patcher.shim_synth`.

    A precision served by a static header uses ``shim_synthesis=False`` + its own
    ``maths_header``.  ``ff`` is such a precision as of the ``kokkosMaths_ff.h`` enrichment
    (commit d0f5b35): its header layers the ``ql::`` leaves on the custom
    ``ql::ffun::ffcomplex`` container (not ``Kokkos::complex<FloatFloat>``, which fails the
    ``is_floating_point`` static_assert — the original STOP #EEE), so it joins the dd/quad
    static-header ladder.  A precision with neither a static header nor a shim path stays
    ``available=False`` and fails loud rather than degrading to dd (reverse-STOP #SS).
    """

    precision: TargetPrecision
    define_macro: str | None      # driver #define selecting this arm; None = default arm
    maths_header: str             # precision maths header the wrapper's arm includes
    cpp_output: str               # TOutput template arg (the complex container)
    cpp_scalar: str               # TMass / TScale template arg (the real scalar)
    printer_name: str             # ql_app Printer struct name emitted into the driver
    two_limb: bool                # scalar is an extended multi-limb aggregate (not native)
    available: bool               # servable without enrichment (static header OR shim)
    # The scalar's limb members, most-significant first — what the narrowing printer
    # sums to rebuild a caller-precision value.  Empty for a native single scalar.
    # ``two_limb`` says *whether* the scalar is an aggregate; ``limbs`` says which
    # members it actually has, which is not always ``("hi","lo")``: QuadFloat carries
    # four FP32 words ``f0..f3`` and defines no ``.hi``/``.lo`` at all.
    limbs: tuple[str, ...] = ("hi", "lo")
    caller_type: str = "double"   # the app-boundary caller precision the flip narrows to
    shim_synthesis: bool = False  # serve via a generated leaf shim (Phase-2 float downshift)
    maths_reference_header: str = "kokkosMaths.h"  # double reference the shim layers on
    reference_scalar: str = "double"               # scalar token the shim rewrites FROM


# Precision profile table.  DD is wired for Phase-1 (static header); FLOAT is wired for
# Phase-2 (shim synthesis — no static header, library-native leaves); FF is wired for
# Phase-2 via its own **static enrichment header** ``kokkosMaths_ff.h`` (commit d0f5b35),
# which layers the ``ql::`` leaves on the custom ``ql::ffun::ffcomplex`` container instead
# of ``Kokkos::complex<FloatFloat>`` — sidestepping STOP #EEE at the container level.  FF joins
# the same static-header ladder as dd/quad (STOP #SS: the ladder branches on the profile's
# ``maths_header`` value, never on a hard-coded precision name).
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
        available=True),           # enabled via kokkosMaths_ff.h enrichment (commit d0f5b35);
                                   # static wrapper + custom FloatFloatComplex container clears STOP #EEE
    TargetPrecision.QF: PrecisionProfile(
        precision=TargetPrecision.QF,
        define_macro="USE_QF_COMPLEX",
        maths_header="kokkosMaths_qf.h",
        cpp_output="ql::qfun::qfcomplex",
        cpp_scalar="ql::qfun::qfloat",
        printer_name="QFPrinter",
        two_limb=True,                     # extended aggregate — but FOUR limbs, not two
        limbs=("f0", "f1", "f2", "f3"),
        available=True),                   # enabled via kokkosMaths_qf.h enrichment; all
                                           # five box groups build + run at qf (T4 probe,
                                           # runs/qcdloop/qf_flip_probe.py)
    TargetPrecision.FLOAT: PrecisionProfile(
        precision=TargetPrecision.FLOAT,
        define_macro=None,                 # default arm (double reference) + generated shim
        maths_header="kokkosMaths.h",      # unused when shim_synthesis (kept for the table)
        cpp_output="Kokkos::complex<float>",
        cpp_scalar="float",
        printer_name="FloatPrinter",
        two_limb=False,
        limbs=(),                          # native single scalar — a plain cast narrows it
        available=True,                    # Phase-2: served by shim synthesis (no enrichment)
        shim_synthesis=True,
        maths_reference_header="kokkosMaths.h",
        reference_scalar="double"),
}

# The file name of the generated leaf shim, per precision (in the cloned TU only).
def _shim_filename(prof: PrecisionProfile) -> str:
    return f"kokkosMaths_{prof.precision.value}_shim.hpp"


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

    Two shapes, selected by the ``target`` profile:

    * **shim-synthesis** (Phase-2 float): a two-line wrapper — the double *reference*
      header followed by the generated leaf shim — because the target has no static maths
      header of its own; the shim supplies its missing non-template leaves on top of the
      reference (design §3.2).  The driver needs no ``#define`` for this arm.
    * **static header** (Phase-1 dd + quad + double default): the fork-shape ladder — one
      ``#if defined(<macro>) -> #include <header>`` arm per *available macro* profile, then
      the ``USE_QUAD_COMPLEX`` CUDA-quad arm (preserved from the snapshot), then the default
      double ``kokkosMaths.h``.  The wrapper carries every static arm; the *driver's*
      ``#define`` selects one.  This is the shape the fork's wrapper has and the snapshot's
      lacks (design §5.5).
    """
    prof = profile_for(target)   # validate the requested target is emittable
    if prof.shim_synthesis:
        return (
            "//\n"
            "// QCDLoop + Kokkos 2025 — precision wrapper (pipeline-generated, Phase-2 "
            "shim synthesis)\n"
            "//\n"
            f"// Target precision {prof.precision.value!r} has no static maths header: the\n"
            f"// double reference {prof.maths_reference_header!r} supplies every template "
            "leaf,\n"
            "// and the pipeline-synthesized shim below supplies the non-template leaf\n"
            "// siblings at the target precision (library-native bindings).\n"
            "//\n"
            "// Generated into the CLONED working tree only; the snapshot is pristine "
            "(STOP #Z).\n"
            "// Do not hand-edit — regenerate via agents.patcher.tu_emit.\n"
            "\n"
            "#pragma once\n"
            "\n"
            f'#include "{prof.maths_reference_header}"   // double REFERENCE header '
            "(unchanged)\n"
            f'#include "{_shim_filename(prof)}"   // pipeline-synthesized target-precision '
            "leaves\n")
    arms: list[str] = []
    arm_doc: list[str] = []
    for prof in PROFILES.values():
        if prof.available and prof.define_macro is not None:
            arms.append(f'#if defined({prof.define_macro})\n'
                        f'#include "{prof.maths_header}"')
            arm_doc.append(f"//   {prof.define_macro:<16} -> {prof.cpp_scalar} "
                           f"({prof.maths_header})\n")
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
        "// Selects the precision maths header from the driver's build define.  The arms\n"
        "// below are generated from the AVAILABLE profiles in agents.patcher.tu_emit, so\n"
        "// this list cannot drift from the ladder underneath it:\n"
        f"{''.join(arm_doc)}"
        "//   USE_QUAD_COMPLEX -> CUDA __nv_fp128 quad (CUDA only)\n"
        "//   neither          -> Kokkos::complex<double> (kokkosMaths.h)\n"
        "//\n"
        "// NB USE_QUAD_COMPLEX (CUDA __nv_fp128, one 128-bit word) and USE_QF_COMPLEX\n"
        "// (quad-FLOAT, four FP32 words) are different precisions despite both reading\n"
        "// as 'quad'.  They are separate arms and never interchangeable.\n"
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
        recon = narrow_extended_scalar("v", prof.caller_type, prof.limbs)
        return (
            f"struct {prof.printer_name} {{\n"
            f"    // app-output boundary narrow: extended -> caller precision (STOP #TT)\n"
            f"    static void emit(std::string& out, const {prof.cpp_scalar}& v) {{\n"
            f"        out += dhex({recon});\n"
            f"    }}\n"
            f"}};")
    # Native single-limb extended scalar (float): a plain cast is the reconstruction.
    recon = narrow_extended_scalar("v", prof.caller_type, prof.limbs)
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
    shim_path: Path | None = None   # generated leaf shim (shim-synthesis profiles only)


def _refuse_snapshot(path: Path) -> None:
    """Fail loud if ``path`` is inside the pristine snapshot tree (STOP #Z)."""
    p = path.resolve()
    if p == _SNAPSHOT_DIR or _SNAPSHOT_DIR in p.parents:
        raise TUEmitError(
            f"refusing to write {p} under the pristine snapshot {_SNAPSHOT_DIR} "
            f"(STOP #Z); emit into a cloned tree instead")


def _emit_shim(clone_tree: Path, prof: PrecisionProfile) -> Path:
    """Generate the leaf shim for a shim-synthesis profile into the clone (STOP #Z guarded).

    Reads the double *reference* header from the clone, extracts its non-template leaf
    inventory, and (re)writes ``kokkosMaths_<precision>_shim.hpp`` when absent or when the
    inventory sha differs from the existing shim's stamp — a cheap no-op when the reference
    header is unchanged (§3.4).  Never touches the reference header; refuses any write that
    resolves under the snapshot.
    """
    ref_path = clone_tree / prof.maths_reference_header
    if not ref_path.is_file():
        raise TUEmitError(
            f"shim reference header {prof.maths_reference_header!r} not found under "
            f"{clone_tree} — cannot synthesize the {prof.precision.value} leaf shim")
    shim_path = clone_tree / _shim_filename(prof)
    _refuse_snapshot(shim_path)
    reference_text = ref_path.read_text()
    shim_text = render_shim(reference_text,
                            reference_scalar=prof.reference_scalar,
                            target_scalar=prof.cpp_scalar,
                            reference_name=prof.maths_reference_header,
                            precision_label=prof.precision.value)
    # sha-keyed regeneration: reuse the cached shim iff its stamp matches the fresh one.
    if shim_path.is_file():
        want = read_embedded_sha(shim_text)
        have = read_embedded_sha(shim_path.read_text())
        if want is not None and want == have:
            return shim_path
    shim_path.write_text(shim_text)
    return shim_path


def emit_flip_tu(clone_tree: Path, group_header: str, driver_dir: Path,
                 target: TargetPrecision = TargetPrecision.DD) -> FlipTU:
    """Emit the precision-flip build artifacts for one group into a cloned tree.

    Writes the precision-parameterized wrapper into ``clone_tree`` (overwriting the
    clone's copy, never the snapshot — guarded) and the per-group driver ``.cpp`` into
    ``driver_dir``.  For a **shim-synthesis** profile (Phase-2 float) it additionally
    generates the target-precision leaf shim into the clone (:func:`_emit_shim`).  Returns
    the paths.  ``clone_tree`` must be a real clone (contains ``boxGPU.h`` + ``box/``) and
    must NOT be the snapshot.
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

    shim_path = _emit_shim(clone_tree, prof) if prof.shim_synthesis else None
    wrapper_path.write_text(render_wrapper(target))
    driver_dir.mkdir(parents=True, exist_ok=True)
    grp_stem = Path(group_header).stem
    driver_path = driver_dir / f"boxGPU_flip_{grp_stem}_{prof.precision.value}.cpp"
    driver_path.write_text(render_group_driver(group_header, target))
    return FlipTU(wrapper_path=wrapper_path, driver_path=driver_path,
                  group_header=group_header, target=target, shim_path=shim_path)
