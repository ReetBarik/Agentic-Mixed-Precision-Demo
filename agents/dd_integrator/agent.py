"""DD-integrator agent — STUB (whole-app double-double integration).

Sibling of :mod:`agents.tracked_integrator.agent`: given a target library's
headers and a driver that exercises them, its job is to make the library callable
with a **double-double** (~30–31 digit) numeric type so a candidate patch's
precision loss can be measured against a high-precision ground truth.  The
Validator invokes it to produce the DD ground-truth build (see
``agents/validator``).

=============================  STUB STATUS  ================================

This is a STUB.  It does **no** LLM generation.  It only supports the one
application we already have a hand-written double-double port for —
``qcdloop@ddfun_enabled`` — and for that app it simply confirms the DD headers
are present and returns the path to the existing, hand-written
``kokkosMaths_dd.h``.  For any other application it fails loudly rather than
guessing.

Concretely, :func:`integrate`:

* verifies the caller pointed it at a qcdloop tree carrying the ddfun_enabled DD
  triple — ``kokkosMaths_dd.h`` + ``dd_math.hpp`` + ``dd_complex.hpp`` (the
  ``ql::ddfun`` fork, co-located in ``src/qcdloop/``);
* if present, returns the path to that ``kokkosMaths_dd.h`` — no LLM call, no
  generation, nothing written;
* if absent, raises ``RuntimeError`` making clear this stub only supports
  ``qcdloop@ddfun_enabled`` and that a real dd_integrator is needed for other
  apps.

===========================  DEFERRED REAL WORK  ==========================

The real dd_integrator will mirror :mod:`agents.tracked_integrator`: an
LLM-driven generation pass, reusing the shared machinery in
:mod:`agents.integrator_base` (SOURCE_HASH cache, streaming shim, bounded retry
loop, C8 boundary patcher parameterized on the DD scalar type name, e.g.
``quad::ddfun::ddouble``).  It would target the canonical vendored DD headers at
``third_party/include/{dd_math.hpp, dd_complex.hpp}`` and generate a
``<app>_dd.h``-style shim (``ql::Constants`` specializations + math overloads for
the DD complex/real types) for an arbitrary app.

The one DD-specific wrinkle beyond the tracked integrator: the DD constant tables
(Chebyshev / Bernoulli / π / log2 …) must be materialized as **hex-encoded
``(hi, lo)`` double pairs**, because a decimal literal only carries ~16 digits and
would silently truncate the low word.  qcdloop's ``ddfun_enabled`` branch does
this offline via ``scripts/gen_dd_constants.cpp`` (splits a quad-precision literal
into the two doubles and emits ``make_dd(0x…, 0x…)`` lines pasted into
``kokkosMaths_dd.h``).  The real dd_integrator must drive an equivalent codegen
step so constants survive at full DD precision — not emit decimal literals.
"""

from __future__ import annotations

from pathlib import Path

from agents.integrator_base.region import RegionIntegrationResult

# The ddfun_enabled DD triple that marks a qcdloop tree as DD-ready.  All three
# must be co-located (same directory) — a bare kokkosMaths_dd.h without the
# dd_math/dd_complex it includes is not a usable DD tree.
_DD_TRIPLE = ("kokkosMaths_dd.h", "dd_math.hpp", "dd_complex.hpp")

_UNSUPPORTED_MSG = (
    "dd_integrator STUB: no double-double headers found under {root}. This stub "
    "only supports qcdloop@ddfun_enabled — it expects the hand-written DD triple "
    "({triple}) co-located in the target tree (e.g. src/qcdloop/). A real "
    "dd_integrator (LLM-driven DD shim generation, mirroring tracked_integrator) "
    "is needed for other apps; see this module's docstring."
)


def integrate(
    target_library_headers,
    driver_source_path,
    dd_library_path=None,
    existing_shim=None,
    *,
    cfg=None,
    out_path=None,
    app_name=None,
) -> Path:
    """Return the path to the hand-written ``kokkosMaths_dd.h`` for a DD-ready tree.

    Signature mirrors :func:`agents.tracked_integrator.agent.integrate` (with
    ``dd_library_path`` in the slot ``tracked_repo_path`` occupied there — it will
    point at the canonical vendored DD headers once the real integrator lands).

    Parameters
    ----------
    target_library_headers:
        Path to the target library's header directory (or a tree root containing
        it).  Must carry the ddfun_enabled DD triple; searched recursively.
    driver_source_path:
        Unused by the stub (accepted for API symmetry).  The real integrator will
        read it to know which library symbols to shim.
    dd_library_path:
        Unused by the stub.  The real integrator will read the canonical DD
        headers (``third_party/include/``) from here.
    existing_shim, cfg, out_path, app_name:
        Unused by the stub (accepted for API symmetry).  A real integrator would
        use ``cfg`` for the LLM call, ``existing_shim`` to extend, ``out_path`` /
        ``app_name`` for placement — driven through :mod:`agents.integrator_base`.

    Returns
    -------
    Path
        The existing hand-written ``kokkosMaths_dd.h``.

    Raises
    ------
    RuntimeError
        If the target tree carries no co-located DD triple (i.e. it is not a
        ``qcdloop@ddfun_enabled`` tree the stub can serve).
    """
    root = Path(target_library_headers).resolve()
    if not root.is_dir():
        raise NotADirectoryError(
            f"target_library_headers is not a directory: {root}"
        )

    dd_header = _locate_dd_triple(root)
    if dd_header is None:
        raise RuntimeError(
            _UNSUPPORTED_MSG.format(root=root, triple=", ".join(_DD_TRIPLE))
        )
    return dd_header


def _locate_dd_triple(root: Path) -> Path | None:
    """Find a directory under ``root`` holding all of :data:`_DD_TRIPLE`.

    Returns the path to that directory's ``kokkosMaths_dd.h``, or ``None`` if no
    directory carries the full co-located triple.  Checks ``root`` and the
    conventional ``root/src/qcdloop`` first, then any directory containing a
    ``kokkosMaths_dd.h`` anywhere under ``root``.
    """
    def _has_triple(d: Path) -> bool:
        return all((d / name).is_file() for name in _DD_TRIPLE)

    preferred = [root, root / "src" / "qcdloop"]
    for d in preferred:
        if _has_triple(d):
            return (d / "kokkosMaths_dd.h").resolve()

    for cand in sorted(root.rglob("kokkosMaths_dd.h")):
        if cand.is_file() and _has_triple(cand.parent):
            return cand.resolve()
    return None


_REGION_STUB_MSG = (
    "dd_integrator.integrate_region is a BOUNDED STUB (scope decision (b), see "
    "HANDOFF.md): the P7 region contract, cheap validation and the "
    "RegionIntegrationResult return shape are implemented, but LLM-driven regional "
    "double-double generation is deferred. Beyond the ff twin, a real "
    "implementation must materialize any DD constant table this region touches as "
    "hex-encoded (hi, lo) double pairs (see this module's docstring / qcdloop's "
    "scripts/gen_dd_constants.cpp) so constants survive at full DD precision. The "
    "Patcher exercises this path through an *injected* integrator today. "
    "Region requested: {file}:{line_start}-{line_end} scalar={scalar_type}."
)


def integrate_region(
    *,
    file: str,
    line_start: int,
    line_end: int,
    variables: list[str],
    working_tree: str,
    scalar_type: str = "ddouble",
    direction: str = "in",
    out_dir: Path,
    attempt: int = 0,
    repo_path: str | None = None,
) -> RegionIntegrationResult:
    """Regional double-double promotion (P7) — sibling of :func:`integrate`.

    Signature mirrors ``ff_integrator.integrate_region`` exactly, with
    ``scalar_type="ddouble"`` (design §P7 "one module, two functions").  Returns
    the shared :class:`RegionIntegrationResult` (shim path(s) + boundary patch).

    BOUNDED STUB — see :data:`_REGION_STUB_MSG` / HANDOFF.md scope decision (b).
    The DD-specific wrinkle beyond the ff path is constant-table hex codegen
    (hex-encoded ``(hi, lo)`` double pairs); real generation reusing
    :mod:`agents.integrator_base` is deferred.  The Patcher consumes an injected
    integrator for this path today.
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    raise NotImplementedError(_REGION_STUB_MSG.format(
        file=file, line_start=line_start, line_end=line_end, scalar_type=scalar_type))
