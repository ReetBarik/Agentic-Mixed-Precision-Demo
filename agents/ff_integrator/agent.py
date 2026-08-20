"""FF-integrator agent — regional float-float promotion.

Sibling of :mod:`agents.dd_integrator` and :mod:`agents.tracked_integrator`, but
with a deliberately **different scope and API**.  Where the tracked and dd
integrators are *whole-app* (instrument / DD-promote an entire library so it is
callable with an instrumented / extended type), the ff integrator is
*regional*: the Patcher hands it a specific code region — a file + line range and
the set of variables in play — and it promotes **just that region** to a
float-float (``Kokkos::Experimental::FloatFloat`` / ``Kokkos::Experimental::FloatFloatComplex``) representation,
producing a small shim plus a boundary patch that converts at the region's edges
and leaves the rest of the app untouched.

Float-float ("single-single") emulates ~double precision from two ``float``s; the
strategy is the catalog's "float-float emulation (DD recovery)" entry — recover
lost precision in a hot region cheaply, without paying whole-app double-double.

Implementation.  :func:`integrate_region` is a thin wrapper over the shared
engine :func:`agents.integrator_base.regional.run_integrate_region` (the ff/dd
twins share everything but their ruleset + concrete C++ type spellings): it reads
the region at the pinned SHA, recovers the write set (Fix C), asks the LLM for an
``FloatFloat``/``FloatFloatComplex`` shim (SOURCE_HASH-cached, ``attempt``-varied for the
Patcher's N=3 retry), and pairs it with the deterministic boundary patch from
:mod:`agents.integrator_base.boundary` (promote reads on entry, demote writes on
exit).  The float-float type itself is *vendored* (``third_party/include/
ff_math.hpp`` + ``ff_complex.hpp``), not LLM-generated — the shim only adds the
named-constant wrappers and missing operators the region references.

Whole-app ff mode (:func:`integrate`) is a documented follow-up and raises
``NotImplementedError``; the regional path is what the Patcher drives.
"""

from __future__ import annotations

from pathlib import Path

from agents.integrator_base import regional
from agents.integrator_base.region import RegionIntegrationResult

_NOT_IMPLEMENTED_MSG = (
    "ff_integrator.integrate (whole-app) is not implemented: only the *regional* "
    "float-float promotion path (integrate_region) is built. Whole-app ff mode is "
    "a documented follow-up (see docs/KNOWN_LIMITATIONS.md); the regional path is what the Patcher "
    "drives per remediation intent."
)

# ---------------------------------------------------------------------------
# Regional ruleset (the LLM system prompt) + the concrete C++ type spellings.
# The prompt bytes are the single source of truth for the ruleset hash folded
# into every regional shim's SOURCE_HASH (cache.compute_region_hash).
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.txt").read_text(encoding="utf-8")

_SPEC = regional.RegionalSpec(
    system_prompt=_SYSTEM_PROMPT,
    cpp_scalar="Kokkos::Experimental::FloatFloat",
    cpp_complex="Kokkos::Experimental::FloatFloatComplex",
    vendored_headers=["ff_math.hpp", "ff_complex.hpp"],
    shim_prefix="ff",
)

# Public alias — the Patcher fan-out (Phase 2a) reads the concrete C++ scalar
# spelling / two_limb flag from here so type spellings have one source of truth.
SPEC = _SPEC



def integrate(*args, **kwargs):
    """Whole-app ff mode — not implemented (documented follow-up).

    Kept for structural symmetry with the tracked / dd integrators.  The regional
    path (:func:`integrate_region`) is the one the Patcher drives.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def integrate_region(
    *,
    file: str,
    line_start: int,
    line_end: int,
    variables: list[str],
    working_tree: str,
    scalar_type: str = "FloatFloat",
    caller_type: str = "double",
    direction: str = "in",
    out_dir: Path,
    attempt: int = 0,
    repo_path: str | None = None,
    cfg=None,
    llm_fn=None,
) -> RegionIntegrationResult:
    """Regional float-float promotion (design §P4 call shape).

    Reads the region source at ``working_tree`` (a SHA), asks the LLM for a
    ``Kokkos::Experimental::FloatFloat`` / ``Kokkos::Experimental::FloatFloatComplex`` shim, and pairs it with a
    deterministic boundary patch (promote reads on entry, demote writes on exit).
    Returns the shared :class:`RegionIntegrationResult` (shim path(s) + boundary
    patch + token count).

    ``scalar_type`` is the Patcher's short tag (``"FloatFloat"``); the concrete C++
    spelling comes from :data:`_SPEC`.  ``llm_fn(system, user, attempt) -> str`` is
    a test seam (``None`` -> real streaming call via ``cfg`` /
    :class:`~agents.config.PipelineConfig`).  Never raises past the seam — failures
    return an ``llm_failed`` result for the Patcher's bounded retry (P4).
    """
    return regional.run_integrate_region(
        _SPEC,
        file=file, line_start=line_start, line_end=line_end,
        variables=variables, working_tree=working_tree,
        scalar_type=scalar_type, caller_type=caller_type, direction=direction,
        out_dir=out_dir, attempt=attempt, repo_path=repo_path,
        cfg=cfg, llm_fn=llm_fn,
    )
