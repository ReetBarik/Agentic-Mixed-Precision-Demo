"""Float-integrator agent — regional demotion to plain 32-bit ``float``.

Sibling of :mod:`agents.ff_integrator` and :mod:`agents.dd_integrator`, and the
first extension of the regional integrator ruleset to a NEW target type since ff
and dd.  Where the ff/dd integrators *promote* a hot region to a foreign extended
scalar (``quad::ffun::ffloat`` / ``quad::ddfun::ddouble``) to recover precision,
the float integrator *demotes* the region to the native builtin ``float`` for
speedup — fewer bytes moved, wider SIMD, faster transcendentals — leaving the rest
of the app at ``double``.

Why a regional/LLM path for float at all.  A non-templated region carries a bare
``double`` keyword token, so the Patcher's cheap plain-type-edit
(``double``→``float``) rewrites it directly — no LLM needed, and that path is kept
for such regions.  But a template-typed HPC kernel writes the region in terms of a
template parameter (``TOutput`` / ``TScale``), with no literal ``double`` token to
rewrite; the plain edit is mechanically inapplicable there (CALIBRATION.md §Bug 4,
"0 to float"), so float was previously unreachable on the whole template surface.
This agent makes float reachable on template-typed code by generating a
``float``-specialized regional shim exactly as the ff/dd integrators generate their
extended-scalar shims — the substitution is structurally identical, only the target
type differs.

Implementation.  :func:`integrate_region` is a thin wrapper over the shared engine
:func:`agents.integrator_base.regional.run_integrate_region` with a ``float``
:class:`~agents.integrator_base.regional.RegionalSpec`.  Three spec knobs capture
what makes ``float`` a *native* target rather than a two-limb extended one:

* ``two_limb=False`` — ``float`` has no ``{hi, lo}`` limbs, so the boundary patch
  demotes region writes with a plain ``static_cast<caller>(w)`` instead of two-limb
  reconstruction (which would reference nonexistent ``.hi``/``.lo`` and never
  compile);
* ``emit_bridges=False`` — a native ``float`` argument binds a ``double`` overload
  by an implicit widening conversion, so the Gap-A namespace-qualified bridge lint
  (which guards against narrowing a *foreign* extended value) does not apply;
* ``derive_constants=False`` — ``float`` carries no sub-limb precision, so a source
  literal is just its ``float`` literal and the app's own ``Constants<float>`` (in
  scope at the include site) supplies named constants; the two-limb Bailey-split
  derivation is irrelevant.

The escape hatch (Rule 6 / R4 in the ruleset): when the integrator cannot decide
whether a use-site wants ``float`` or must stay ``double`` (a ``double`` driver
constant like ``M_PI`` threaded in), the model emits ``#error`` — a hard build
failure beats a silent precision slip.

Whole-app float mode (:func:`integrate`) is a documented follow-up and raises
``NotImplementedError``; the regional path is what the Patcher drives.
"""

from __future__ import annotations

from pathlib import Path

from agents.integrator_base import regional
from agents.integrator_base.region import RegionIntegrationResult

_NOT_IMPLEMENTED_MSG = (
    "float_integrator.integrate (whole-app) is not implemented: only the *regional* "
    "float demotion path (integrate_region) is built, mirroring ff_integrator. The "
    "regional path is what the Patcher drives per remediation intent."
)

# ---------------------------------------------------------------------------
# Regional ruleset (the LLM system prompt) + the concrete C++ type spellings.
# The prompt bytes are the single source of truth for the ruleset hash folded
# into every regional shim's SOURCE_HASH (cache.compute_region_hash); a brand-new
# prompt file means float shims never collide with ff/dd shims in the cache.
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.txt").read_text(encoding="utf-8")

_SPEC = regional.RegionalSpec(
    system_prompt=_SYSTEM_PROMPT,
    cpp_scalar="float",
    cpp_complex="std::complex<float>",
    vendored_headers=[],            # float is a builtin — no vendored header
    shim_prefix="float",
    # float is a NATIVE single-limb type — see the module docstring.
    two_limb=False,
    emit_bridges=False,
    derive_constants=False,
)


def integrate(*args, **kwargs):
    """Whole-app float mode — not implemented (documented follow-up).

    Kept for structural symmetry with the ff / dd / tracked integrators.  The
    regional path (:func:`integrate_region`) is the one the Patcher drives.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)


def integrate_region(
    *,
    file: str,
    line_start: int,
    line_end: int,
    variables: list[str],
    working_tree: str,
    scalar_type: str = "float",
    caller_type: str = "double",
    direction: str = "in",
    out_dir: Path,
    attempt: int = 0,
    repo_path: str | None = None,
    cfg=None,
    llm_fn=None,
) -> RegionIntegrationResult:
    """Regional demotion to native ``float`` (mirrors ``ff_integrator.integrate_region``).

    Reads the region source at ``working_tree`` (a SHA), asks the LLM for a
    ``float`` shim (the types/operators/constants the region needs once its reads
    are demoted to ``float``), and pairs it with a deterministic boundary patch
    (demote reads on entry, widen writes back to ``caller_type`` on exit).  Returns
    the shared :class:`RegionIntegrationResult` (shim path(s) + boundary patch +
    token count).

    ``scalar_type`` is the Patcher's short tag (``"float"``); the concrete C++
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
