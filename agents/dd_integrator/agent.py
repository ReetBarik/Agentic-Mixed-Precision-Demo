"""DD-integrator agent — STUB (whole-app double-double integration).

Sibling of :mod:`agents.tracked_integrator.agent`: given a target library's
headers and a driver that exercises them, its job is to make the library callable
with a **double-double** (~30–31 digit) numeric type so a candidate patch's
precision loss can be measured against a high-precision ground truth.  The
Validator invokes it to produce the DD ground-truth build (see
``agents/validator``).

=============================  STUB STATUS  ================================

Two entrypoints, two scopes.  :func:`integrate_region` (regional DD promotion,
design §P7) is **implemented** — a thin wrapper over the shared engine
:func:`agents.integrator_base.regional.run_integrate_region`, LLM-driven, sibling
to ``ff_integrator.integrate_region``.  :func:`integrate` (whole-app DD) is still
a **STUB**: it does **no** LLM generation and only supports the one application we
already have a hand-written double-double port for — ``qcdloop@ddfun_enabled`` —
for which it confirms the DD headers are present and returns the path to the
existing hand-written ``kokkosMaths_dd.h`` (Validator's ground-truth build path).
For any other application it fails loudly rather than guessing.

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
``Kokkos::Experimental::DoubleDouble``).  It would target the canonical vendored DD headers at
``third_party/include/{dd_math.hpp, dd_complex.hpp}`` and generate a
``<app>_dd.h``-style shim (``ql::Constants`` specializations + math overloads for
the DD complex/real types) for an arbitrary app.

The one DD-specific wrinkle beyond the tracked integrator: the DD constant tables
(Chebyshev / Bernoulli / π / log2 …) must be materialized as **hex-encoded
``(hi, lo)`` double pairs**, because a decimal literal only carries ~16 digits and
would silently truncate the low word.  qcdloop's ``ddfun_enabled`` branch does
this offline via ``scripts/gen_dd_constants.cpp`` (splits a quad-precision literal
into the two doubles and emits ``DoubleDouble::from_bits(0x…, 0x…)`` lines pasted into
``kokkosMaths_dd.h``).  The real dd_integrator must drive an equivalent codegen
step so constants survive at full DD precision — not emit decimal literals.
"""

from __future__ import annotations

from pathlib import Path

from agents.integrator_base import regional
from agents.integrator_base.region import RegionIntegrationResult

# ---------------------------------------------------------------------------
# Regional ruleset + concrete C++ type spellings (mirrors ff_integrator; the
# shared engine in agents/integrator_base/regional.py does the work).  The prompt
# bytes feed the regional shim's SOURCE_HASH via cache.compute_region_hash.  The
# DD-specific wrinkle beyond ff is the hex-encoded (hi, lo) constant-table note.
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.txt").read_text(encoding="utf-8")

_SPEC = regional.RegionalSpec(
    system_prompt=_SYSTEM_PROMPT,
    cpp_scalar="Kokkos::Experimental::DoubleDouble",
    cpp_complex="Kokkos::Experimental::DoubleDoubleComplex",
    vendored_headers=["dd_math.hpp", "dd_complex.hpp"],
    shim_prefix="dd",
    constant_note=(
        "## DD constant tables (hard requirement)\n"
        "Any double-double constant this region needs MUST be materialized at full "
        "precision — never as a decimal literal (it truncates the low word). Resolve "
        "each via the Rule R3 cascade IN ORDER: (1) a vendored `dd_*()` free function; "
        "(2) a known `DoubleDouble::from_bits(0x<hi>ULL, 0x<lo>ULL)` hex pair; (3) derive from the "
        "constant's own source definition — a source `double` literal (e.g. "
        "`TScale(1e-50)`) promotes to `DoubleDouble::from_bits(<bits of that double>, 0x0)` with a ZERO "
        "low word (correct — a source literal has only double precision), and a "
        "closed form over catalog constants composes from their known pairs; "
        "(4) only if none apply, the Rule R4 #error. Any values pre-derived for you "
        "appear under 'Source-derivable constants' — use them verbatim."
    ),
)

# Public alias — the Patcher fan-out (Phase 2a) reads the concrete C++ scalar
# spelling / two_limb flag from here so type spellings have one source of truth.
SPEC = _SPEC


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


def integrate_region(
    *,
    file: str,
    line_start: int,
    line_end: int,
    variables: list[str],
    working_tree: str,
    scalar_type: str = "DoubleDouble",
    caller_type: str = "double",
    direction: str = "in",
    out_dir: Path,
    attempt: int = 0,
    repo_path: str | None = None,
    cfg=None,
    llm_fn=None,
) -> RegionIntegrationResult:
    """Regional double-double promotion (P7) — sibling of :func:`integrate`.

    Signature mirrors ``ff_integrator.integrate_region`` exactly, with
    ``scalar_type="DoubleDouble"`` (design §P7 "one module, two functions"); the concrete
    C++ spelling (``Kokkos::Experimental::DoubleDouble`` / ``DoubleDoubleComplex``) comes from :data:`_SPEC`.
    Thin wrapper over the shared engine
    :func:`agents.integrator_base.regional.run_integrate_region` — reads the region
    at ``working_tree``, recovers writes (Fix C), LLM-generates a DoubleDouble shim
    (SOURCE_HASH-cached, ``attempt``-varied), and pairs it with a deterministic
    boundary patch.  The DD-specific wrinkle beyond the ff twin — hex-encoded
    ``(hi, lo)`` constant tables — is codified in the ruleset (Rule R3) and the
    ``constant_note`` carried on :data:`_SPEC`.  Returns the shared
    :class:`RegionIntegrationResult`; never raises past the seam.
    """
    return regional.run_integrate_region(
        _SPEC,
        file=file, line_start=line_start, line_end=line_end,
        variables=variables, working_tree=working_tree,
        scalar_type=scalar_type, caller_type=caller_type, direction=direction,
        out_dir=out_dir, attempt=attempt, repo_path=repo_path,
        cfg=cfg, llm_fn=llm_fn,
    )
