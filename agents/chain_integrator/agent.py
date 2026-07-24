"""Chain-integrator agent — regional double-double promotion for CHAIN links (Phase 2f).

Peer of :mod:`agents.dd_integrator`.  A cancellation-cascade *chain* is widened to
double-double as a coordinated envelope by :mod:`agents.patcher.chain_promote`; that
deterministic core inserts the variants, reroutes, and boundary casts.  This agent
supplies the LLM half — the per-region ddouble shim (types, operators, named
constants) each chain link needs to compile — exactly as ``dd_integrator`` does for a
lone region, but under a ruleset carrying the extra **C9 (chain-boundary
coordination)** rule: a value a link produces that is read by the next link stays
ddouble (chain-internal contract), and only the chain's outermost read/write converts
to caller precision (chain-boundary contract, owned by the boundary patch).

Because the target is the same vendored double-double type as ``dd_integrator``
(``quad::ddfun::ddouble`` / ``ddcomplex`` from ``third_party/include/``), the shim
merges into the SAME canonical per-family header ``ql_shim_dd.h`` (``shim_prefix="dd"``).
The C9-augmented system prompt gives this integrator its own SOURCE_HASH silo, so a
chain link's shim is cached independently of a lone dd region's — both coexist in
``ql_shim_dd.h`` (signature-dedup, keep-first).

There is no whole-app scope for a chain integrator (a chain is intrinsically
regional), so :func:`integrate` raises ``NotImplementedError`` for API symmetry with
the ff/dd peers; :func:`integrate_region` is the real entrypoint.
"""

from __future__ import annotations

from pathlib import Path

from agents.integrator_base import regional
from agents.integrator_base.region import RegionIntegrationResult

# The chain ruleset = the dd regional ruleset + the C9 chain-boundary rule.  Its
# bytes feed the regional shim's SOURCE_HASH via cache.compute_region_hash, so this
# prompt siloes chain shims from lone-dd shims (both still land in ql_shim_dd.h).
_SYSTEM_PROMPT = (Path(__file__).parent / "system_prompt.txt").read_text(encoding="utf-8")

_SPEC = regional.RegionalSpec(
    system_prompt=_SYSTEM_PROMPT,
    cpp_scalar="quad::ddfun::ddouble",
    cpp_complex="quad::ddfun::ddcomplex",
    vendored_headers=["dd_math.hpp", "dd_complex.hpp"],
    shim_prefix="dd",                 # merge into the shared ql_shim_dd.h
    constant_note=(
        "## DD constant tables (hard requirement)\n"
        "Any double-double constant this region needs MUST be materialized at full "
        "precision — never as a decimal literal (it truncates the low word). Resolve "
        "each via the Rule R3 cascade IN ORDER: (1) a vendored `dd_*()` free function; "
        "(2) a known `make_dd(0x<hi>ULL, 0x<lo>ULL)` hex pair; (3) derive from the "
        "constant's own source definition — a source `double` literal (e.g. "
        "`TScale(1e-50)`) promotes to `make_dd(<bits of that double>, 0x0)` with a ZERO "
        "low word (correct — a source literal has only double precision), and a "
        "closed form over catalog constants composes from their known pairs; "
        "(4) only if none apply, the Rule R4 #error. Any values pre-derived for you "
        "appear under 'Source-derivable constants' — use them verbatim.\n"
        "## Chain-boundary contract (C9)\n"
        "This region is one link of a promoted chain: values it produces that are read "
        "by another link stay ddouble (do not truncate at the region exit); only the "
        "chain's outermost read/write converts to caller precision (the boundary patch "
        "does that). Any overload you supply for a function that is itself on the chain "
        "MUST return ddouble, never a value narrowed to double."
    ),
)

# Public alias — the Patcher chain path (Phase 2f) reads the concrete C++ scalar
# spelling / two_limb flag from here so type spellings have one source of truth.
SPEC = _SPEC


def integrate(*args, **kwargs):
    """No whole-app scope for a chain integrator — chains are intrinsically regional.

    Present only for API symmetry with the ff/dd peers.  Use
    :func:`integrate_region` (driven by :mod:`agents.patcher.chain_promote`)."""
    raise NotImplementedError(
        "chain_integrator has no whole-app scope; a chain is intrinsically regional — "
        "use integrate_region (driven by agents.patcher.chain_promote)")


def integrate_region(
    *,
    file: str,
    line_start: int,
    line_end: int,
    variables: list[str],
    working_tree: str,
    scalar_type: str = "ddouble",
    caller_type: str = "double",
    direction: str = "in",
    out_dir: Path,
    attempt: int = 0,
    repo_path: str | None = None,
    cfg=None,
    llm_fn=None,
) -> RegionIntegrationResult:
    """Generate the ddouble shim for one chain-link region (P7 / C9).

    Signature mirrors ``dd_integrator.integrate_region`` exactly; the only difference
    is the C9-augmented ruleset carried on :data:`_SPEC`.  Thin wrapper over the shared
    engine :func:`agents.integrator_base.regional.run_integrate_region` — reads the
    region at ``working_tree``, recovers writes, LLM-generates a ddouble shim
    (SOURCE_HASH-cached, ``attempt``-varied), and pairs it with a deterministic
    boundary patch.  Returns the shared :class:`RegionIntegrationResult`; never raises
    past the seam.
    """
    return regional.run_integrate_region(
        _SPEC,
        file=file, line_start=line_start, line_end=line_end,
        variables=variables, working_tree=working_tree,
        scalar_type=scalar_type, caller_type=caller_type, direction=direction,
        out_dir=out_dir, attempt=attempt, repo_path=repo_path,
        cfg=cfg, llm_fn=llm_fn,
    )
