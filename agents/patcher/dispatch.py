"""P3 dispatch — map a ``kind`` to one of four code paths and run its generator.

The 11 kinds collapse to four dispatch paths (design §P3 "Four dispatch paths"):

* **regional-integrator** — ``float-to-ff``, ``double-to-ff`` (→ ff shim),
  ``double-to-dd`` (→ dd shim), and ``ff-to-dd`` (composite: revert ff, then
  install dd).  Calls ``{ff,dd}_integrator.integrate_region``.
* **plain-type-edit** — ``float-to-double``, ``double-to-float``.  AST/keyword
  type-node rewrite (see :mod:`agents.patcher.edits`).
* **git-revert** — ``ff-to-double``, ``dd-to-double`` (strip a prior install),
  and ``ff-to-float`` (composite: revert ff, then plain-edit double→float).
* **llm-rewrite** — ``reformulate-kahan``, ``reformulate-identity``.

Each generator mutates the already-clean working tree and returns a :class:`Gen`.
It never builds or commits — that is the caller's (agent.py) shared gate+commit
step, so the bounded retry (P4) can wrap generation + build uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from agents.integrator_base import boundary
from agents.integrator_base.region import RegionIntegrationResult
from agents.patcher import edits, gitops, result as R, rewrites
from agents.strategy.models import (
    TRANSITION_KINDS, VIA_CHAIN, VIA_REGIONAL, RemediationIntent)

# ---- dispatch-path tags ----
PATH_REGIONAL = "regional"
PATH_PLAIN_EDIT = "plain_edit"
PATH_REVERT = "revert"
PATH_LLM_REWRITE = "llm_rewrite"
PATH_CHAIN = "chain"           # Phase 2f: coordinated whole-chain dd promotion

_TO_FF = frozenset({"float-to-ff", "double-to-ff"})
_TO_DD = frozenset({"double-to-dd"})
_PLAIN = frozenset({"float-to-double", "double-to-float"})
_REVERT = frozenset({"ff-to-double", "dd-to-double", "ff-to-float"})
_REWRITE = frozenset({"reformulate-kahan", "reformulate-identity"})
# ``-to-float`` demotions that, when Strategy tags them ``via="regional"`` (a
# template-typed region with no bare ``double`` token), route to the LLM/regional
# float integrator instead of the plain-edit / git-revert path (Wave 2).
_TO_FLOAT = frozenset({"double-to-float", "ff-to-float"})


def dispatch_path(kind: str, via: str = "plain") -> str:
    # Phase 2f: a chain-scoped promotion (via="chain") is realized as a coordinated
    # whole-chain dd envelope regardless of the (double-to-dd) kind.
    if via == VIA_CHAIN:
        return PATH_CHAIN
    # Wave 2: a ``-to-float`` demotion on a template-typed region (via="regional")
    # is realized by generating a float-specialized shim, exactly like ff/dd.
    if via == VIA_REGIONAL and kind in _TO_FLOAT:
        return PATH_REGIONAL
    if kind in _TO_FF or kind in _TO_DD or kind == "ff-to-dd":
        return PATH_REGIONAL
    if kind in _PLAIN:
        return PATH_PLAIN_EDIT
    if kind in _REVERT:
        return PATH_REVERT
    if kind in _REWRITE:
        return PATH_LLM_REWRITE
    raise ValueError(f"no dispatch path for kind {kind!r}")


def is_llm_driven(path: str) -> bool:
    """LLM-driven paths get the N=3 bounded retry (P4); deterministic ones don't."""
    return path in (PATH_REGIONAL, PATH_LLM_REWRITE, PATH_CHAIN)


def is_retryable_misgen(gate) -> bool:
    """P4a — "retry everything, for now".

    The regional-integrator misgen patterns don't exist yet, so we can't classify
    a build failure as a genuine compile error vs an LLM misgen.  Start by
    retrying every (non-timeout) gate failure on an LLM-driven path; add
    pattern-matching once Stage-4 failure data lands.
    """
    return True


@dataclass
class Gen:
    """Outcome of one generation attempt (before the build gate)."""
    ok: bool
    status: str | None = None          # failure status when not ok
    err_kind: str | None = None
    detail: str | None = None
    shim_paths: list[str] = field(default_factory=list)
    boundary_patch_path: str | None = None
    llm_tokens: int = 0
    # Phase-2a fan-out extras (empty on the classic regional / non-fan-out paths):
    declared_variants: list[str] = field(default_factory=list)
    files_touched: list[str] = field(default_factory=list)
    in_place_region: bool = False      # region was IN the entry point (no new symbol)


@dataclass
class PatchDeps:
    repo_root: Path
    parent_sha: str
    target_path: Path
    shims_dir: Path
    patches_dir: Path
    # integrators: {"ff": callable, "dd": callable} with the integrate_region shape
    integrators: dict
    # llm_call(system, user, attempt) -> str  (rewrite text)
    llm_call: Callable[[str, str, int], str]
    # Phase-2a: per-pass fan-out settings (None -> classic regional shim+boundary path)
    fanout: object | None = None


def generate(intent: RemediationIntent, deps: PatchDeps, attempt: int,
             path: str) -> Gen:
    """Run the generator for ``path``, mutating the (clean) working tree."""
    # Phase 2f: the chain path is the coordinated fix FOR cancellation cascades, so it
    # is exempt from the signal_class awaiting_algorithmic_rewrite filter below (which
    # skips single-region precision rungs on cascade regions).  Route it first.
    if path == PATH_CHAIN:
        return _gen_chain(intent, deps, attempt)

    # Phase 2e signal_class filter: a precision-rung intent on a cancellation-cascade
    # / local-cancellation region is structurally inert (a wider type cannot restore
    # catastrophically cancelled digits).  Short-circuit BEFORE any LLM/build to the
    # terminal awaiting_algorithmic_rewrite status — no shim generation, no compile.
    # Reformulate kinds are exempt (they ARE the algorithmic fix), as is any pass that
    # supplies no signal_class map (fail-open).  See fanout.py's module docstring.
    awaiting = _awaiting_rewrite(intent, deps)
    if awaiting is not None:
        return awaiting

    if path == PATH_REGIONAL:
        # Phase 2a: a regional intent in a fan-out-enabled pass is realized as
        # per-caller-path function variants instead of a type-specialization shim
        # (Blocker #1 fix).  Falls back to the classic path when fan-out cannot place
        # the region (not in a resolvable/reachable function — e.g. a chain rep).
        fanout = getattr(deps, "fanout", None)
        if fanout is not None and getattr(fanout, "enabled", False):
            return _gen_regional_fanout(intent, deps, attempt)
        return _gen_regional(intent, deps, attempt)
    if path == PATH_PLAIN_EDIT:
        return _gen_plain_edit(intent, deps)
    if path == PATH_REVERT:
        return _gen_revert(intent, deps)
    if path == PATH_LLM_REWRITE:
        return _gen_rewrite(intent, deps, attempt)
    raise ValueError(f"unknown dispatch path {path!r}")


def _awaiting_rewrite(intent: RemediationIntent, deps: PatchDeps) -> Gen | None:
    """Terminal ``awaiting_algorithmic_rewrite`` Gen, or None if the filter is inert.

    Fires only for a *precision transition* (``TRANSITION_KINDS``; reformulate kinds
    are the fix, not the target) on a region the report classes
    ``cancellation_cascade`` / ``local_cancellation``, when the pass supplied a
    ``signal_class_by_region`` map (via ``FanoutSettings``).  No LLM, no build."""
    from agents.patcher import fanout as fo
    if intent.kind not in TRANSITION_KINDS:
        return None
    fanout = getattr(deps, "fanout", None)
    sc_map = getattr(fanout, "signal_class_by_region", None) if fanout else None
    if not sc_map:
        return None
    signal_class = sc_map.get(intent.target.location)
    if not fo.awaits_algorithmic_rewrite(signal_class):
        return None
    return Gen(False, R.AWAITING_ALGORITHMIC_REWRITE, R.ERR_AWAITING_REWRITE,
               f"awaiting_algorithmic_rewrite: region {intent.target.location} "
               f"signal_class={signal_class}; precision rungs are structurally inert "
               f"(wider intermediates cannot rescue chained/near-equal cancellation); "
               f"awaiting an algorithmic rewrite catalog entry (Strategy models "
               f"reformulate-kahan / reformulate-identity via "
               f"agents/strategy/walk.py::_rewrites_for; not yet wired into the fan-out)")


# ---------------------------------------------------------------------------
# 1. regional-integrator
# ---------------------------------------------------------------------------

def _region_text_from_disk(target_path, line_start: int, line_end: int) -> str | None:
    """Read the inclusive 1-based region ``[line_start, line_end]`` from the working
    tree (reset to the parent SHA at gen time).  Returns ``None`` on any I/O or range
    error — the 2d-B write-truncation gate then simply doesn't fire (honest
    build+measure), never a crash."""
    try:
        lines = Path(target_path).read_text(
            encoding="utf-8", errors="ignore").split("\n")
    except (OSError, TypeError, ValueError):
        return None
    if line_start < 1 or line_end > len(lines) or line_start > line_end:
        return None
    return "\n".join(lines[line_start - 1:line_end])


def _gen_regional(intent: RemediationIntent, deps: PatchDeps, attempt: int) -> Gen:
    to = intent.kind.split("-to-")[-1]
    # Target scalar tag + integrator key.  ``float`` (Wave 2) is a native demotion
    # target routed here only when Strategy tagged the intent ``via="regional"``
    # (a template-typed region — dispatch_path made that call); ff/dd are the
    # extended promotion/demotion targets.
    scalar = {"ff": "ffloat", "dd": "ddouble"}.get(to, "float")
    which = {"ff": "ff", "dd": "dd"}.get(to, "float")
    # Caller precision to demote/widen region writes back to on exit.  Only
    # float-to-ff promotes from float; every other regional transition converts
    # against double (ff-to-dd / ff-to-float revert to double first, below).
    caller_type = "float" if intent.kind == "float-to-ff" else "double"
    integrator = deps.integrators.get(which)
    if integrator is None:
        return Gen(False, R.LLM_GEN_FAILED, R.ERR_INTEGRATOR,
                   f"no {which}_integrator wired")

    # Characterization region keys are bare basenames (``B2m.h``) but the file may
    # live in a subdir (``box/B2m.h``).  The integrator reads the source at a SHA
    # via ``git show`` and labels its boundary patch with this path, both of which
    # need the repo-relative path, not the bare name.  ``deps.target_path`` was
    # resolved by ``resolve_in_tree`` (precheck guarantees the file exists).
    try:
        rel_file = deps.target_path.resolve().relative_to(
            deps.repo_root.resolve()).as_posix()
    except (AttributeError, ValueError):
        rel_file = intent.target.file

    # composite ff-to-dd / ff-to-float (regional): strip the prior ff install
    # first (back to the clean double baseline), then generate the dd / float shim
    # against that clean region.
    if intent.kind in ("ff-to-dd", "ff-to-float"):
        rv = _do_revert(intent, deps, "-to-ff")
        if not rv.ok:
            return rv

    # Phase 2d-B write-boundary-truncation gate (regional / chain-representative path).
    # The region's enclosing function is unknown to the fan-out call graph, so its Case-B
    # write targets are recovered region-locally.  When this UPCAST truncates every landing
    # back to caller precision (no wider persistent sink), the promotion is numerically
    # inert — fail terminally here, upstream of the LLM shim gen + build.  Only ever fires
    # for an upcast with a source-visible caller-precision store (never a native float
    # downcast, never a bare return), so a genuine remediation is not suppressed.
    _, _wt_two_limb, _ = _precision_cpp(which)
    if _wt_two_limb:
        region_text = _region_text_from_disk(
            deps.target_path, intent.target.line_start, intent.target.line_end)
        if region_text is not None:
            from agents.shared.region_scan import region_writes_from_source
            wt_writes = region_writes_from_source(region_text)
            if boundary.write_truncation_inert(
                    region_text, list(intent.target.variables), wt_writes,
                    _wt_two_limb, caller_type=caller_type):
                return Gen(False, R.WRITE_TRUNCATION, R.ERR_TRUNCATION,
                           f"write_truncation: region {intent.target.location} promotes "
                           f"to {scalar} but every landing truncates back to "
                           f"{caller_type} (no wider persistent sink) — numerically "
                           f"inert at the boundary")

    try:
        res: RegionIntegrationResult = integrator(
            file=rel_file,
            line_start=intent.target.line_start,
            line_end=intent.target.line_end,
            variables=list(intent.target.variables),
            working_tree=deps.parent_sha,
            repo_path=str(deps.repo_root),
            scalar_type=scalar,
            caller_type=caller_type,
            direction="in",
            out_dir=deps.shims_dir,
            attempt=attempt,
        )
    except NotImplementedError as exc:
        return Gen(False, R.LLM_GEN_FAILED, R.ERR_INTEGRATOR, str(exc))
    except Exception as exc:  # noqa: BLE001 - integrator crash → retryable gen fail
        return Gen(False, R.LLM_GEN_FAILED, R.ERR_INTEGRATOR, repr(exc))

    if res is None or not getattr(res, "ok", False):
        detail = getattr(res, "error", None) or "integrator returned no result"
        toks = getattr(res, "llm_tokens", 0) if res else 0
        return Gen(False, R.LLM_GEN_FAILED, R.ERR_INTEGRATOR, detail, llm_tokens=toks)

    boundary_patch_path = None
    if res.boundary_patch:
        boundary_patch_path = deps.patches_dir / f"{intent.rationale_id or 'patch'}.patch"
        boundary_patch_path.parent.mkdir(parents=True, exist_ok=True)
        boundary_patch_path.write_text(res.boundary_patch)
        try:
            gitops.apply_patch(deps.repo_root, res.boundary_patch)
        except gitops.GitError as exc:
            # the integrator's own boundary patch didn't apply → misgen, retryable
            return Gen(False, R.LLM_GEN_FAILED, R.ERR_APPLY,
                       f"boundary patch failed to apply: {exc}",
                       llm_tokens=res.llm_tokens)

    return Gen(True, shim_paths=list(res.shim_paths),
               boundary_patch_path=str(boundary_patch_path) if boundary_patch_path else None,
               llm_tokens=res.llm_tokens)


# ---------------------------------------------------------------------------
# 1b. regional-integrator via call-graph fan-out (Phase 2a)
# ---------------------------------------------------------------------------

def _precision_cpp(which: str) -> tuple[str, bool, str]:
    """Concrete C++ scalar spelling + two-limb flag + complex spelling for ``which``.

    Read from the integrator ``SPEC`` objects so type spellings have one source of
    truth (no duplication of ``quad::ffun::ffloat`` etc. in the Patcher).  The complex
    spelling (``quad::ffun::ffcomplex`` / ``quad::ddfun::ddcomplex`` /
    ``Kokkos::complex<float>``) drives the Phase-2d complex-container promotion."""
    from agents.chain_integrator.agent import SPEC as _CH
    from agents.dd_integrator.agent import SPEC as _DD
    from agents.ff_integrator.agent import SPEC as _FF
    from agents.float_integrator.agent import SPEC as _FL
    spec = {"dd": _DD, "ff": _FF, "float": _FL, "chain_dd": _CH}[which]
    return spec.cpp_scalar, spec.two_limb, spec.cpp_complex


def _gen_regional_fanout(intent: RemediationIntent, deps: PatchDeps, attempt: int) -> Gen:
    """Realize a regional intent as per-caller-path variants (design Phase 2a).

    Generates+installs the extended-precision shim via the same integrator as the
    classic path (LLM, retryable), then — instead of applying that integrator's
    include-site boundary patch — splices the promoted region into copied function
    variants and cascades the renames up to the entry point (see
    :mod:`agents.patcher.fanout`).  Falls back to :func:`_gen_regional` when the
    region cannot be placed in the call graph (e.g. a chain representative).
    """
    from agents.patcher import fanout as fo
    from agents.patcher.call_graph import CallGraphError
    from agents.shared.region_scan import extract_region_writes

    to = intent.kind.split("-to-")[-1]
    scalar = {"ff": "ffloat", "dd": "ddouble"}.get(to, "float")
    which = {"ff": "ff", "dd": "dd"}.get(to, "float")
    caller_type = "float" if intent.kind == "float-to-ff" else "double"
    integrator = deps.integrators.get(which)
    if integrator is None:
        return Gen(False, R.LLM_GEN_FAILED, R.ERR_INTEGRATOR, f"no {which}_integrator wired")

    try:
        rel_file = deps.target_path.resolve().relative_to(
            deps.repo_root.resolve()).as_posix()
    except (AttributeError, ValueError):
        rel_file = intent.target.file

    # composite ff-to-dd / ff-to-float: strip the prior ff install first (as classic).
    if intent.kind in ("ff-to-dd", "ff-to-float"):
        rv = _do_revert(intent, deps, "-to-ff")
        if not rv.ok:
            return rv

    # Call graph (built once per pass, cached).  A build failure is terminal — fail
    # loud rather than silently miss edges (new manifest mode call_graph_build_failed).
    try:
        graph = fo.graph_for_pass(deps.fanout, deps.repo_root)
    except CallGraphError as exc:
        return Gen(False, R.PATCH_APPLY_FAILED, R.ERR_APPLY,
                   f"call_graph_build_failed: {exc}")

    # 1. Generate + install the shim (LLM); its boundary patch is intentionally
    #    ignored — the fan-out splices the promotion into variant copies instead.
    try:
        res: RegionIntegrationResult = integrator(
            file=rel_file, line_start=intent.target.line_start,
            line_end=intent.target.line_end, variables=list(intent.target.variables),
            working_tree=deps.parent_sha, repo_path=str(deps.repo_root),
            scalar_type=scalar, caller_type=caller_type, direction="in",
            out_dir=deps.shims_dir, attempt=attempt)
    except NotImplementedError as exc:
        return Gen(False, R.LLM_GEN_FAILED, R.ERR_INTEGRATOR, str(exc))
    except Exception as exc:  # noqa: BLE001
        return Gen(False, R.LLM_GEN_FAILED, R.ERR_INTEGRATOR, repr(exc))
    if res is None or not getattr(res, "ok", False):
        detail = getattr(res, "error", None) or "integrator returned no result"
        toks = getattr(res, "llm_tokens", 0) if res else 0
        return Gen(False, R.LLM_GEN_FAILED, R.ERR_INTEGRATOR, detail, llm_tokens=toks)

    shim_include = f"ql_shim_{which}.h"          # regional.canonical_shim_name
    scalar_cpp, two_limb, complex_cpp = _precision_cpp(which)
    # Phase 2d: app source roots for template-parameter → concrete-type resolution
    # (which reads are complex containers).  Empty → complex promotion disabled
    # (degrades to the pre-2d scalar-only transform).
    app_roots = list(getattr(deps.fanout, "app_source_roots", []) or [])

    # 2. Region writes (Fix C) for the promotion; non-fatal on scan error.
    try:
        writes = extract_region_writes(rel_file, intent.target.line_start,
                                       intent.target.line_end, deps.parent_sha,
                                       tracked_type=caller_type)
    except Exception:  # noqa: BLE001
        writes = []

    # 3. Fan out: variant copies + rename cascade.  A FanoutError (region not in a
    #    resolvable/reachable function — e.g. a chain representative) falls back to
    #    the classic regional boundary path so the intent still gets a fair shot.
    from agents.patcher.variant_naming import VariantNameError
    try:
        fr = fo.fan_out_region(
            file=rel_file, line_start=intent.target.line_start,
            line_end=intent.target.line_end, reads=list(intent.target.variables),
            writes=list(writes), integral=deps.fanout.integral, graph=graph,
            tree_root=str(deps.repo_root), scalar_type=scalar_cpp, two_limb=two_limb,
            shim_include=shim_include, caller_type=caller_type,
            max_paths=deps.fanout.max_paths,
            complex_type=complex_cpp, app_source_roots=app_roots)
    except VariantNameError as exc:
        # A collision is a fan-out bug, not a chain rep — surface it terminally
        # (deterministic; retrying cannot help) as the new manifest failure mode.
        return Gen(False, R.PATCH_APPLY_FAILED, R.ERR_APPLY,
                   f"variant_name_collision: {exc}")
    except fo.FanoutError:
        # Region not in a resolvable/reachable function (e.g. a chain representative)
        # → fall back to the classic regional boundary path.
        return _gen_regional(intent, deps, attempt)

    # Phase 2c promotion_no_op gate: the fan-out placed + wired the variant, but the
    # region promotion retyped nothing (empty payload) — the body is byte-identical to
    # the original at the region, so the candidate would be a bit-for-bit clone of the
    # baseline.  Terminal + deterministic (source-derived reads won't change on retry).
    if not fr.promotion_applied:
        return Gen(False, R.PROMOTION_NO_OP, R.ERR_EMPTY,
                   f"promotion_no_op: region {intent.target.location} promotes nothing "
                   f"(empty payload; reads={fr.reads_used}) — variant body is "
                   f"byte-identical to the original")

    # Phase 2d-B write-boundary-truncation gate: the promotion retyped the body, but this
    # UPCAST truncates every landing back to caller precision (no wider persistent sink),
    # so the candidate is numerically inert (delta == baseline).  Terminal + deterministic
    # (a wider rung would truncate identically) — fail here instead of paying a build.
    if fr.write_truncation:
        return Gen(False, R.WRITE_TRUNCATION, R.ERR_TRUNCATION,
                   f"write_truncation: region {intent.target.location} promotes to "
                   f"{scalar_cpp} but every landing truncates back to {caller_type} "
                   f"(no wider persistent sink) — numerically inert at the boundary")

    return Gen(True, shim_paths=list(res.shim_paths), llm_tokens=res.llm_tokens,
               declared_variants=list(fr.declared_variants),
               files_touched=list(fr.files_touched),
               in_place_region=fr.in_place_region)


# ---------------------------------------------------------------------------
# 1c. chain-scoped dd promotion via the coordinated fan-out (Phase 2f)
# ---------------------------------------------------------------------------

def _gen_chain(intent: RemediationIntent, deps: PatchDeps, attempt: int) -> Gen:
    """Realize a chain-scoped dd promotion as a coordinated multi-region envelope.

    Generates a ddouble shim for EACH chain region via the chain integrator (all
    merge into the one ``ql_shim_dd.h``), then hands the whole ``chain_lines`` set to
    :func:`agents.patcher.chain_promote.chain_promote`, which splices per-integral
    variants + reroutes the chain-internal calls bottom-up.  The chain-scope 2c/2d
    gates (empty-whole-chain / outermost-truncation) map to the SAME terminal statuses
    the single-region fan-out uses.  Requires fan-out settings (entry_point/integral).
    """
    from agents.patcher import chain_promote as cp
    from agents.patcher import fanout as fo
    from agents.patcher.call_graph import CallGraphError
    from agents.patcher.variant_naming import VariantNameError

    fanout = getattr(deps, "fanout", None)
    if fanout is None:
        return Gen(False, R.LLM_GEN_FAILED, R.ERR_INTEGRATOR,
                   "chain path requires fan-out settings (entry_point / integral)")
    # Prefer the chain integrator; fall back to the dd integrator (same ddouble target).
    integrator = deps.integrators.get("chain_dd") or deps.integrators.get("dd")
    if integrator is None:
        return Gen(False, R.LLM_GEN_FAILED, R.ERR_INTEGRATOR,
                   "no chain_dd/dd integrator wired")

    chain_lines = [tuple(t) for t in (getattr(intent, "chain_lines", None) or [])]
    if not chain_lines:
        return Gen(False, R.PATCH_APPLY_FAILED, R.ERR_APPLY,
                   f"chain intent {intent.rationale_id} carries no chain_lines")

    try:
        graph = fo.graph_for_pass(fanout, deps.repo_root)
    except CallGraphError as exc:
        return Gen(False, R.PATCH_APPLY_FAILED, R.ERR_APPLY,
                   f"call_graph_build_failed: {exc}")

    scalar_cpp, two_limb, complex_cpp = _precision_cpp("chain_dd")
    shim_include = "ql_shim_dd.h"                 # regional.canonical_shim_name(dd)
    app_roots = list(getattr(fanout, "app_source_roots", []) or [])

    # 1. Generate a ddouble shim for each chain region (LLM, retryable); the boundary
    #    patches are ignored — chain_promote splices the promotion into variant copies.
    total_tokens = 0
    all_shims: list[str] = []
    resolved_lines: list[tuple[str, int, int]] = []
    for (cfile, cls, cle) in chain_lines:
        fd = graph.enclosing_function(cfile, cls)
        if fd is None:
            return Gen(False, R.PATCH_APPLY_FAILED, R.ERR_APPLY,
                       f"chain region {cfile}:{cls}-{cle} not inside any known function")
        try:
            rel_file = Path(fd.file).resolve().relative_to(
                deps.repo_root.resolve()).as_posix()
        except (AttributeError, ValueError):
            rel_file = cfile
        resolved_lines.append((rel_file, cls, cle))
        try:
            res: RegionIntegrationResult = integrator(
                file=rel_file, line_start=cls, line_end=cle, variables=[],
                working_tree=deps.parent_sha, repo_path=str(deps.repo_root),
                scalar_type="ddouble", caller_type="double", direction="in",
                out_dir=deps.shims_dir, attempt=attempt)
        except NotImplementedError as exc:
            return Gen(False, R.LLM_GEN_FAILED, R.ERR_INTEGRATOR, str(exc))
        except Exception as exc:  # noqa: BLE001 - integrator crash → retryable gen fail
            return Gen(False, R.LLM_GEN_FAILED, R.ERR_INTEGRATOR, repr(exc))
        if res is None or not getattr(res, "ok", False):
            detail = getattr(res, "error", None) or "integrator returned no result"
            toks = getattr(res, "llm_tokens", 0) if res else 0
            return Gen(False, R.LLM_GEN_FAILED, R.ERR_INTEGRATOR,
                       f"chain region {cfile}:{cls}: {detail}", llm_tokens=total_tokens + toks)
        total_tokens += res.llm_tokens
        all_shims.extend(res.shim_paths)

    # 2. Coordinated chain promotion (deterministic variant splice + reroute cascade).
    manifest = cp.ChainManifest(
        chain_id=intent.rationale_id or "chain", integral=fanout.integral,
        entry_point=fanout.entry_point, lines=resolved_lines)
    try:
        cr = cp.chain_promote(
            manifest=manifest, graph=graph, tree_root=str(deps.repo_root),
            scalar_type=scalar_cpp, two_limb=two_limb, shim_include=shim_include,
            caller_type="double", complex_type=complex_cpp,
            max_paths=fanout.max_paths, app_source_roots=app_roots)
    except VariantNameError as exc:
        return Gen(False, R.PATCH_APPLY_FAILED, R.ERR_APPLY,
                   f"variant_name_collision: {exc}")
    except fo.FanoutError as exc:
        return Gen(False, R.PATCH_APPLY_FAILED, R.ERR_APPLY, f"chain_fanout_failed: {exc}")

    # 3. Chain-scope 2c / 2d-B gates (envelope, not individual links).
    if not cr.promotion_applied:
        return Gen(False, R.PROMOTION_NO_OP, R.ERR_EMPTY,
                   f"chain promotion_no_op: chain {manifest.chain_id} promotes nothing "
                   f"across all {len(chain_lines)} regions (empty whole-chain payload)")
    if cr.write_truncation:
        return Gen(False, R.WRITE_TRUNCATION, R.ERR_TRUNCATION,
                   f"chain write_truncation: an INTERIOR region of chain "
                   f"{manifest.chain_id} truncates every landing back to double (no wider "
                   f"persistent sink) — an intra-chain write injects double roundoff "
                   f"between links, breaking the chain (the outermost region's "
                   f"exit-truncation is the designed boundary and is exempt)")

    return Gen(True, shim_paths=all_shims, llm_tokens=total_tokens,
               declared_variants=list(cr.declared_variants),
               files_touched=list(cr.files_touched),
               in_place_region=(cr.in_place_regions > 0 and not cr.declared_variants))


# ---------------------------------------------------------------------------
# 2. plain-type-edit
# ---------------------------------------------------------------------------

def _gen_plain_edit(intent: RemediationIntent, deps: PatchDeps) -> Gen:
    src, dst = intent.kind.split("-to-")
    try:
        edits.rewrite_types(deps.target_path, intent.target.line_start,
                            intent.target.line_end, src, dst)
    except edits.EditError as exc:
        # No bare `src` keyword token on the region line → the region is
        # template-typed, so this plain-edit rung is *inapplicable*, not a
        # malformed intent (benign; Strategy advances the walk).
        return Gen(False, R.PATCH_INAPPLICABLE, R.ERR_APPLY, str(exc))
    return Gen(True)


# ---------------------------------------------------------------------------
# 3. git-revert
# ---------------------------------------------------------------------------

def _gen_revert(intent: RemediationIntent, deps: PatchDeps) -> Gen:
    suffix = "-to-dd" if intent.kind == "dd-to-double" else "-to-ff"
    rv = _do_revert(intent, deps, suffix)
    if not rv.ok:
        return rv
    if intent.kind == "ff-to-float":
        # composite: after stripping ff (back to double), demote double → float.
        # A template-typed region has no bare `double` token to rewrite → the
        # float rung is inapplicable (benign), not a strategy_bug.
        try:
            edits.rewrite_types(deps.target_path, intent.target.line_start,
                                intent.target.line_end, "double", "float")
        except edits.EditError as exc:
            return Gen(False, R.PATCH_INAPPLICABLE, R.ERR_APPLY, str(exc))
    return Gen(True)


def _do_revert(intent: RemediationIntent, deps: PatchDeps, suffix: str) -> Gen:
    sha = gitops.find_introducing_commit(
        deps.repo_root, "HEAD", intent.target.file,
        intent.target.line_start, suffix)
    if sha is None:
        return Gen(False, R.PATCH_APPLY_FAILED, R.ERR_APPLY,
                   f"no introducing '{suffix}' commit for {intent.target.location}")
    try:
        gitops.revert_no_commit(deps.repo_root, sha)
    except gitops.GitError as exc:
        return Gen(False, R.PATCH_APPLY_FAILED, R.ERR_APPLY, str(exc))
    return Gen(True)


# ---------------------------------------------------------------------------
# 4. llm-rewrite
# ---------------------------------------------------------------------------

def _gen_rewrite(intent: RemediationIntent, deps: PatchDeps, attempt: int) -> Gen:
    if deps.llm_call is None:
        return Gen(False, R.LLM_GEN_FAILED, R.ERR_LLM, "no llm_call wired")
    src = rewrites.region_source(deps.target_path, intent.target.line_start,
                                 intent.target.line_end)
    system, user = rewrites.build_prompt(intent, src)
    try:
        new_src = deps.llm_call(system, user, attempt)
    except Exception as exc:  # noqa: BLE001 - LLM error → retryable gen fail
        return Gen(False, R.LLM_GEN_FAILED, R.ERR_LLM, repr(exc))
    if not new_src or not new_src.strip():
        return Gen(False, R.LLM_GEN_FAILED, R.ERR_LLM, "LLM returned empty rewrite")
    try:
        rewrites.apply_rewrite(deps.target_path, intent.target.line_start,
                               intent.target.line_end, new_src)
    except ValueError as exc:
        return Gen(False, R.PATCH_APPLY_FAILED, R.ERR_APPLY, str(exc))
    return Gen(True)
