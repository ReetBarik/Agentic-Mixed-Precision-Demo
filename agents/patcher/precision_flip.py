"""Phase-1 template-argument promotion — detection + routing (deliverable 1).

The Phase-1 *correctness* mechanism is a **per-integral whole-TU precision flip**:
a flagged integral is compiled in its own translation unit at its own precision (the
generalization of the ``QL_MODE=vanilla|dd`` split the Validator already drives), and
its dd output is narrowed to caller precision only at the app-output boundary reusing
the acc1482 designed-exit transforms.  This is a *build-orchestration* mechanism, not a
source rewrite — it emits no dd-typed source into a caller-precision TU, so the whole
``instantiation_gate`` Shape-1/2/3/4 boundary taxonomy has no surface to appear on
(TEMPLATE_ARG_PROMOTION_DESIGN.md §2.2b, §2.3).

This module owns the **decision** only — "does integral X route to the precision-flip
path, and to what target precision?".  It mutates no tree and builds nothing.  The
decision is:

* **shape-based**, never identifier-based — an integral is a candidate iff (i) it was
  flagged for a wider precision by characterization (the existing ``signal_class`` /
  chain signal — never a baked-in integral name), AND (ii) its enclosing subtree is
  *fully template-parametric* (every definition reachable from the entry point down to
  the region's frames is a template, so the compiler can re-instantiate the whole
  subtree at a different scalar type).  No ``Kokkos`` / ``ql`` / ``BO`` token appears in
  any predicate here (feedback_no_placeholder_patterns).
* **precision-parameterized**, never dd-hardcoded — the target precision is carried as a
  value (:class:`TargetPrecision`), so Phase 2/3 can route the same integral to ``ff`` /
  ``float`` without touching this module (hard-coding to dd would be STOP #SS).
* **deterministic** — a pure function of the call graph + the dd-flag signal; no LLM.

Non-candidates route to :data:`Route.RAW_DOUBLE` (Decision 4: a rejected integral falls
back to the raw caller-precision baseline, not element promotion).  The
element-promotion / rule-(d) machinery is **retained but dormant** on the Phase-1
correctness path (Decision 5) — :func:`route_integral` never selects it; it is the
reserved Phase-3 demotion path and the fallback for genuinely non-parametric contexts,
selected by the existing chain/fanout dispatch, not here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from agents.patcher.call_graph import CallGraph, FuncDef


class TargetPrecision(str, Enum):
    """A precision the flip can target.  Phase-1 uses ``DD``; the enum exists so the
    routing/emission stack is precision-parameterized from the start (Decision 2 —
    hard-coding to dd is STOP #SS).  ``FF`` / ``FLOAT`` are wired for Phase 2/3.

    ``QF`` (quad-float, 4xFP32, ~28.9 digits) sits between ``DOUBLE`` and ``DD`` on the
    ladder: it is the cheap alternative to dd on fp32-heavy silicon, where four FP32
    words cost far less than a double-double.  Unlike dd it does NOT widen the exponent
    range — qf stays FP32-bounded at ~3.4e38, which the strategy walk's fp32-family
    range guard has to account for on the correctness path, not just for speedups."""

    DD = "dd"
    QF = "qf"
    FF = "ff"
    FLOAT = "float"


class Route(str, Enum):
    """Where :func:`route_integral` sends an integral on the Phase-1 correctness path.

    * :data:`PRECISION_FLIP` — a candidate: build its TU at the target precision
      (deliverable 2) and dispatch its symbol (deliverable 3).
    * :data:`RAW_DOUBLE` — a non-candidate: the raw caller-precision baseline, no
      promotion machinery invoked (Decision 4).  This is *also* where a rejected
      candidate lands after the acceptance gate (deliverable 5) — the gate flips the
      route, this function only makes the *initial* structural routing.
    """

    PRECISION_FLIP = "precision_flip"
    RAW_DOUBLE = "raw_double"


@dataclass(frozen=True)
class ParametricityResult:
    """Outcome of the deliverable-(a) feasibility check for one integral's subtree.

    ``parametric`` is the go/no-go; ``non_template`` names every reachable definition
    that is NOT a template (empty iff ``parametric``) so the report can show *why* a
    subtree was rejected rather than a bare ``False``.  ``unresolved`` names callees
    that appear as edges but have no definition in the graph (a leaf outside the parsed
    unit — e.g. a ``std::`` call); these do not by themselves defeat parametricity (a
    scalar-library leaf is resolved by overload, not by the subtree's type args), but
    are surfaced for the report.
    """

    parametric: bool
    frames_checked: tuple[str, ...] = ()
    non_template: tuple[str, ...] = ()
    unresolved: tuple[str, ...] = ()


@dataclass(frozen=True)
class FlipDecision:
    """The routing decision for one integral (deliverable 1's output)."""

    integral: str
    route: Route
    target: TargetPrecision | None
    reason: str
    parametricity: ParametricityResult | None = None


# --------------------------------------------------------------------------- #
# deliverable (a) — subtree template-parametricity
# --------------------------------------------------------------------------- #

def _reachable_frames(graph: CallGraph, targets: list[str],
                      *, max_depth: int = 32) -> tuple[list[str], list[str]]:
    """Names reachable from the entry point that lie on a path to any ``targets`` frame.

    Walks the forward call graph from ``graph.root`` and keeps every node from which a
    target is reachable (the union of the root→target subtrees).  Returns
    ``(frames, unresolved)`` where ``frames`` is the sorted set of *defined* nodes on
    those paths (including the root and the targets) and ``unresolved`` is the sorted
    set of callee names that have no definition (leaves outside the parsed unit).
    """
    want = set(targets)
    # Downward reachability of a target from each node (memoized DFS over edges).
    leads_to_target: dict[str, bool] = {}
    unresolved: set[str] = set()

    def dfs(node: str, trail: tuple[str, ...]) -> bool:
        if node in leads_to_target:
            return leads_to_target[node]
        if len(trail) > max_depth:
            return False
        hit = node in want
        for callee in graph.edges.get(node, ()):        # forward edges: caller→callees
            if callee not in graph.defs:
                unresolved.add(callee)
                continue
            if callee in trail:                          # simple-path guard (cycle-safe)
                continue
            if dfs(callee, trail + (callee,)):
                hit = True
        leads_to_target[node] = hit
        return hit

    root = graph.root
    dfs(root, (root,))
    frames = sorted(n for n, hit in leads_to_target.items() if hit)
    # A target with no incoming reachability (unreachable from root) is still reported
    # as a frame so the caller sees it in ``frames_checked`` — but it will be flagged.
    for t in want:
        if t in graph.defs and t not in frames:
            frames.append(t)
    return sorted(set(frames)), sorted(unresolved)


def subtree_is_parametric(graph: CallGraph, targets: list[str],
                          *, max_depth: int = 32) -> ParametricityResult:
    """Deliverable (a): is every frame from the entry point to ``targets`` a template?

    ``targets`` are the enclosing-function names of the integral's flagged regions
    (resolve them from the region ``(file, line)`` via
    :meth:`CallGraph.enclosing_function` before calling).  The subtree is parametric iff
    every *defined* frame on a root→target path is a template definition
    (:attr:`FuncDef.is_template`) — then the compiler generates a fresh instantiation of
    the whole subtree at a different scalar type, which is exactly the precision flip.

    A frame with several definitions (overloads) counts as a template iff its
    preprocessor-active definition(s) are templates — a non-template active overload
    breaks parametricity (the flip could bind it at the wrong precision).  Unresolved
    callees (no definition) are scalar-library leaves resolved by overload, not by the
    subtree's type args, so they do not defeat parametricity; they are reported.
    """
    if not targets:
        return ParametricityResult(parametric=False, non_template=("<no target frame>",))
    frames, unresolved = _reachable_frames(graph, targets, max_depth=max_depth)
    non_template: list[str] = []
    for name in frames:
        active = graph.active_defs(name) or graph.defs.get(name, [])
        # A frame is parametric iff it has ≥1 active def and every active def is a
        # template.  No active def at all (name known only as an edge target) is treated
        # as unresolved-leaf, not a break.
        tdefs = [fd for fd in active if isinstance(fd, FuncDef)]
        if tdefs and not all(fd.is_template for fd in tdefs):
            non_template.append(name)
    # A requested target that is unreachable/undeclared defeats the check (we cannot
    # prove its subtree parametric if we cannot even place it).
    missing = [t for t in targets if t not in graph.defs]
    parametric = not non_template and not missing
    return ParametricityResult(
        parametric=parametric,
        frames_checked=tuple(frames),
        non_template=tuple(sorted(set(non_template) | set(missing))),
        unresolved=tuple(unresolved))


# --------------------------------------------------------------------------- #
# deliverable 1 — routing decision
# --------------------------------------------------------------------------- #

def route_integral(integral: str, *, dd_flagged: bool, graph: CallGraph,
                   target_frames: list[str],
                   target: TargetPrecision = TargetPrecision.DD,
                   max_depth: int = 32) -> FlipDecision:
    """Route one integral to the precision-flip path or the raw-double baseline.

    An integral is a Phase-1 candidate iff BOTH:

      (i) ``dd_flagged`` — characterization flagged it for a wider precision (the
          existing dd-need signal: a COMPUTED cascade chain / a dd-transition intent).
          This is passed in by the harness, never derived from the integral's *name*.
      (ii) its enclosing subtree is fully template-parametric
          (:func:`subtree_is_parametric` over ``target_frames``).

    Candidates route to :data:`Route.PRECISION_FLIP` at ``target`` (dd for Phase-1;
    the parameter keeps the stack precision-parameterized — STOP #SS).  Everything else
    routes to :data:`Route.RAW_DOUBLE` (Decision 4): a dd-un-flagged integral needs no
    promotion, and a flagged-but-non-parametric one is out of the template-arg path's
    scope (its fallback is the dormant element/rule-d machinery, selected by the
    existing chain dispatch — not here).  The acceptance gate (deliverable 5) may later
    demote a *built-but-no-lift* candidate back to raw double; that is a separate step.
    """
    if not dd_flagged:
        return FlipDecision(
            integral=integral, route=Route.RAW_DOUBLE, target=None,
            reason="not dd-flagged (characterization sees no wider-precision need)")
    para = subtree_is_parametric(graph, target_frames, max_depth=max_depth)
    if not para.parametric:
        detail = ", ".join(para.non_template) or "no target frame resolved"
        return FlipDecision(
            integral=integral, route=Route.RAW_DOUBLE, target=None,
            reason=f"subtree not fully template-parametric (non-template: {detail}) "
                   f"— out of the template-arg path; element/rule-d fallback applies",
            parametricity=para)
    return FlipDecision(
        integral=integral, route=Route.PRECISION_FLIP, target=target,
        reason=f"dd-flagged + subtree parametric ({len(para.frames_checked)} frames) "
               f"→ per-integral TU precision flip at {target.value}",
        parametricity=para)


# --------------------------------------------------------------------------- #
# deliverable 5 (Phase-2) — downshift routing
# --------------------------------------------------------------------------- #

# The Phase-2 downshift preference order, cheapest (narrowest) precision first.  The router
# picks the first *available* target in this order; ``FF`` sits here so the stack stays
# precision-parameterized (STOP #SS), but it is filtered out at runtime whenever it is not
# an available target (STOP #EEE: no library-native ff container/leaves → no shim path).
DOWNSHIFT_PREFERENCE: tuple[TargetPrecision, ...] = (
    TargetPrecision.FLOAT, TargetPrecision.FF)


def route_downshift(integral: str, *, dd_candidate: bool, graph: CallGraph,
                    target_frames: list[str],
                    available_targets: "frozenset[TargetPrecision] | set[TargetPrecision]",
                    preference: tuple[TargetPrecision, ...] = DOWNSHIFT_PREFERENCE,
                    max_depth: int = 32) -> FlipDecision:
    """Route one raw-double integral to a *narrower* precision (Phase-2), or keep it double.

    A downshift candidate is a raw-double integral (``dd_candidate=False`` — a Phase-1 dd
    accept is **never** downshifted, STOP #ZZ) whose enclosing subtree is fully
    template-parametric.  The router walks ``preference`` (cheapest precision first) and
    selects the **first target that is both available and parametric**:

    * ``available_targets`` is the set of precisions the emission stack can serve without
      enrichment — passed in by the caller (from the profile table) so this module never
      imports it (and never hard-codes which precisions are live).  With float-only Phase-2
      this is ``{FLOAT}``; ``FF`` is filtered out (STOP #EEE).
    * parametricity is the same structural check as the upshift path
      (:func:`subtree_is_parametric`) — a non-parametric subtree is out of the template-arg
      downshift's scope and stays double.

    A dd candidate, a non-parametric subtree, or no available/parametric target all route to
    :data:`Route.RAW_DOUBLE`.  The acceptance gate (deliverable 6, ``lift_direction=downshift``)
    may still demote a *built-but-precision-losing* candidate back to raw double afterward;
    this function makes only the initial structural routing.
    """
    if dd_candidate:
        return FlipDecision(
            integral=integral, route=Route.RAW_DOUBLE, target=None,
            reason="dd candidate (Phase-1 accept) — never downshifted (STOP #ZZ)")
    para = subtree_is_parametric(graph, target_frames, max_depth=max_depth)
    if not para.parametric:
        detail = ", ".join(para.non_template) or "no target frame resolved"
        return FlipDecision(
            integral=integral, route=Route.RAW_DOUBLE, target=None,
            reason=f"subtree not fully template-parametric (non-template: {detail}) "
                   f"— out of the downshift path; stays raw double",
            parametricity=para)
    live = [t for t in preference if t in available_targets]
    if not live:
        offered = ", ".join(t.value for t in preference) or "<none>"
        return FlipDecision(
            integral=integral, route=Route.RAW_DOUBLE, target=None,
            reason=f"no available downshift target among [{offered}] "
                   f"(none servable without enrichment) — stays raw double",
            parametricity=para)
    target = live[0]
    return FlipDecision(
        integral=integral, route=Route.PRECISION_FLIP, target=target,
        reason=f"raw-double + subtree parametric ({len(para.frames_checked)} frames) "
               f"→ downshift to {target.value} (first available of "
               f"[{', '.join(t.value for t in preference)}])",
        parametricity=para)
