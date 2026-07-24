"""SOLVER_STAGE markdown snapshot generator (Phase 2e).

Renders a ``SolveResult`` + ``QueueBuild`` into the reviewer-facing report the
task specifies: queue rank table, per-candidate outcomes, final precision
distribution + whole-app p100, rejected-float landings, and the greedy
left-on-the-table observations, plus the "Reet review" handoff.

Gate: **regression-relative, 0.5-digit margin vs the double baseline** (Reet's
Stage-2 call — replaces the Stage-1 absolute ``p100 >= 6``, which was
structurally unsatisfiable when the target integral is the whole-app global-min
hotspot; see SOLVER_STAGE1_B12.md).
"""

from __future__ import annotations

from pathlib import Path

from agents.solver.queue import QueueBuild
from agents.solver.solver import (ACCEPTED, APPLY_FAILED, REJECTED,
                                  SKIPPED_RESOLVED, SolveResult)


def _fmt(x, nd=4):
    return "—" if not isinstance(x, (int, float)) else f"{x:.{nd}f}"


def _rank_reason(rung: str) -> str:
    return {"float": "cheapest rung (try first)",
            "ff": "float-float; tried when float rejected",
            "dd": "double-double; most conservative, tried last"}.get(rung, rung)


def build_markdown(res: SolveResult, qb: QueueBuild, *, integral: str,
                   tree_path: str, diff_path: str, manifest_path: str,
                   report_regions: dict, margin: float, solve_wall_sec: float,
                   snapshot: dict, per_integral_floor: dict | None = None,
                   baseline_hotspot: dict | None = None) -> str:
    L: list[str] = []
    ap = L.append

    ap(f"# Solver Stage — {integral} (greedy, regression-relative gate, "
       f"{margin:g}-digit margin)")
    ap("")
    ap(f"Phase 2e: a pipeline stage that **writes an optimized source tree**.  "
       f"Greedy sequential-layering over the fan-out's measured DISCRIM "
       f"`(region, rung)` cells, ranked float<ff<dd, each applied on top of the "
       f"accumulated tree, kept iff the whole-app p100 does not regress more than "
       f"the margin below the double baseline.")
    ap("")
    ap(f"* **Integral:** {integral}")
    ap(f"* **Manifest:** `{manifest_path}`")
    ap(f"* **Gate:** regression-relative — accept iff candidate "
       f"`min_precise_digits >= baseline_min_precise_digits - {margin:g}` "
       f"(the same 0.5-digit regression guard `validate()` bundles; FP128 whole-app "
       f"oracle).  Replaces the Stage-1 absolute `p100 >= 6` (Reet 2026-07-24).")
    ap(f"* **Snapshot:** seed={snapshot.get('seed')}, "
       f"sample_count={snapshot.get('sample_count')}")
    ap(f"* **Merged tree:** `{tree_path}` (HEAD `{(res.final_head or '')[:12]}`)")
    ap(f"* **Cumulative diff:** `{diff_path}`")
    ap(f"* **Solve wall:** {solve_wall_sec}s")
    ap("")

    # -- headline --
    ap("## Headline")
    ap("")
    if res.stopped:
        ap(f"**STOPPED — `{res.stopped}`.** {res.stop_detail}")
    else:
        thr = res.accept_threshold
        ap(f"Queue exhausted normally. Baseline whole-app p100 = "
           f"**{_fmt(res.baseline_min)}**; accept threshold = "
           f"**{_fmt(thr)}** (baseline − {margin:g}); final = "
           f"**{_fmt(res.final_min)}** digits.  {len(res.accepted)} accepted, "
           f"{len(res.rejected)} rejected, "
           f"{len([o for o in res.outcomes if o.outcome == APPLY_FAILED])} "
           f"apply-failed, "
           f"{len([o for o in res.outcomes if o.outcome == SKIPPED_RESOLVED])} "
           f"skipped (region already resolved).")
    ap("")

    # -- blocking finding (structural stop) --
    if res.stopped == "stopped_gate_unimplementable":
        _blocking_finding(ap, res, qb, integral, margin, baseline_hotspot)

    # -- queue rank table --
    ap("## Candidate queue (rank order)")
    ap("")
    ap(f"{len(qb.queue)} DISCRIM competitors; {len(qb.inert)} measured-INERT "
       f"excluded (byte-identical whole-app output → no speedup); "
       f"{len(qb.non_measured)} never reached `measured` (2c/2d terminal gates + "
       f"signal_class `awaiting_algorithmic_rewrite`).")
    ap("")
    ap("| # | region | rung | why ranked here | Δ (region) | baseline Δ |")
    ap("|---|--------|------|-----------------|-----------|-----------|")
    for i, c in enumerate(qb.queue):
        ap(f"| {i} | `{c.region_id}` | {c.rung} | {_rank_reason(c.rung)} | "
           f"{c.delta_effective:.3e} | {c.baseline_delta_effective:.3e} |")
    ap("")
    ap("> Intra-rung tiebreak = region_id ascending (deterministic; the "
       "measurement layer gives no principled cross-region order within a rung — "
       "flop-weighting is a v2 refinement, see handoff).")
    ap("")

    # -- per-candidate outcomes --
    ap("## Per-candidate outcomes")
    ap("")
    ap("| region | rung | outcome | p100 before | p100 after | validator | wall | reason |")
    ap("|--------|------|---------|------------|-----------|-----------|------|--------|")
    for o in res.outcomes:
        ap(f"| `{o.candidate.region_id}` | {o.candidate.rung} | **{o.outcome}** | "
           f"{_fmt(o.min_before)} | {_fmt(o.min_after)} | "
           f"{o.validator_verdict or '—'} | {o.wall_sec}s | {o.reason} |")
    ap("")

    # -- final precision distribution --
    ap(f"## Final precision distribution ({integral} regions)")
    ap("")
    dist = res.precision_distribution()
    ap("| precision | region count |")
    ap("|-----------|-------------|")
    for rung in ("float", "ff", "dd", "double"):
        ap(f"| {rung} | {dist.get(rung, 0)} |")
    ap("")
    ap(f"**Final whole-app min_precise_digits (p100): {_fmt(res.final_min)}** "
       f"(baseline {_fmt(res.baseline_min)}, threshold "
       f"{_fmt(res.accept_threshold)}).")
    ap("")
    landed = {rid: r for rid, r in res.region_final.items() if r != "double"}
    if landed:
        ap("Regions that moved off double:")
        for rid, rung in sorted(landed.items()):
            ap(f"* `{rid}` → **{rung}**")
        ap("")

    # -- rejected-float landings --
    ap("## Regions where float was proposed but rejected")
    ap("")
    rejected_float = [o for o in res.rejected if o.candidate.rung == "float"]
    if not rejected_float:
        ap("None — every float candidate that reached the gate held it "
           "(or no float candidate was rejected).")
    else:
        ap("| region | float p100 | landed at | why |")
        ap("|--------|-----------|-----------|-----|")
        for o in rejected_float:
            rid = o.candidate.region_id
            landed_rung = res.region_final.get(rid, "double")
            why = ("no cheaper-or-equal rung held; stayed double"
                   if landed_rung == "double"
                   else f"backed off to {landed_rung}")
            ap(f"| `{rid}` | {_fmt(o.min_after)} | {landed_rung} | "
               f"float regressed p100 below the threshold "
               f"({_fmt(res.accept_threshold)}); {why} |")
    ap("")

    # -- apply failures --
    apply_failed = [o for o in res.outcomes if o.outcome == APPLY_FAILED]
    if apply_failed:
        ap("## Apply failures on the accumulated tree")
        ap("")
        ap("These cells were `measured` in the flat fan-out but the Patcher "
           "failed to regenerate them on top of the accumulated tree — a "
           "layering-interaction data-quality signal (the solver is the first "
           "consumer to apply patches sequentially):")
        ap("")
        ap("| region | rung | patcher status | detail |")
        ap("|--------|------|----------------|--------|")
        for o in apply_failed:
            ap(f"| `{o.candidate.region_id}` | {o.candidate.rung} | "
               f"{o.patcher_status} | {o.reason} |")
        ap("")

    # -- inert excluded (recorded, not applied) --
    if qb.inert:
        ap("## Measured-INERT cells excluded from the queue")
        ap("")
        ap("Byte-identical whole-app output (`delta_effective == "
           "baseline_delta_effective`): the promotion was a numerical no-op, so "
           "it carries no speedup and is left at double.  These are the residue "
           "the 2c/2d `promotion_no_op` / `write_truncation` gates could not prove "
           "statically pre-build.")
        ap("")
        ap("| region | rung | Δ (== baseline) |")
        ap("|--------|------|----------------|")
        for c in qb.inert:
            ap(f"| `{c.region_id}` | {c.rung} | {c.delta_effective:.3e} |")
        ap("")

    # -- greedy left-on-the-table --
    ap("## What the greedy assumption may leave on the table")
    ap("")
    _left_on_table(ap, res, qb)
    ap("")

    # -- handoff --
    _handoff(ap, res, qb, integral, margin, solve_wall_sec)

    return "\n".join(L) + "\n"


def _blocking_finding(ap, res, qb, integral, margin, baseline_hotspot) -> None:
    ap("## ⛔ Blocking finding — the baseline itself is unscoreable")
    ap("")
    ap(f"The solver stopped on the **baseline**: the unpatched whole-app "
       f"min_precise_digits could not be scored ({res.baseline_min!r}). Under the "
       f"regression-relative gate the baseline is the reference every candidate is "
       f"compared against — with no reference there is nothing to gate, so the "
       f"solver stops and flags rather than guessing (STOP-and-flag discipline, "
       f"preserved from Stage 1).")
    ap("")
    ap("This is *not* the Stage-1 low-baseline case (a well-defined but sub-6 "
       "baseline is fine now — that is the whole point of the regression-relative "
       "gate). A truly unscoreable baseline means a crash / NaN / empty min on the "
       "vanilla tree: investigate the build or the validator battery for this "
       "integral before proceeding.")
    ap("")
    if baseline_hotspot:
        ap(f"**Baseline hotspot (last scored, may be partial):** integral "
           f"`{baseline_hotspot.get('integral')}`, sample "
           f"{baseline_hotspot.get('sample_idx')}, component "
           f"`{baseline_hotspot.get('component')}` — precise digits "
           f"{_fmt(baseline_hotspot.get('precise_digits'))}.")
        ap("")


def _left_on_table(ap, res: SolveResult, qb: QueueBuild) -> None:
    if res.stopped == "stopped_gate_unimplementable":
        ap("* N/A — the walk never got past the baseline (unscoreable). See the "
           "blocking finding.")
        return
    accepted_regions = {o.candidate.region_id for o in res.accepted}
    rejected = [o for o in res.rejected]
    solo_rejected_regions = {o.candidate.region_id for o in rejected
                             if o.candidate.region_id not in accepted_regions}
    if not rejected:
        ap("* No candidate was rejected — the greedy walk accepted every DISCRIM "
           "competitor it could apply, so there is no solo/joint gap to record "
           "for this integral.")
    else:
        ap(f"* **{len(solo_rejected_regions)} region(s)** had a demotion rejected "
           f"solo and never landed lower: "
           f"{', '.join('`'+r+'`' for r in sorted(solo_rejected_regions)) or '—'}. "
           f"v1 does no joint re-measurement, so a pair whose *combined* demotion "
           f"would have held (but neither held alone) is invisible here — record "
           f"for a v2 joint-measurement pass.")
    ap("* Greedy first-accept-per-region takes the **cheapest** rung that holds, "
       "never re-examining whether a more conservative rung would have freed a "
       "sibling region to demote further.")
    if qb.inert:
        ap(f"* **{len(qb.inert)} measured-INERT cells** were excluded as no-ops. "
           f"The 2c/2d investigation established these are structural no-ops, not "
           f"hidden wins.")


def _handoff(ap, res: SolveResult, qb: QueueBuild, integral: str, margin: float,
             solve_wall_sec: float) -> None:
    ap("## Reet review")
    ap("")
    ap("### Solver-design judgment calls")
    ap("")
    ap(f"1. **Regression-relative gate ({margin:g}-digit margin vs baseline).** "
       f"Accept iff `cand_min >= baseline_min - {margin:g}`. The baseline is "
       f"measured once on the unpatched tree at solve start and is the reference "
       f"for the whole run. This is the same 0.5-digit figure `validate()` uses as "
       f"its regression guard (`DEFAULT_MAX_REGRESSION`), reused not re-invented. "
       f"It replaces the Stage-1 absolute `p100 >= 6`, which was unsatisfiable when "
       f"the target integral is the whole-app global-min hotspot (B12 = 3.69).")
    ap("2. **Queue = measured DISCRIM only; measured-INERT excluded.** An INERT "
       "cell would trivially 'accept' (byte-identical → no regression) and lock the "
       "region at a no-op rung ahead of a genuinely cheaper DISCRIM rung.")
    ap("3. **Intra-rung tiebreak = region_id ascending.** Deterministic but "
       "arbitrary; flop-weighting (WI3 table exists) is the obvious v2 upgrade.")
    ap("4. **`current_precision='double'` for every intent.** Sound only under "
       "first-accept-per-region (a region is patched at most once, from double).")
    ap("")
    ap("### Cost")
    ap("")
    n_cand = len(qb.queue)
    built = [o for o in res.outcomes if o.wall_sec and o.wall_sec > 0]
    n_built = len(built)
    per_cand = (sum(o.wall_sec for o in built) / n_built) if n_built else 0.0
    ap(f"* {integral} solve: **{solve_wall_sec}s** wall; {n_built} of {n_cand} "
       f"queued candidates actually built+validated "
       f"(~{per_cand:.0f}s/candidate incl. Patcher fan-out build + whole-app "
       f"validate; the rest were skipped/short-circuited"
       + (" — the run STOPPED on the baseline" if res.stopped else "") + ").")
    ap("")
    ap("### Measurement-layer notes")
    ap("")
    apply_failed = [o for o in res.outcomes if o.outcome == APPLY_FAILED]
    if apply_failed:
        ap(f"* **{len(apply_failed)} apply-failure(s)** on the accumulated tree "
           f"for cells that were `measured` in the flat fan-out. See the "
           f"apply-failures table.")
    else:
        ap("* No apply-failures: every measured DISCRIM cell re-generated cleanly "
           "on the accumulated tree.")
    if qb.inert:
        ap(f"* **{len(qb.inert)} measured-INERT cells** slipped past the 2c/2d "
           f"static gates (build + measure but produce byte-identical output).")
    if res.stopped == "stopped_gate_unimplementable":
        ap(f"* **The baseline was unscoreable** — see the blocking finding.")
    else:
        ap(f"* Baseline whole-app p100 = {_fmt(res.baseline_min)} (accept threshold "
           f"{_fmt(res.accept_threshold)}). Confirm this matches expectation for "
           f"the target integral; a large mismatch would indicate snapshot/oracle "
           f"drift.")


def write_report(path, res: SolveResult, qb: QueueBuild, **kw) -> Path:
    path = Path(path)
    path.write_text(build_markdown(res, qb, **kw))
    return path
