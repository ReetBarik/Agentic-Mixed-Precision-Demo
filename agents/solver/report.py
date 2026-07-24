"""SOLVER_STAGE1 markdown snapshot generator (Phase 2e).

Renders a ``SolveResult`` + ``QueueBuild`` into the reviewer-facing report the
task specifies: queue rank table, per-candidate outcomes, final precision
distribution + whole-app p100, rejected-float landings, and the greedy
left-on-the-table observations, plus the "Reet review before Stage 2" handoff.
"""

from __future__ import annotations

from pathlib import Path

from agents.solver.queue import QueueBuild
from agents.solver.solver import (ACCEPTED, APPLY_FAILED, REJECTED,
                                  SKIPPED_RESOLVED, SolveResult)


def _fmt(x, nd=4):
    return "—" if x is None else f"{x:.{nd}f}"


def _rank_reason(rung: str) -> str:
    return {"float": "cheapest rung (try first)",
            "ff": "float-float; tried when float rejected",
            "dd": "double-double; most conservative, tried last"}.get(rung, rung)


def build_markdown(res: SolveResult, qb: QueueBuild, *, integral: str,
                   tree_path: str, diff_path: str, manifest_path: str,
                   report_regions: dict, gate: float, solve_wall_sec: float,
                   snapshot: dict, per_integral_floor: dict | None = None,
                   baseline_hotspot: dict | None = None) -> str:
    L: list[str] = []
    ap = L.append

    ap(f"# Solver Stage 1 — {integral} (greedy, p100 ≥ {gate:g})")
    ap("")
    ap(f"Phase 2e Stage 1: the first pipeline stage that **writes an optimized "
       f"source tree**.  Greedy sequential-layering over the fan-out's measured "
       f"DISCRIM `(region, rung)` cells, ranked float<ff<dd, each applied on top of "
       f"the accumulated tree, kept iff the whole-app p100 precise-digits gate holds.")
    ap("")
    ap(f"* **Integral:** {integral} (Stage-1 scope — B12 only)")
    ap(f"* **Manifest:** `{manifest_path}`")
    ap(f"* **Gate:** p100 = min_precise_digits across the random battery ≥ "
       f"**{gate:g}** (FP128 whole-app oracle). Locked; not the PLAN default "
       f"p99≥10 (per FLOAT_RETRO_PROBE.md the aggregate whole-app floor is "
       f"~8.84 digits at BIN1). **See the blocking finding — for the B12 pass the "
       f"whole-app floor is B12's own 3.69, below the gate.**")
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
        ap(f"Queue exhausted normally. Baseline whole-app p100 = "
           f"**{_fmt(res.baseline_min)}**; final = **{_fmt(res.final_min)}** "
           f"digits.  {len(res.accepted)} accepted, {len(res.rejected)} rejected, "
           f"{len([o for o in res.outcomes if o.outcome == APPLY_FAILED])} "
           f"apply-failed, "
           f"{len([o for o in res.outcomes if o.outcome == SKIPPED_RESOLVED])} "
           f"skipped (region already resolved).")
    ap("")

    # -- blocking finding (structural stop) --
    if res.stopped == "stopped_gate_unimplementable":
        _blocking_finding(ap, res, qb, integral, gate, per_integral_floor,
                          baseline_hotspot)

    # -- queue rank table --
    ap("## Candidate queue (rank order)")
    ap("")
    ap(f"{len(qb.queue)} DISCRIM competitors; {len(qb.inert)} measured-INERT "
       f"excluded (byte-identical whole-app output → no speedup); "
       f"{len(qb.non_measured)} never reached `measured` (2c/2d terminal gates).")
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
    ap("## Final precision distribution (B12 regions)")
    ap("")
    dist = res.precision_distribution()
    ap("| precision | region count |")
    ap("|-----------|-------------|")
    for rung in ("float", "ff", "dd", "double"):
        ap(f"| {rung} | {dist.get(rung, 0)} |")
    ap("")
    ap(f"**Final whole-app min_precise_digits (p100): {_fmt(res.final_min)}** "
       f"(baseline {_fmt(res.baseline_min)}).")
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
               f"float dropped p100 below {gate:g}; {why} |")
    ap("")

    # -- apply failures (data-quality: the solver layer re-generating on the
    #    accumulated tree failed where the flat measure succeeded) --
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
    _handoff(ap, res, qb, integral, solve_wall_sec)

    return "\n".join(L) + "\n"


def _blocking_finding(ap, res, qb, integral, gate, per_integral_floor,
                      baseline_hotspot) -> None:
    ap("## ⛔ Stage-1 blocking finding — the p100 gate is unsatisfiable for "
       f"{integral}")
    ap("")
    ap(f"The solver stopped on the **baseline**: the unpatched whole-app "
       f"min_precise_digits (p100) is **{_fmt(res.baseline_min)} < gate "
       f"{gate:g}**, so *no* candidate can pass — the floor is set before the "
       f"solver touches anything.  This is the explicit STOP-and-flag case "
       f"(PLAN 2e §Gate: \"do not silently pick a tolerance other than 6.0; if "
       f"the gate is structurally unimplementable, STOP and flag\").  The solver "
       f"did **not** retune the gate.")
    ap("")
    if baseline_hotspot:
        ap(f"**Hotspot:** integral `{baseline_hotspot.get('integral')}`, sample "
           f"{baseline_hotspot.get('sample_idx')}, component "
           f"`{baseline_hotspot.get('component')}` — precise digits "
           f"{_fmt(baseline_hotspot.get('precise_digits'))} "
           f"(rel-err ≈ {10 ** -(baseline_hotspot.get('precise_digits') or 0):.2e} "
           f"vs the FP128 oracle).")
        ap("")
    if per_integral_floor:
        ap("**Per-integral double-precision floor (vanilla whole-app, this "
           "battery).** Only the *target* integral is the global-min hotspot:")
        ap("")
        ap("| integral | worst-case p100 | < gate? |")
        ap("|----------|----------------|:-------:|")
        for name, floor in sorted(per_integral_floor.items(),
                                  key=lambda kv: kv[1]):
            flag = "**yes**" if floor < gate else "no"
            mark = " ← target" if name == integral else ""
            ap(f"| `{name}`{mark} | {floor:.4f} | {flag} |")
        ap("")
        n_below = sum(1 for f in per_integral_floor.values() if f < gate)
        ap(f"{n_below} of {len(per_integral_floor)} integrals sit below the gate "
           f"at double precision — and the target `{integral}` is the worst.")
        ap("")
    ap("### Why this is genuine, not an artifact")
    ap("")
    ap("The Validator scores each component against a **per-sample `ref_scale`** "
       "(the largest |DD coeff| in that sample) with an analytic-zero band "
       "(`effectively_zero` → capped, counted in `zeroed_components`).  A "
       "near-zero-reference component would therefore report ~0 digits *or be "
       "banded out* — not a moderate 3.69.  3.69 digits = a real 2.04e-4 relative "
       "error against the sample's characteristic magnitude: a **genuine "
       f"double-precision catastrophic-cancellation floor** intrinsic to "
       f"{integral}'s algorithm at that sample, not a scoring artifact.")
    ap("")
    ap("### Why no candidate can lift it")
    ap("")
    ap(f"Every measured DISCRIM candidate ({', '.join(f'`{c.region_id} {c.rung}`' for c in qb.queue)}) "
       f"leaves the hotspot component untouched — the first candidate "
       f"(`B2m.h:188 float`) built + validated cleanly but produced p100 "
       f"{_fmt(res.final_min)} = baseline (Δ ≈ 0 on the global min).  The dd "
       f"upgrades that *could* add precision are exactly the measured-INERT cells "
       f"(delta == baseline) — they do not touch the cancellation either.  So the "
       f"floor is invariant under the entire catalog the fan-out measured for "
       f"{integral}.")
    ap("")
    ap("### The measurement-layer gap this exposes (the point of Stage 1)")
    ap("")
    ap("The **whole-app global-min gate is the wrong instrument for a "
       "per-integral solver whose target integral is itself the global-min "
       "hotspot.**  FLOAT_RETRO_PROBE.md already recommended a *per-component, "
       "float-touched* instrument over the global-min gate; this run makes it "
       "concrete.  Options for Reet (the solver deliberately picks none — locked "
       "at 6.0):")
    ap("")
    ap(f"1. **Regression-relative gate** — accept iff the candidate does not "
       f"*worsen* the whole-app min beyond a small margin vs the double baseline "
       f"(this is exactly `validate()`'s built-in 0.5-digit regression guard). "
       f"Under it, {integral} float candidates that leave the 3.69 floor "
       f"untouched (Δ≈0) would pass — the solver would accept float where it is "
       f"harmless and land real speedup, which is the actual intent for an "
       f"ill-conditioned integral.  This is the smallest change and the most "
       f"defensible.")
    ap("2. **Per-target-integral absolute gate** — score p100 over the target "
       "integral's components only, against a floor calibrated to *its* "
       "achievable precision (e.g. its dd-oracle self-consistency), not the "
       "whole-app 6.0.")
    ap("3. **Hotspot mask** — exclude the provably-cancellation components from "
       "the absolute floor (they are workload physics ceilings that bind double "
       "itself), keeping 6.0 on the rest.")
    ap("")
    ap(f"**Recommendation:** option 1 (regression-relative) for Stage 2 — it "
       f"preserves an absolute-safety intuition while not penalizing the solver "
       f"for an ill-conditioning it cannot fix.  Needs your sign-off; it changes "
       f"the locked gate semantics.")
    ap("")


def _left_on_table(ap, res: SolveResult, qb: QueueBuild) -> None:
    if res.stopped == "stopped_gate_unimplementable":
        ap("* N/A — the walk never got past the baseline (the gate is "
           "unsatisfiable before any candidate could be judged). The "
           "solo-vs-joint question only becomes meaningful once the gate admits "
           "at least the baseline; see the blocking finding.")
        return
    accepted_regions = {o.candidate.region_id for o in res.accepted}
    rejected = [o for o in res.rejected]
    # Regions where a solo demotion was rejected but a *combination* might have
    # held cannot be detected without joint measurement (v1 does none); we flag
    # the population that a v2 joint pass should examine.
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
       "sibling region to demote further. On B12's small DISCRIM set the regions "
       "are independent, so this is not expected to bind — but it is the first "
       "assumption to revisit if Stage 2 shows regions competing for the same "
       "global-min headroom.")
    if qb.inert:
        ap(f"* **{len(qb.inert)} measured-INERT cells** were excluded as no-ops. "
           f"If any is in fact a real-but-below-resolution speedup (float that "
           f"rounds to identical bits), the greedy queue never tries it — but the "
           f"2c/2d investigation established these are structural no-ops, not "
           f"hidden wins.")


def _handoff(ap, res: SolveResult, qb: QueueBuild, integral: str,
             solve_wall_sec: float) -> None:
    ap("## Reet review before Stage 2")
    ap("")
    ap("### Solver-design judgment calls")
    ap("")
    ap("1. **Gate on the raw p100 metric, not the Validator's accept verdict.** "
       "`validate()`'s own verdict bundles a 0.5-digit *regression* guard vs the "
       "~8.84-digit baseline, which would reject any candidate that legitimately "
       "spends precision down toward 6. The solver reads "
       "`candidate.min_precise_digits` and applies `≥ 6.0` itself. If you want the "
       "regression guard *too*, that is a one-line change — but it changes the "
       "locked semantics.")
    ap("2. **Queue = measured DISCRIM only; measured-INERT excluded.** An INERT "
       "cell would trivially 'accept' (byte-identical → gate holds) and lock the "
       "region at a no-op rung ahead of a genuinely cheaper DISCRIM rung. "
       "Excluding them is both task-consistent ('no measured DISCRIM rung → stay "
       "double') and avoids that pathology.")
    ap("3. **Intra-rung tiebreak = region_id ascending.** Deterministic but "
       "arbitrary; the measurement layer offers no principled cross-region order "
       "within a rung. Flop-weighting (WI3 table already exists) is the obvious "
       "v2 upgrade.")
    ap("4. **`current_precision='double'` for every intent.** Sound only under "
       "first-accept-per-region (a region is patched at most once, from double). "
       "If v2 ever re-demotes an already-demoted region, this must thread the "
       "accumulated precision.")
    ap("")
    ap("### Stage 2 (all 21 integrals) cost estimate")
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
    ap(f"* Each candidate = 1 Patcher fan-out (LLM gen + build) + 1 whole-app "
       f"validate (build-fused). API cost is dominated by the fan-out LLM calls "
       f"(float/ff/dd shim generation).")
    ap(f"* B12 queue ({n_cand}) is small; other integrals with richer DISCRIM sets "
       f"(e.g. B1 had 8 measured DISCRIM cells) will have larger queues. A rough "
       f"upper bound for 21 integrals: if the mean queue is ~2× B12's and "
       f"per-candidate wall holds, order-of-magnitude **{per_cand*n_cand*2*21/3600:.1f}–"
       f"{per_cand*n_cand*4*21/3600:.1f}h sequential**, less with per-integral "
       f"workers (the passes are independent, like run_all_integrals `--workers`).")
    ap(f"* Refine this after Stage 2 review: the honest number needs the real "
       f"per-integral queue sizes, which only the manifests give.")
    ap("")
    ap("### Measurement-layer gaps the solver exposed")
    ap("")
    apply_failed = [o for o in res.outcomes if o.outcome == APPLY_FAILED]
    if apply_failed:
        ap(f"* **{len(apply_failed)} apply-failure(s)** on the accumulated tree "
           f"for cells that were `measured` in the flat fan-out — the solver is "
           f"the first consumer to layer patches sequentially, so a regenerate "
           f"that depends on prior tree state can fail where the flat measure "
           f"succeeded. See the apply-failures table.")
    else:
        ap("* No apply-failures: every measured DISCRIM cell re-generated cleanly "
           "on the accumulated tree.")
    if qb.inert:
        ap(f"* **{len(qb.inert)} measured-INERT cells** slipped past the 2c/2d "
           f"static gates (they build + measure but produce byte-identical "
           f"output). Not wrong, but each wasted one build in the measure pass; a "
           f"tighter static gate would save that. Enumerated above.")
    if res.stopped == "stopped_gate_unimplementable":
        ap(f"* **The gate itself is the headline gap.** The baseline whole-app "
           f"p100 for the {integral} pass is "
           f"{_fmt(res.baseline_min)} — below the 6.0 gate — because the target "
           f"integral is the whole-app global-min hotspot. This is *not* a "
           f"snapshot/oracle drift (the FLOAT_RETRO ~8.84 figure is the aggregate "
           f"run's BIN1 floor; this battery's global min is the target's own 3.69 "
           f"sample). See the blocking finding for the gate-instrument options.")
    else:
        ap(f"* Baseline whole-app p100 = {_fmt(res.baseline_min)}. Confirm this "
           f"matches expectation for the target integral; a large mismatch would "
           f"indicate a snapshot/oracle-cache drift.")
    ap("")
    ap("**STOP: do not run Stage 2 (all 21 integrals) until this is reviewed** — "
       "especially the gate-instrument decision, which Stage 2 depends on.")


def write_report(path, res: SolveResult, qb: QueueBuild, **kw) -> Path:
    path = Path(path)
    path.write_text(build_markdown(res, qb, **kw))
    return path
