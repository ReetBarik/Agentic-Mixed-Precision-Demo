#!/usr/bin/env python3
"""Phase 2f — Tier-B Stage-1: chain-scoped double-double promotion.

Runs the coordinated whole-chain dd promotion (design Phase 2f) on the 4 MEASURED
Tier-B integrals B10/B12/B13/B14 — the highest-confidence COMPUTED targets from the
Item 6/7 bound decomposition.  For each integral it:

  1. clones an isolated working tree + fail-fast builds vanilla,
  2. filters the 5k report to that integral (keeps its cascade_chains),
  3. picks the DOMINANT COMPUTED cascade chain (max measured rel-err, tightness in
     the COMPUTED band) and builds ONE ``chain_dd`` candidate spanning its lines,
  4. drives ``agents.solver.solve`` with a chain-aware ``apply_fn`` that routes the
     candidate through the Patcher chain path (via="chain") — coordinated variant
     splice + reroute cascade + chain-scope 2c/2d gates,
  5. accepts under the positive-lift gate (cand p100 >= accumulated + 0.5) or reverts
     with a chain_no_lift / chain_regression tag.

v1 promotes the dominant chain per integral (one coordinated envelope); unioning
multiple chains / all-victims per integral is a Stage-2 refinement.  STOP after
Stage-1 for Reet review — do NOT proceed to Group B / all 21.

Run under the venv + module env with the Argo proxy up, detached so it survives:
    tmux new-session -d -s tierb runs/qcdloop/tier_b_stage1.sh
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from agents.config import PipelineConfig                              # noqa: E402
from agents.patcher.agent import make_patcher_fn                      # noqa: E402
from agents.patcher.fanout import (FanoutSettings, clear_graph_cache,  # noqa: E402
                                   signal_class_map)
from agents.per_integral_orchestrator.filter_report import filter_report  # noqa: E402
from agents.per_integral_orchestrator.orchestrator import _git_clone  # noqa: E402
from agents.shared.bound_decomposition import TIGHT_HI, TIGHT_LO      # noqa: E402
from agents.solver import (ApplyResult, ValidateResult, solve)        # noqa: E402
from agents.solver.queue import Candidate                             # noqa: E402
from agents.strategy.agent import _new_run_id                         # noqa: E402
from agents.strategy.characterization import load_chains             # noqa: E402
from agents.strategy.gitops import GitRepo                            # noqa: E402
from agents.strategy.models import (RegionTarget, RemediationIntent,  # noqa: E402
                                    VIA_CHAIN)
from agents.validator import runner as _runner                        # noqa: E402
from agents.validator import scorer as _scorer                        # noqa: E402
from agents.validator import tail as _tail                            # noqa: E402
from agents.validator.agent import make_validator_fn                  # noqa: E402
from runs.qcdloop.run_strategy_e2e import _build_headers_repo, _git   # noqa: E402

APP_CMAKE_DIR = HERE / "app"
# Phase 2f: chain_dd positive-lift gate (Reet 2026-07-24) — accept iff the chain
# lifts whole-app p100 by >= this vs the accumulated-min before it.
LIFT_MARGIN = 0.5
# Reet's target correctness tolerance (validate() reporting-only; the solver gate is
# lift-relative and reads min_precise_digits directly, which is tolerance-independent).
VALIDATE_TOLERANCE = 6.0
STAGE1_INTEGRALS = ["B10", "B12", "B13", "B14"]


def _is_computed(chain) -> bool:
    return chain.tightness is not None and TIGHT_LO <= chain.tightness <= TIGHT_HI


def _dominant_computed_chain(chains):
    """The floor-driving COMPUTED cascade chain (max measured rel-err), or None."""
    computed = [c for c in chains if _is_computed(c)]
    if not computed:
        return None
    return max(computed, key=lambda c: c.max_rel_err)


def _chain_lines_of(chain) -> list[tuple[str, int, int]]:
    """Deduped (file, line_start, line_end) tuples spanning the chain."""
    seen: set = set()
    out: list[tuple[str, int, int]] = []
    for t in chain.lines:
        key = (t.file, t.line_start, t.line_end)
        if key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _run_one_integral(integral: str, args, out_root: Path, vanilla_headers: Path,
                      base_repo: Path, starting_sha: str) -> dict:
    """Run Tier-B on one integral; return a summary dict for the aggregate report."""
    out_dir = out_root / integral
    out_dir.mkdir(parents=True, exist_ok=True)

    # -- isolated tree clone + fail-fast vanilla build --
    tree = out_dir / f"tree_{integral}"
    if tree.exists():
        shutil.rmtree(tree)
    _git_clone(base_repo, tree)
    _runner.build_driver(tree, "vanilla", out_dir / "vanilla_gate_build",
                         Path(args.kokkos_root))

    # -- filtered per-integral report (keeps cascade_chains) --
    filtered = out_dir / f"report_{integral}.json"
    filter_report(Path(args.report).resolve(), integral, filtered)
    idata = json.loads(filtered.read_text())["integrals"][integral]
    report_regions = idata["regions"]

    # -- pick the dominant COMPUTED cascade chain -> one chain_dd candidate --
    chains, cmeta = load_chains(filtered)
    n_computed = sum(1 for c in chains if _is_computed(c))
    dom = _dominant_computed_chain(chains)
    summary = {"integral": integral, "n_chains": cmeta.get("n_chains", 0),
               "n_computed": n_computed}
    if dom is None:
        summary.update(outcome="no_computed_chain", chain_id=None)
        print(f"[{integral}] no COMPUTED cascade chain — skipping", flush=True)
        return summary

    chain_lines = _chain_lines_of(dom)
    cand = Candidate(
        region_id=dom.chain_id, rung="chain_dd", kind="double-to-dd",
        intent="correctness", via="chain",
        delta_effective=1e-30, baseline_delta_effective=1e-4,   # placeholder DISCRIM
        chain_lines=tuple(chain_lines), predicted_lift=dom.predicted_lift,
        # Phase 2f kernel-scope: gate this chain against ITS integral's own floor, not
        # the whole-app p100 (pinned by whichever kernel is worst — B12's hotspot).
        target_kernel=integral)

    print(f"\n=== Tier-B Stage-1: {integral} ===", flush=True)
    print(f"  chains total/computed : {cmeta.get('n_chains',0)} / {n_computed}", flush=True)
    print(f"  dominant chain        : {dom.chain_id}", flush=True)
    print(f"  tightness             : {dom.tightness:.3e}", flush=True)
    print(f"  measured max_rel_err  : {dom.max_rel_err:.3e}", flush=True)
    print(f"  predicted dd lift     : +{dom.predicted_lift:.2f} digits", flush=True)
    print(f"  chain_lines ({len(chain_lines)}):", flush=True)
    for f, ls, le in chain_lines:
        print(f"      {f}:{ls}" + (f"-{le}" if le != ls else ""), flush=True)

    # -- pipeline wiring (mirrors run_solver_stage1.main) --
    run_id = _new_run_id(str(starting_sha) + integral + dom.chain_id)
    branch = f"tierb/{run_id}"
    run_dir = out_dir / "patcher_runs"
    run_dir.mkdir(parents=True, exist_ok=True)

    clear_graph_cache()
    fanout = FanoutSettings(entry_point=args.entry_point, integral=integral,
                            app_source_roots=[str(HERE / "src")],
                            signal_class_by_region=signal_class_map(report_regions))
    build_config = {"app_cmake_dir": str(APP_CMAKE_DIR), "kokkos_root": args.kokkos_root}
    patcher_fn = make_patcher_fn(build_config=build_config,
                                 config=PipelineConfig(), fanout=fanout)
    snapshot = {"seed": args.seed, "sample_count": args.sample_count}
    base_state = {
        "vanilla_headers": str(vanilla_headers),
        "dd_source_repo": args.dd_repo, "dd_ref": args.dd_ref,
        "accepted_patches": [], "kokkos_root": args.kokkos_root,
        "tail_samples": _tail.load_tail_samples(filtered),
    }
    validator_fn = make_validator_fn(
        base_state, starting_sha, str(tree), tolerance=args.tolerance,
        baseline_spec=_scorer.qcdloop_baseline_spec())

    repo = GitRepo(tree)
    repo.create_branch(branch, starting_sha)
    _iter = {"i": 0}
    _diag = {"baseline_hotspot": None}

    def apply_fn(c, parent):
        i = _iter["i"]; _iter["i"] += 1
        clear_graph_cache()      # graph must reflect the accumulated tree
        f0, l0, e0 = c.chain_lines[0]
        intent = RemediationIntent(
            target=RegionTarget(file=f0, line_start=l0, line_end=e0, variables=[]),
            kind="double-to-dd", intent="correctness", current_precision="double",
            rationale_id=c.region_id, via=VIA_CHAIN, chain_lines=list(c.chain_lines))
        ctx = {"run_id": run_id, "branch": branch, "repo_path": str(tree),
               "parent_sha": parent, "run_dir": str(run_dir), "iter_id": i}
        t0 = time.monotonic()
        resp = patcher_fn(intent.to_patcher(), ctx)
        arts = resp.get("artifacts") or {}
        return ApplyResult(
            ok=(resp.get("status") == "ok"),
            candidate_sha=resp.get("candidate_sha"),
            patcher_status=resp.get("status"),
            gate_binary=arts.get("gate_binary"),
            gate_tree_hash=arts.get("gate_tree_hash"),
            error=resp.get("error"),
            wall_sec=round(time.monotonic() - t0, 1))

    def validate_fn(candidate_sha, gate_binary, gate_tree_hash):
        ctx = {"run_id": run_id, "branch": branch, "repo_path": str(tree),
               "tolerance": args.tolerance, "snapshot": snapshot,
               "iter_id": _iter["i"], "gate_binary": gate_binary,
               "gate_tree_hash": gate_tree_hash}
        t0 = time.monotonic()
        v = validator_fn(candidate_sha, ctx)
        cand_stats = v.get("candidate") or {}
        curr_stats = v.get("current") or {}
        if _diag["baseline_hotspot"] is None:
            _diag["baseline_hotspot"] = curr_stats.get("hotspot")
        return ValidateResult(
            cand_min=cand_stats.get("min_precise_digits"),
            curr_min=curr_stats.get("min_precise_digits"),
            combined_cand_min=v.get("cand_min_precise_digits"),
            # Phase 2f kernel-scope: per-integral floors for the kernel-scoped gate.
            cand_per_kernel=cand_stats.get("per_integral_min_precise_digits") or {},
            curr_per_kernel=curr_stats.get("per_integral_min_precise_digits") or {},
            verdict=v.get("verdict"), verdict_reason=v.get("verdict_reason"),
            wall_sec=round(time.monotonic() - t0, 1))

    def revert_fn(parent):
        repo.reset_hard(parent)

    def head_fn():
        return repo.head()

    def on_event(o):
        tag = f" [{o.reason_tag}]" if o.reason_tag else ""
        print(f"  {o.candidate.region_id} -> {o.outcome}{tag} "
              f"min {o.min_before}->{o.min_after} ({o.reason}) [{o.wall_sec}s]",
              flush=True)

    t_solve = time.monotonic()
    res = solve([cand], apply_fn=apply_fn, validate_fn=validate_fn,
                revert_fn=revert_fn, head_fn=head_fn, margin=args.margin,
                all_region_ids={dom.chain_id}, on_event=on_event)
    solve_wall = round(time.monotonic() - t_solve, 1)

    # -- persist per-integral artifacts --
    diff_path = out_dir / "final.diff"
    repo.write_cumulative_diff(starting_sha, diff_path)
    o = res.outcomes[0] if res.outcomes else None
    # Whole-app lift (pinned by the worst kernel — kept for continuity / cross-kernel
    # visibility) and the kernel-scoped lift the gate actually decided on (Phase 2f).
    measured_lift = (
        (res.final_min - res.baseline_min)
        if (res.final_min is not None and res.baseline_min is not None) else None)
    k_baseline = res.baseline_by_kernel.get(integral)
    k_final = res.final_by_kernel.get(integral)
    kernel_lift = (
        (k_final - k_baseline)
        if (k_final is not None and k_baseline is not None) else None)
    summary.update(
        chain_id=dom.chain_id, tightness=dom.tightness,
        measured_max_rel_err=dom.max_rel_err, predicted_lift=dom.predicted_lift,
        chain_lines=[f"{f}:{ls}" for f, ls, _ in chain_lines],
        baseline_min=res.baseline_min, final_min=res.final_min,
        measured_lift=measured_lift,
        # Phase 2f kernel-scope: this integral's OWN floor + the lift the gate decided on.
        kernel_baseline_min=k_baseline, kernel_final_min=k_final,
        kernel_measured_lift=kernel_lift,
        baseline_by_kernel=res.baseline_by_kernel, final_by_kernel=res.final_by_kernel,
        outcome=(o.outcome if o else "no_outcome"),
        reason_tag=(o.reason_tag if o else None),
        patcher_status=(o.patcher_status if o else None),
        reason=(o.reason if o else None),
        declared_dd=res.region_final.get(dom.chain_id) == "chain_dd",
        stopped=res.stopped, solve_wall_sec=solve_wall,
        baseline_hotspot=_diag["baseline_hotspot"],
        diff_path=str(diff_path), tree=str(tree))
    (out_dir / "tierb_result.json").write_text(json.dumps(summary, indent=2))
    print(f"  -> whole-app baseline {res.baseline_min} final {res.final_min} "
          f"(lift {measured_lift}); kernel[{integral}] baseline {k_baseline} "
          f"final {k_final} (lift {kernel_lift}) outcome {summary['outcome']}",
          flush=True)
    return summary


def _write_report(path: Path, results: list[dict], args, today: str) -> None:
    lines = [f"# Tier-B Stage-1 — chain-scoped dd promotion ({today})", "",
             "Phase 2f coordinated whole-chain double-double promotion on the 4 "
             "measured Tier-B integrals. v1 promotes the dominant COMPUTED cascade "
             "chain per integral (one coordinated envelope).",
             "",
             f"- gate: positive lift >= {args.margin} digits vs accumulated-min "
             f"(chain_dd); tolerance {args.tolerance} (reporting-only)",
             f"- seed {args.seed}, sample_count {args.sample_count}, entry {args.entry_point}",
             "",
             "## Per-integral outcome (kernel-scoped gate)", "",
             "The gate now scores each chain against ITS integral's own p100 floor "
             "(kernel-scope, Reet 2026-07-25), not the whole-app min pinned by the "
             "worst kernel (B12's hotspot). Whole-app columns are kept for cross-kernel "
             "visibility.",
             "",
             "| I | kernel baseline | kernel final | kernel lift | predicted lift | "
             "app baseline | app final | outcome | chain | lines |",
             "|---|---|---|---|---|---|---|---|---|---|"]

    def _f(x):
        return f"{x:.4f}" if isinstance(x, (int, float)) else "—"

    def _lift(x):
        return f"{x:+.2f}" if isinstance(x, (int, float)) else "—"

    for r in results:
        if r.get("outcome") == "no_computed_chain":
            lines.append(f"| {r['integral']} | — | — | — | — | — | — | "
                         f"no_computed_chain | — | — |")
            continue
        pl = r.get("predicted_lift")
        pl_s = f"+{pl:.2f}" if isinstance(pl, (int, float)) else "—"
        oc = r.get("outcome", "?")
        if r.get("reason_tag"):
            oc = f"{oc} ({r['reason_tag']})"
        lines.append(
            f"| {r['integral']} | {_f(r.get('kernel_baseline_min'))} | "
            f"{_f(r.get('kernel_final_min'))} | {_lift(r.get('kernel_measured_lift'))} | "
            f"{pl_s} | {_f(r.get('baseline_min'))} | {_f(r.get('final_min'))} | "
            f"{oc} | {r.get('chain_id','')} | {len(r.get('chain_lines',[]))} |")
    lines += ["", "## Predicted vs measured lift (kernel-scoped)", ""]
    for r in results:
        if r.get("outcome") == "no_computed_chain":
            continue
        lines.append(
            f"- **{r['integral']}** ({r.get('chain_id')}): predicted "
            f"+{r.get('predicted_lift', 0):.2f}, kernel-measured "
            f"{_lift(r.get('kernel_measured_lift'))} "
            f"({_f(r.get('kernel_baseline_min'))} -> {_f(r.get('kernel_final_min'))}), "
            f"whole-app lift {_lift(r.get('measured_lift'))}, "
            f"tightness {r.get('tightness')}, "
            f"patcher_status={r.get('patcher_status')}, "
            f"declared_dd={r.get('declared_dd')}")
        lines.append(f"    - lines: {', '.join(r.get('chain_lines', []))}")
    lines += ["", "## Notes",
              "- Kernel-scope gate (Reet 2026-07-25): each chain gated against its own "
              "integral's p100 floor, not the whole-app min (which B12's hotspot pins). "
              "The whole-app gate rejected B14 as chain_no_lift because it couldn't move "
              "the global min; the kernel-scope gate measures B14's own coefficient lift.",
              "- Chain-scope 2d-B (Fix 1): the gate now fires only on INTERIOR chain "
              "regions; the outermost region's exit-truncation is the designed output "
              "boundary and is exempt (was false-positiving B10/B12 pre-build).",
              "- STOP after Stage-1 for review; Group B / all-21 not run.",
              "- v1 = dominant chain per integral; multi-chain union deferred to Stage-2.",
              ""]
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--integrals", default=",".join(STAGE1_INTEGRALS),
                    help="comma-separated integrals (default B10,B12,B13,B14)")
    ap.add_argument("--report", default=str(HERE / "report_5k.json"))
    ap.add_argument("--out-dir", default=str(HERE / "tier_b_stage1"))
    ap.add_argument("--base-repo", default=str(Path.home() / "amp_tierb_stage1_base_repo"))
    ap.add_argument("--dd-repo", default=str(Path.home() / "qcdloop"))
    ap.add_argument("--dd-ref", default="ddfun_enabled")
    ap.add_argument("--kokkos-root", default=str(Path.home() / "kokkos-install"))
    ap.add_argument("--sample-count", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--entry-point", default="BO")
    ap.add_argument("--margin", type=float, default=LIFT_MARGIN,
                    help="chain_dd positive-lift threshold in digits (default 0.5).")
    ap.add_argument("--tolerance", type=float, default=VALIDATE_TOLERANCE)
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args(argv)

    integrals = [s.strip() for s in args.integrals.split(",") if s.strip()]
    out_root = Path(args.out_dir).resolve()
    if args.clean and out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if not Path(args.report).is_file():
        raise SystemExit(f"missing report: {args.report}")

    # -- pristine-baseline guard + shared base repo (one clone source for all) --
    vanilla_headers = REPO / "runs" / "qcdloop_headers_full"
    dirty = _git(REPO, "status", "--porcelain", str(vanilla_headers))
    if dirty:
        raise SystemExit(f"main tree {vanilla_headers} is dirty; refusing to run:\n{dirty}")
    base_repo = Path(args.base_repo).resolve()
    starting_sha = _build_headers_repo(base_repo, vanilla_headers)

    print("=== Tier-B Stage-1 config ===", flush=True)
    print(f"  integrals   : {integrals}", flush=True)
    print(f"  report      : {args.report}", flush=True)
    print(f"  out         : {out_root}", flush=True)
    print(f"  starting_sha: {starting_sha}", flush=True)
    print(f"  gate        : chain lift >= {args.margin} digits vs accumulated", flush=True)
    print("=============================", flush=True)

    results = []
    for integral in integrals:
        try:
            results.append(_run_one_integral(integral, args, out_root, vanilla_headers,
                                             base_repo, starting_sha))
        except Exception as exc:   # noqa: BLE001 - one integral's crash must not kill the batch
            import traceback
            traceback.print_exc()
            results.append({"integral": integral, "outcome": "harness_error",
                            "error": repr(exc)})

    today = datetime.date.today().isoformat()
    md = out_root / f"TIER_B_STAGE1_{today}.md"
    _write_report(md, results, args, today)
    (out_root / "tierb_stage1_results.json").write_text(json.dumps(results, indent=2))

    print("\n=== Tier-B Stage-1 summary (kernel-scoped) ===", flush=True)
    for r in results:
        print(f"  {r['integral']:5s} {r.get('outcome','?'):28s} "
              f"kernel_lift={r.get('kernel_measured_lift')} "
              f"(pred +{r.get('predicted_lift', 0)}) app_lift={r.get('measured_lift')}",
              flush=True)
    print(f"  wrote {md}", flush=True)
    print("  STOP after Stage-1 — Reet reviews before Group B / all 21.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
