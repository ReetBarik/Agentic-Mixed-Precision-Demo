#!/usr/bin/env python3
"""Phase 2e Stage 2 — greedy solver across ALL 21 integrals (per-integral trees).

Fans the single-integral solver (:mod:`runs.qcdloop.run_solver_stage1`) across every
integral that has a measured scorer manifest, one isolated pass per integral (own
base-repo clone + out-dir), optionally in parallel workers.  Each pass writes its own
optimized source tree + ``SOLVER_STAGE1_<I>.md`` snapshot; this driver then
aggregates the 21 per-integral ``solver_result.json`` files into ``SOLVER_STAGE2.md``.

Scope (Reet handback, Stage-2 prep): the artifact is **21 per-integral trees on
disk**, each individually valid under its own whole-app validation.  This driver does
NOT merge them into one qcdloop tree — cross-integral merge policy for shared regions
is a subsequent Phase 2f, designed after seeing these 21 trees and their
disagreements (the shared-region disagreement table below is 2f's input).

Gate: regression-relative, 0.5-digit margin vs the double baseline (per integral).

Run detached (survives the session), under the venv + module env with the proxy up:
    tmux new-session -d -s solver2 runs/qcdloop/run_solver_stage2.sh

Usage:
    python runs/qcdloop/run_solver_stage2.py \
        --report        runs/qcdloop/report_5k.json \
        --manifest-dir  runs/qcdloop/per_integral_out_stage2 \
        --out-dir       runs/qcdloop/solver_stage2 \
        --workers 4
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

RUNNER = HERE / "run_solver_stage1.py"


def _manifest_for(manifest_dir: Path, integral: str) -> Path:
    return manifest_dir / integral / f"manifest_scorer_{integral}.jsonl"


def _integrals_with_manifest(manifest_dir: Path, report: Path,
                             requested: list[str] | None) -> list[str]:
    """Integrals to solve: those requested (or all in the report) that HAVE a
    measured scorer manifest on disk (skip + warn for any missing)."""
    if requested:
        names = list(requested)
    else:
        names = sorted(json.loads(report.read_text()).get("integrals", {}))
    have, missing = [], []
    for n in names:
        (have if _manifest_for(manifest_dir, n).is_file() else missing).append(n)
    if missing:
        print(f"[stage2] WARNING no scorer manifest for {missing} under "
              f"{manifest_dir}; skipping (run the measurement pass first).",
              file=sys.stderr)
    return have


def _solve_one(task: dict) -> dict:
    """Run the single-integral solver as an isolated subprocess; return a summary."""
    integral = task["integral"]
    out_dir = Path(task["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / f"solve_{integral}.log"
    cmd = [
        task["python"], str(RUNNER),
        "--integral", integral,
        "--manifest", task["manifest"],
        "--report", task["report"],
        "--out-dir", str(out_dir),
        "--base-repo", task["base_repo"],
        "--dd-repo", task["dd_repo"], "--dd-ref", task["dd_ref"],
        "--kokkos-root", task["kokkos_root"],
        "--sample-count", str(task["sample_count"]), "--seed", str(task["seed"]),
        "--entry-point", task["entry_point"],
        "--margin", str(task["margin"]), "--tolerance", str(task["tolerance"]),
        "--clean",
    ]
    t0 = time.monotonic()
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                              cwd=str(REPO))
    wall = round(time.monotonic() - t0, 1)
    result_json = out_dir / "solver_result.json"
    if proc.returncode != 0 or not result_json.is_file():
        return {"integral": integral, "ok": False, "wall_sec": wall,
                "returncode": proc.returncode, "log": str(log_path)}
    payload = json.loads(result_json.read_text())
    payload["_awaiting_rewrite"] = _count_awaiting(Path(task["manifest"]))
    payload["ok"] = True
    payload["wall_sec"] = wall
    payload["log"] = str(log_path)
    return payload


def _count_awaiting(manifest: Path) -> int:
    """Cells flagged ``awaiting_algorithmic_rewrite`` (signal_class filter) — the
    Kahan/identity plumbing backlog for this integral."""
    n = 0
    if not manifest.is_file():
        return 0
    for line in manifest.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        meta = row.get("patcher_metadata") or {}
        if (meta.get("failure_mode") == "awaiting_algorithmic_rewrite"
                or meta.get("patcher_status") == "awaiting_algorithmic_rewrite"):
            n += 1
    return n


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--report", default=str(HERE / "report_5k.json"))
    ap.add_argument("--integrals", nargs="*", default=None,
                    help="Integral names (default: all in the report that have a "
                         "scorer manifest under --manifest-dir).")
    ap.add_argument("--manifest-dir",
                    default=str(HERE / "per_integral_out_stage2"),
                    help="Root of the measurement pass output (per-integral "
                         "manifest_scorer_<I>.jsonl live under <dir>/<I>/).")
    ap.add_argument("--out-dir", default=str(HERE / "solver_stage2"))
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dd-repo", default=str(Path.home() / "qcdloop"))
    ap.add_argument("--dd-ref", default="ddfun_enabled")
    ap.add_argument("--kokkos-root", default=str(Path.home() / "kokkos-install"))
    ap.add_argument("--sample-count", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--entry-point", default="BO")
    ap.add_argument("--margin", type=float, default=0.5)
    ap.add_argument("--tolerance", type=float, default=10.0)
    args = ap.parse_args(argv)

    report = Path(args.report).resolve()
    manifest_dir = Path(args.manifest_dir).resolve()
    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    integrals = _integrals_with_manifest(manifest_dir, report, args.integrals)
    if not integrals:
        raise SystemExit(f"no integrals with a scorer manifest under {manifest_dir}")

    print("=== solver Stage 2 config ===", flush=True)
    print(f"  report      : {report}", flush=True)
    print(f"  manifest_dir: {manifest_dir}", flush=True)
    print(f"  out_dir     : {out_root}", flush=True)
    print(f"  integrals   : {integrals} ({len(integrals)})", flush=True)
    print(f"  workers     : {args.workers}", flush=True)
    print(f"  gate        : regression-relative, margin {args.margin} vs baseline",
          flush=True)
    print("=============================", flush=True)

    tasks = [{
        "integral": i,
        "manifest": str(_manifest_for(manifest_dir, i)),
        "report": str(report),
        "out_dir": str(out_root / i),
        # per-integral base repo so parallel passes never rebuild a shared clone
        "base_repo": str(Path.home() / f"amp_solver_stage2_base_{i}"),
        "dd_repo": args.dd_repo, "dd_ref": args.dd_ref,
        "kokkos_root": args.kokkos_root,
        "sample_count": args.sample_count, "seed": args.seed,
        "entry_point": args.entry_point,
        "margin": args.margin, "tolerance": args.tolerance,
        "python": sys.executable,
    } for i in integrals]

    t0 = time.monotonic()
    results: list[dict] = []
    if args.workers <= 1:
        for t in tasks:
            print(f"-> solve {t['integral']} ...", flush=True)
            r = _solve_one(t)
            _print_pass(r)
            results.append(r)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_solve_one, t): t for t in tasks}
            for fut in as_completed(futs):
                r = fut.result()
                _print_pass(r)
                results.append(r)
    total_wall = round(time.monotonic() - t0, 1)

    results.sort(key=lambda r: r.get("integral", ""))
    md = _build_stage2_markdown(results, total_wall, args)
    md_path = out_root / "SOLVER_STAGE2.md"
    md_path.write_text(md)
    (out_root / "stage2_results.json").write_text(json.dumps(results, indent=2))
    print(f"\n=== wrote {md_path} ===", flush=True)
    return 0 if all(r.get("ok") for r in results) else 1


def _print_pass(r: dict) -> None:
    if not r.get("ok"):
        print(f"  {r['integral']}: FAILED rc={r.get('returncode')} "
              f"({r.get('log')})", flush=True)
        return
    print(f"  {r['integral']}: baseline={r.get('baseline_min_precise_digits')} "
          f"final={r.get('final_min_precise_digits')} "
          f"dist={r.get('precision_distribution')} "
          f"stopped={r.get('stopped') or '-'} wall={r.get('wall_sec')}s", flush=True)


# ---------------------------------------------------------------------------
# SOLVER_STAGE2.md aggregation
# ---------------------------------------------------------------------------
def _build_stage2_markdown(results: list[dict], total_wall: float, args) -> str:
    L: list[str] = []
    ap = L.append
    ok = [r for r in results if r.get("ok")]
    failed = [r for r in results if not r.get("ok")]

    ap("# Solver Stage 2 — all-integral per-integral solve (regression-relative gate)")
    ap("")
    ap(f"Phase 2e Stage 2: the greedy mixed-precision solver run per integral across "
       f"all {len(results)} integrals with a measured scorer manifest, each producing "
       f"its own optimized source tree under `{args.out_dir}/<I>/tree_<I>`.  Gate is "
       f"regression-relative (candidate p100 ≥ baseline p100 − {args.margin:g}).")
    ap("")
    ap(f"* **Integrals solved:** {len(ok)} ok, {len(failed)} failed")
    ap(f"* **Total wall:** {total_wall}s (workers={args.workers})")
    ap(f"* **Artifact:** {len(ok)} per-integral trees on disk — NOT merged "
       f"(cross-integral merge is Phase 2f; the shared-region disagreement table "
       f"below is its input).")
    ap("")
    if failed:
        ap("## ⚠ Failed passes")
        ap("")
        ap("| integral | returncode | log |")
        ap("|----------|-----------|-----|")
        for r in failed:
            ap(f"| {r['integral']} | {r.get('returncode')} | `{r.get('log')}` |")
        ap("")

    # -- per-integral summary --
    ap("## Per-integral summary")
    ap("")
    ap("| integral | baseline p100 | final p100 | Δ | float | ff | dd | double | "
       "await-rewrite | stopped | wall |")
    ap("|----------|--------------|-----------|---|-------|----|----|--------|"
       "--------------|---------|------|")
    tot = {"float": 0, "ff": 0, "dd": 0, "double": 0}
    tot_await = 0
    for r in ok:
        dist = r.get("precision_distribution", {}) or {}
        for k in tot:
            tot[k] += dist.get(k, 0)
        aw = r.get("_awaiting_rewrite", 0)
        tot_await += aw
        b = r.get("baseline_min_precise_digits")
        f = r.get("final_min_precise_digits")
        delta = (f - b) if isinstance(b, (int, float)) and isinstance(f, (int, float)) else None
        ap(f"| {r['integral']} | {_fmt(b)} | {_fmt(f)} | {_fmt(delta)} | "
           f"{dist.get('float', 0)} | {dist.get('ff', 0)} | {dist.get('dd', 0)} | "
           f"{dist.get('double', 0)} | {aw} | {r.get('stopped') or '-'} | "
           f"{r.get('wall_sec')}s |")
    ap("")

    # -- across-integral rollup --
    ap("## Across-integral totals")
    ap("")
    n_regions_touched = sum(1 for r in ok
                            for rid, rung in (r.get("region_final", {}) or {}).items()
                            if rung != "double")
    ap(f"* **Regions moved off double (across all integrals):** {n_regions_touched} "
       f"(float={tot['float']}, ff={tot['ff']}, dd={tot['dd']}).")
    ap(f"* **`awaiting_algorithmic_rewrite` cells (Kahan/identity backlog):** "
       f"{tot_await} — cancellation-cascade / local-cancellation regions where a "
       f"precision rung is structurally inert; these await the algorithmic-rewrite "
       f"wiring (a separate phase; Strategy models the catalog in "
       f"`agents/strategy/walk.py::_rewrites_for`).")
    n_stopped = sum(1 for r in ok if r.get("stopped"))
    ap(f"* **Passes that STOPPED (unscoreable baseline):** {n_stopped}.")
    ap("")

    # -- shared-region disagreement (Phase 2f input) --
    _shared_region_table(ap, ok)

    # -- handoff --
    _stage2_handoff(ap, ok, failed, tot_await)

    return "\n".join(L) + "\n"


def _shared_region_table(ap, ok: list[dict]) -> None:
    ap("## Shared-region disagreement inventory (Phase 2f input)")
    ap("")
    ap("Regions touched by more than one integral, with the final precision each "
       "integral independently assigned.  A row with >1 distinct precision is a "
       "cross-integral disagreement Phase 2f's merge policy must resolve.")
    ap("")
    # region_id -> {integral: final_precision} (only non-double landings recorded;
    # a region a pass left at double is not a "touch" for merge purposes)
    by_region: dict[str, dict[str, str]] = {}
    for r in ok:
        integ = r["integral"]
        for rid, rung in (r.get("region_final", {}) or {}).items():
            if rung == "double":
                continue
            by_region.setdefault(rid, {})[integ] = rung
    shared = {rid: m for rid, m in by_region.items() if len(m) > 1}
    if not shared:
        ap("_No region was moved off double by more than one integral — no "
           "cross-integral disagreement to resolve._")
        ap("")
        return
    disagree = {rid: m for rid, m in shared.items() if len(set(m.values())) > 1}
    ap(f"{len(shared)} region(s) touched by >1 integral; {len(disagree)} with "
       f"conflicting precision.")
    ap("")
    ap("| region | integrals → precision | conflict? |")
    ap("|--------|----------------------|:---------:|")
    for rid in sorted(shared):
        m = shared[rid]
        tuples = ", ".join(f"{i}→{p}" for i, p in sorted(m.items()))
        conflict = "**yes**" if len(set(m.values())) > 1 else "no"
        ap(f"| `{rid}` | {tuples} | {conflict} |")
    ap("")


def _stage2_handoff(ap, ok, failed, tot_await) -> None:
    ap("## Reet review before Phase 2f")
    ap("")
    ap("### Regression-relative gate behavior in practice")
    ap("")
    stopped = [r for r in ok if r.get("stopped")]
    borderline = [r for r in ok
                  if isinstance(r.get("baseline_min_precise_digits"), (int, float))
                  and r["baseline_min_precise_digits"] < 6.0]
    ap(f"* {len(stopped)} pass(es) stopped on an unscoreable baseline "
       f"(crash/NaN/no min) — {[r['integral'] for r in stopped] or 'none'}.")
    ap(f"* {len(borderline)} integral(s) had a double baseline below the old "
       f"absolute-6 gate but were still solvable under the regression-relative gate "
       f"— {[r['integral'] for r in borderline] or 'none'}. (These are exactly the "
       f"ill-conditioned integrals the Stage-1 absolute gate would have blocked.)")
    ap("")
    ap("### signal_class filter savings")
    ap("")
    ap(f"* {tot_await} `awaiting_algorithmic_rewrite` cells across all integrals — "
       f"each is one precision rung (dd correctness attempt) skipped with no LLM "
       f"generation and no build.  That is the measurement-pass build/LLM saving the "
       f"filter bought, and simultaneously the size of the Kahan/identity plumbing "
       f"backlog Phase 2f-plus must work through.")
    ap("")
    ap("### Shared-region disagreement (Phase 2f input)")
    ap("")
    ap("* See the inventory table above — it enumerates every region two or more "
       "integrals moved off double, and flags where they disagree on the target "
       "precision.  Phase 2f's cross-integral merge policy consumes this.")
    ap("")
    ap("### Measurement-layer gaps exposed at scale")
    ap("")
    if failed:
        ap(f"* {len(failed)} pass(es) failed outright "
           f"({[r['integral'] for r in failed]}); inspect their logs — a solver "
           f"failure at scale that Stage 1 (B12-only) could not have surfaced.")
    else:
        ap("* No pass failed outright.")
    ap("* Confirm each per-integral tree builds + validates on its own before Phase "
       "2f attempts any merge.")


def _fmt(x, nd=4):
    return "—" if not isinstance(x, (int, float)) else f"{x:.{nd}f}"


if __name__ == "__main__":
    raise SystemExit(main())
