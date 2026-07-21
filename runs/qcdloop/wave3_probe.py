#!/usr/bin/env python3
"""wave3_probe.py — WAVE3 characterization step-4 probe (read-only wrt agents/).

Drives ``make_patcher_fn`` directly (one dd-integrator intent per region) OFF THE
PRISTINE BASE, i.e. each region's shim is the FIRST into the translation unit — no
sibling shim present. This isolates the *region-intrinsic* generation outcome from
the *TU-assembly collision* that dominates the 10k residual.

Attribution:
  * A region that failed in the 10k with `redefinition of Constants<T>` (collision
    cluster) is expected to ACCEPT here — proving the failure is sibling-context-
    dependent (structural), NOT region-intrinsic and NOT Wave-1-stale.
  * A region that fails region-intrinsically (R4 escape, codegen defect) is
    expected to REPRODUCE here.

Run under the venv + gcc/13.3.0 + cmake/3.28.3 module env (see run wrapper).
"""
from __future__ import annotations

import re
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from agents.config import PipelineConfig  # noqa: E402
from agents.patcher.agent import make_patcher_fn  # noqa: E402

sys.path.insert(0, str(HERE))
from run_strategy_e2e import _build_headers_repo  # noqa: E402

# (file, line, 10k_failure_class, prediction)
REGIONS = [
    ("box/B1m.h", 63,  "collision (redefinition Constants<ddouble>)", "ACCEPT (structural, not intrinsic)"),
    ("box/B2m.h", 84,  "collision (redefinition Constants<ddouble>)", "ACCEPT (structural, not intrinsic)"),
    ("box/B1m.h", 62,  "codegen defect (duplicate 'inline')",          "?? tests postprocess/prompt"),
    ("box/B4m.h", 184, "R4 escape (ql::cLn manual classification)",    "REPRODUCE (intrinsic transcendental)"),
    ("box/B3m.h", 105, "template-id mismatch (Real<ddouble>)",         "?? intrinsic ADL-bridge"),
]


def _err_sig(build_log: str | None) -> str:
    if not build_log or not Path(build_log).exists():
        return "no-log"
    for ln in Path(build_log).read_text(errors="replace").splitlines():
        m = re.search(r"error:\s*(.*)", ln)
        if m:
            return m.group(1).strip()[:100]
    return "no-error-line"


def main() -> int:
    vanilla_headers = REPO / "runs" / "qcdloop_headers_full"
    kokkos_root = str(Path.home() / "kokkos-install")
    workdir = Path(tempfile.mkdtemp(prefix="wave3_probe_"))
    repo = workdir / "headers_repo"
    starting_sha = _build_headers_repo(repo, vanilla_headers)
    build_config = {"app_cmake_dir": str(HERE / "app"), "kokkos_root": kokkos_root}
    patcher_fn = make_patcher_fn(build_config=build_config, config=PipelineConfig())
    run_dir = workdir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=== wave3 probe (isolation, off pristine base) ===", flush=True)
    print(f"  headers_repo : {repo}\n  starting_sha : {starting_sha}\n", flush=True)

    results = []
    for i, (file, line, cls, pred) in enumerate(REGIONS):
        intent = {
            "kind": "double-to-dd", "intent": "correctness",
            "current_precision": "double", "rationale_id": f"probe_{i}",
            "target": {"file": file, "line_start": line, "line_end": line, "variables": []},
        }
        ctx = {"repo_path": str(repo), "parent_sha": starting_sha,
               "run_dir": str(run_dir), "iter_id": i}
        print(f"--- [{i}] {file}:{line}", flush=True)
        print(f"    10k class : {cls}", flush=True)
        print(f"    predict   : {pred}", flush=True)
        p2 = patcher_fn(intent, ctx)
        status = p2.get("status")
        artifacts = p2.get("artifacts") or {}
        build_log = artifacts.get("build_log_path") or p2.get("build_log_path")
        sig = "ok" if status == "ok" else _err_sig(build_log)
        detail = (p2.get("detail") or "")[:150]
        print(f"    STATUS    : {status}", flush=True)
        print(f"    err_sig   : {sig}", flush=True)
        if detail:
            print(f"    detail    : {detail}", flush=True)
        print("", flush=True)
        results.append((f"{file}:{line}", cls, status, sig))

    print("=== SUMMARY (stale-vs-broken split) ===", flush=True)
    for loc, cls, status, sig in results:
        verdict = "STALE/structural (accepts in isolation)" if status == "ok" \
            else "ACTUALLY-BROKEN (intrinsic)"
        print(f"  {loc:14s} {status:16s} {verdict}", flush=True)
        print(f"       10k={cls}  now_sig={sig}", flush=True)
    print(f"\n  workdir kept: {workdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
