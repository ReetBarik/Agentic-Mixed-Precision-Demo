#!/usr/bin/env python3
"""Targeted Patcher end-to-end rerun of the DD regions that were `dd_untested` in
the 2026-07-18 shakedown (run 20260718_194556_67dbcf37), to confirm the
include-hallucination fix (system-prompt C1 + regional include-set lint).

Not the full Strategy loop — this drives ``make_patcher_fn`` directly, one intent
per region: generate the DD shim + boundary patch, apply, build-gate, commit.
Reports per-region P2 status + whether the shim's include set is clean.

Usage (under the venv + gcc/13.3.0 + cmake/3.28.3 module env):
    python runs/qcdloop/rerun_failing_regions.py
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
from agents.integrator_base import regional as _regional  # noqa: E402
from agents.dd_integrator import agent as _dd  # noqa: E402

# run_strategy_e2e is a sibling module (same dir); reuse its headers-repo builder.
sys.path.insert(0, str(HERE))
from run_strategy_e2e import _build_headers_repo  # noqa: E402

# Regions that were dd_untested in run 20260718_194556_67dbcf37.  The include-only
# group had a hallucinated app-source #include as their SOLE blocker; the
# include+R4 group ALSO needs an un-vendored hex constant (_ieps50 = 1e-50) that
# legitimately trips the Rule R4 escape hatch even with a clean include set.
REGIONS = [
    # (file, line, note)
    ("box/B3m.h", 177, "include-only (needs only _one(); was <qcdloop/constants.h>)"),
    ("box/B2m.h", 65,  "include+R4 (k34c; also needs _ieps50 hex → may R4)"),
    ("box/B0m.h", 69,  "include+R4 (k34c; also needs _ieps50 hex → may R4)"),
    ("box/B1m.h", 62,  "include+R4 (k24c; also needs _ieps50 hex → may R4)"),
]

_ALLOWED = _regional._allowed_include_set(_dd._SPEC)


def _shim_include_report(shim_paths: list[str]) -> str:
    bits = []
    for p in shim_paths:
        text = Path(p).read_text(encoding="utf-8")
        bad = _regional._lint_include_set(text, _ALLOWED)
        bits.append(f"{Path(p).name}: {'CLEAN' if bad is None else 'DIRTY:' + bad[:60]}")
    return "; ".join(bits) if bits else "(no shim)"


def _shim_health_report(shim_paths: list[str]) -> str:
    """Gap A/B health: whether any shim still carries a Rule R4 #error (Gap B) or
    injects a namespace bridge (Gap A)."""
    if not shim_paths:
        return "(no shim)"
    bits = []
    for p in shim_paths:
        text = Path(p).read_text(encoding="utf-8")
        r4 = "R4#error" if "#error" in text else "no-R4"
        bridge = "bridge+" if re.search(r"\bnamespace\s+(?!quad\b)\w+\s*\{", text) else "bridge-"
        bits.append(f"{Path(p).name}: {r4},{bridge}")
    return "; ".join(bits)


# The shim-include ORDERING blocker (HANDOFF, e1d774a rerun): the boundary patch
# spliced the shim #include BEFORE the header's own app includes that declare the
# templates the shim specializes, so the compiler saw the specialization first.
_ORDERING_ERROR = "is not a class template"


def _placement_report(repo: Path, rel_file: str, shim_include: str | None) -> str:
    """Confirm the boundary-patch fix: the shim #include now lands AFTER the app
    #includes in the patched header (Task 1).  Reads the header as committed in
    the run repo; ``n/a`` if the file or shim line is absent."""
    if not shim_include:
        return "n/a(no-shim)"
    target = repo / rel_file
    if not target.exists():
        return "n/a(no-file)"
    lines = target.read_text(encoding="utf-8").splitlines()
    shim_line = f'#include "{shim_include}"'
    shim_idx = next((i for i, ln in enumerate(lines) if ln.strip() == shim_line), None)
    if shim_idx is None:
        return "n/a(shim-not-in-header)"
    app_incs = [i for i, ln in enumerate(lines)
                if ln.strip().startswith('#include "') and i != shim_idx]
    if not app_incs:
        return "OK(no-app-includes,top-placed)"
    return "OK(after-app-includes)" if shim_idx > max(app_incs) else "BAD(before-app-includes)"


def _build_signature(build_log_path: str | None) -> str:
    """Classify a build failure: the OLD shim-ordering error (should be gone now)
    vs a genuinely new reason to flag.  Returns ``ok`` when no log / no error."""
    if not build_log_path or not Path(build_log_path).exists():
        return "no-log"
    text = Path(build_log_path).read_text(encoding="utf-8", errors="replace")
    if _ORDERING_ERROR in text:
        return f"OLD-ORDERING-BLOCKER('{_ORDERING_ERROR}')"
    # first genuine compiler error line, if any (flag a NEW reason)
    for ln in text.splitlines():
        low = ln.lower()
        if " error:" in low or low.startswith("error:"):
            return f"NEW-REASON: {ln.strip()[:100]}"
    return "no-error-in-log"


def main() -> int:
    vanilla_headers = REPO / "runs" / "qcdloop_headers_full"
    kokkos_root = str(Path.home() / "kokkos-install")

    workdir = Path(tempfile.mkdtemp(prefix="rerun_failing_"))
    repo = workdir / "headers_repo"
    starting_sha = _build_headers_repo(repo, vanilla_headers)

    build_config = {"app_cmake_dir": str(HERE / "app"), "kokkos_root": kokkos_root}
    patcher_fn = make_patcher_fn(build_config=build_config, config=PipelineConfig())

    run_dir = workdir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=== rerun config ===", flush=True)
    print(f"  headers_repo : {repo}", flush=True)
    print(f"  starting_sha : {starting_sha}", flush=True)
    print(f"  run_dir      : {run_dir}", flush=True)
    print("====================", flush=True)

    results = []
    for i, (file, line, note) in enumerate(REGIONS):
        intent = {
            "kind": "double-to-dd",
            "intent": "correctness",
            "current_precision": "double",
            "rationale_id": f"rerun_{i}",
            "target": {"file": file, "line_start": line, "line_end": line,
                       "variables": []},
        }
        ctx = {
            "repo_path": str(repo),
            "parent_sha": starting_sha,   # every region off the pristine base
            "run_dir": str(run_dir),
            "iter_id": i,
        }
        print(f"\n--- [{i}] {file}:{line} — {note} ---", flush=True)
        p2 = patcher_fn(intent, ctx)
        status = p2.get("status")
        artifacts = p2.get("artifacts") or {}
        shim_paths = artifacts.get("shim_paths") or []
        shim_include = Path(shim_paths[0]).name if shim_paths else None
        inc = _shim_include_report(shim_paths)
        health = _shim_health_report(shim_paths)
        placement = _placement_report(repo, file, shim_include)
        detail = (p2.get("detail") or "")[:200]
        build_log = artifacts.get("build_log_path") or p2.get("build_log_path")
        signature = _build_signature(build_log) if status != "ok" else "ok"
        print(f"    status     : {status}", flush=True)
        print(f"    includes   : {inc}", flush=True)
        print(f"    placement  : {placement}", flush=True)   # Task 1: shim after app includes
        print(f"    build_sig  : {signature}", flush=True)    # OLD ordering blocker vs NEW reason
        print(f"    gapA/gapB  : {health}", flush=True)
        if detail:
            print(f"    detail     : {detail}", flush=True)
        if build_log:
            print(f"    build_log  : {build_log}", flush=True)
        results.append((f"{file}:{line}", status, inc, placement, signature, note))

    print("\n=== SUMMARY ===", flush=True)
    built = 0
    ordering_regressions = 0
    for loc, status, inc, placement, signature, note in results:
        ok = status == "ok"
        built += ok
        if signature.startswith("OLD-ORDERING") or placement.startswith("BAD"):
            ordering_regressions += 1
        print(f"  {loc:16s} status={status:16s} placement={placement:26s} sig={signature}",
              flush=True)
    print(f"\n  built (P2 ok)        : {built}/{len(results)}", flush=True)
    print(f"  shim-ordering blocker: {ordering_regressions}/{len(results)}  "
          f"(Task 1 target: 0 — was the sole B0m.h:69 blocker)", flush=True)
    print(f"  workdir kept         : {workdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
