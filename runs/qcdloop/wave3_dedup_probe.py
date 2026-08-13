#!/usr/bin/env python3
"""wave3_dedup_probe.py — WAVE3 DEDUP pre-flight probe (validates the fix).

Unlike ``wave3_probe.py`` (which landed each region off the PRISTINE base to prove
the collision is sibling-context-dependent), this probe lands TWO C-COLL regions
**sequentially against the same growing branch**: region A off the pristine base,
then region B off region A's committed candidate. So region B is forced to merge
into the ``Constants<DoubleDouble>`` that region A already installed — the exact TU
assembly that produced ``redefinition of 'struct ql::Constants<...>'`` in the 10k.

Success criteria (task spec):
  1. Both attempts return status=ok.
  2. The emitted TU builds cleanly (status=ok ⇒ the vanilla-driver build gate
     passed — no redefinition error).
  3. Exactly ONE ``template<> struct Constants<DoubleDouble>`` in the assembled TU
     after both commits.
  4. The single ``Constants<DoubleDouble>`` contains members from BOTH regions.
  5. The user-visible call sites in both regions still reference
     ``Constants<...>::_ieps50()`` / ``_one()`` — no app-code rewrite.

Run under the venv + gcc/13.3.0 + cmake/3.28.3 module env (see run wrapper).
"""
from __future__ import annotations

import re
import subprocess
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

# Two C-COLL regions in the SAME file (B2m.h → same box TU), both touching the
# _ieps50 / _one named-constant family, both `redefinition of Constants<DoubleDouble>`
# in the 10k (WAVE3 §Step-3 identifies B2m.h:64/65 as the identical-source
# _ieps50+_one pair; :84 is the sibling C-COLL used in the step-4 probe).
REGION_A = ("box/B2m.h", 64)
REGION_B = ("box/B2m.h", 84)


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args],
                          capture_output=True, text=True, check=True).stdout


def _count_dd_specs(text: str) -> int:
    """Count ``template<> struct Constants<DoubleDouble> {`` definitions.

    Whitespace- and ``::``-agnostic; the trailing ``{`` excludes the forward
    declaration ``struct Constants;`` and the primary template.
    """
    norm = re.sub(r"\s+", "", text).replace("::", "")
    return len(re.findall(r"(?:struct|class)Constants<quadddfunddouble>\{", norm))


def _land(patcher_fn, repo, run_dir, parent_sha, region, i):
    file, line = region
    intent = {
        "kind": "double-to-dd", "intent": "correctness",
        "current_precision": "double", "rationale_id": f"dedup_probe_{i}",
        "target": {"file": file, "line_start": line, "line_end": line, "variables": []},
    }
    ctx = {"repo_path": str(repo), "parent_sha": parent_sha,
           "run_dir": str(run_dir), "iter_id": i}
    print(f"--- [{i}] land {file}:{line}  (parent {parent_sha[:8]})", flush=True)
    p2 = patcher_fn(intent, ctx)
    status = p2.get("status")
    print(f"    STATUS   : {status}", flush=True)
    if status != "ok":
        art = p2.get("artifacts") or {}
        print(f"    detail   : {(p2.get('detail') or '')[:300]}", flush=True)
        print(f"    build_log: {art.get('build_log_path') or p2.get('build_log_path')}",
              flush=True)
    return p2


def main() -> int:
    vanilla_headers = REPO / "runs" / "qcdloop_headers_full"
    kokkos_root = str(Path.home() / "kokkos-install")
    workdir = Path(tempfile.mkdtemp(prefix="wave3_dedup_probe_"))
    repo = workdir / "headers_repo"
    base = _build_headers_repo(repo, vanilla_headers)
    build_config = {"app_cmake_dir": str(HERE / "app"), "kokkos_root": kokkos_root}
    patcher_fn = make_patcher_fn(build_config=build_config, config=PipelineConfig())
    run_dir = workdir / "run"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=== wave3 DEDUP probe (sequential landing, second merges into first) ===",
          flush=True)
    print(f"  headers_repo : {repo}\n  base_sha     : {base}\n", flush=True)

    pa = _land(patcher_fn, repo, run_dir, base, REGION_A, 0)
    if pa.get("status") != "ok":
        print("\nPROBE FAILED: region A did not accept off pristine base.", flush=True)
        return 1
    sha_a = pa["candidate_sha"]

    pb = _land(patcher_fn, repo, run_dir, sha_a, REGION_B, 1)
    if pb.get("status") != "ok":
        print("\nPROBE FAILED: region B did not accept after region A "
              "(the merge did not resolve the collision).", flush=True)
        return 1
    sha_b = pb["candidate_sha"]

    # ---- inspect the assembled TU at region B's committed candidate ----
    print("\n=== assembled TU after BOTH commits (candidate " f"{sha_b[:8]}) ===",
          flush=True)
    canonical = _git(repo, "show", f"{sha_b}:ql_shim_dd.h")
    print("\n----- ql_shim_dd.h (canonical merged shim) -----", flush=True)
    print(canonical, flush=True)

    # criterion 3: exactly one Constants<DoubleDouble> across ALL committed headers
    tree = _git(repo, "ls-tree", "-r", "--name-only", sha_b).splitlines()
    total_dd_specs = 0
    for f in tree:
        if f.endswith((".h", ".hpp", ".cpp")):
            total_dd_specs += _count_dd_specs(_git(repo, "show", f"{sha_b}:{f}"))
    print(f"\n[criterion 3] Constants<DoubleDouble> specs across whole TU: {total_dd_specs}",
          flush=True)

    # criterion 4: both regions' members present in the one spec
    members = [m for m in ("_ieps50", "_one", "_two", "_reps", "_zero")
               if re.search(r"\b" + m + r"\s*\(", canonical)]
    print(f"[criterion 4] members in canonical Constants<DoubleDouble>: {members}", flush=True)

    # criterion 5: app call sites unchanged (still Constants<...>::_ieps50 / _one)
    b2m = _git(repo, "show", f"{sha_b}:box/B2m.h")
    calls_ieps = len(re.findall(r"Constants<[^>]*>::\s*(?:template\s+)?_ieps50", b2m))
    calls_one = len(re.findall(r"Constants<[^>]*>::\s*_one", b2m))
    has_include = '#include "ql_shim_dd.h"' in b2m
    has_dd_free_fn = bool(re.search(r"_ieps50_dd\s*\(|_one_dd\s*\(", b2m))
    print(f"[criterion 5] B2m.h call sites — Constants<>::_ieps50: {calls_ieps}, "
          f"Constants<>::_one: {calls_one}, includes ql_shim_dd.h: {has_include}, "
          f"leaked _name_dd() free fns: {has_dd_free_fn}", flush=True)

    ok = (total_dd_specs == 1 and len(members) >= 2 and not has_dd_free_fn
          and has_include)
    print("\n=== PROBE RESULT:", "PASS ✅" if ok else "FAIL ❌", "===", flush=True)
    print(f"  region A : {REGION_A[0]}:{REGION_A[1]}  -> {pa['status']}", flush=True)
    print(f"  region B : {REGION_B[0]}:{REGION_B[1]}  -> {pb['status']}", flush=True)
    print(f"  workdir kept: {workdir}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
