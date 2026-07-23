#!/usr/bin/env python3
"""Phase-2a compile de-risk probe: does ONE fan-out variant actually build?

Clones the base headers repo, builds the BO-rooted call graph, runs the real
dd/ff integrator (LLM) to install the shim, fans out ONE real B1 region into
per-caller-path variants, then runs the actual vanilla build gate + the
variant-wiring gate.  This isolates the single biggest Phase-2a risk (does the
copied-variant + shim tree compile?) from the expensive whole-app Validator.

Usage (under the venv + module env, proxy up):
    python runs/qcdloop/fanout_build_probe.py --file B0m.h --line 126 --which dd
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from agents.config import PipelineConfig  # noqa: E402
from agents.patcher import fanout, gates  # noqa: E402
from agents.patcher.call_graph import build_call_graph  # noqa: E402
from agents.patcher.dispatch import _precision_cpp  # noqa: E402
from agents.shared.region_scan import extract_region_writes  # noqa: E402


def _clone(base: Path, dest: Path) -> str:
    if dest.exists():
        shutil.rmtree(dest)
    subprocess.run(["git", "clone", "--local", "--quiet", str(base), str(dest)], check=True)
    for k, v in (("user.name", "probe"), ("user.email", "p@l")):
        subprocess.run(["git", "-C", str(dest), "config", k, v], check=True)
    return subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="B0m.h")
    ap.add_argument("--line", type=int, default=126)
    ap.add_argument("--which", choices=["ff", "dd", "float"], default="dd")
    ap.add_argument("--integral", default="B1")
    ap.add_argument("--entry-point", default="BO")
    ap.add_argument("--base-repo", default=str(Path.home() / "amp_per_integral_base_repo"))
    ap.add_argument("--kokkos-root", default=str(Path.home() / "kokkos-install"))
    ap.add_argument("--out", default="/tmp/fanout_probe")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tree = out / "tree"
    sha = _clone(Path(args.base_repo), tree)
    print(f"[probe] cloned base @ {sha[:12]} -> {tree}", flush=True)

    fanout.clear_graph_cache()
    graph = build_call_graph(args.entry_point, tree)
    fd = graph.enclosing_function(args.file, args.line)
    print(f"[probe] region {args.file}:{args.line} in function {fd.name if fd else None}", flush=True)

    # integrator (real LLM) installs the shim + returns tokens
    integ = {"ff": "ff_integrator", "dd": "dd_integrator", "float": "float_integrator"}[args.which]
    mod = __import__(f"agents.{integ}.agent", fromlist=["integrate_region"])
    rel = f"box/{args.file}" if (tree / "box" / args.file).is_file() else args.file
    t0 = time.monotonic()
    res = mod.integrate_region(
        file=rel, line_start=args.line, line_end=args.line, variables=[],
        working_tree=sha, repo_path=str(tree),
        scalar_type={"ff": "ffloat", "dd": "ddouble", "float": "float"}[args.which],
        caller_type="double", direction="in", out_dir=out / "shims", attempt=0,
        cfg=PipelineConfig())
    print(f"[probe] integrator ok={res.ok} tokens={res.llm_tokens} "
          f"({time.monotonic()-t0:.1f}s)", flush=True)
    if not res.ok:
        print(f"[probe] integrator FAILED: {res.error}", flush=True)
        return 2

    scalar_cpp, two_limb = _precision_cpp(args.which)
    try:
        writes = extract_region_writes(rel, args.line, args.line, sha, tracked_type="double")
    except Exception:
        writes = []
    fr = fanout.fan_out_region(
        file=rel, line_start=args.line, line_end=args.line, reads=[], writes=list(writes),
        integral=args.integral, graph=graph, tree_root=str(tree), scalar_type=scalar_cpp,
        two_limb=two_limb, shim_include=f"ql_shim_{args.which}.h")
    print(f"[probe] fan-out declared {len(fr.declared_variants)} variants: "
          f"{fr.declared_variants}", flush=True)
    print(f"[probe] in_place={fr.in_place_region} paths={fr.paths_enumerated} "
          f"touched={[Path(f).name for f in fr.files_touched]}", flush=True)

    # build gate (real cmake vanilla build against the fan-out tree)
    print("[probe] building vanilla driver against fan-out tree ...", flush=True)
    t1 = time.monotonic()
    gate = gates.run_gate(tree, out / "build", out / "logs", 0,
                          kokkos_root=Path(args.kokkos_root))
    print(f"[probe] gate status={gate.status} ({time.monotonic()-t1:.1f}s)", flush=True)
    if not gate.ok:
        print(f"[probe] BUILD FAILED — see {gate.build_log_path}", flush=True)
        tail = Path(gate.build_log_path).read_text()[-2500:] if gate.build_log_path else ""
        print(tail, flush=True)
        return 3

    wiring = gates.check_variant_wiring(tree, fr.declared_variants)
    print(f"[probe] variant-wiring: {'OK' if wiring is None else wiring}", flush=True)
    if fr.declared_variants:
        present, absent = gates.variant_symbols_present(gate.binary_path, fr.declared_variants)
        print(f"[probe] nm symbols present={present} absent(inlined)={absent}", flush=True)
    print("[probe] PASS — fan-out variant tree compiled + wired.", flush=True)
    return 0 if wiring is None else 4


if __name__ == "__main__":
    raise SystemExit(main())
