#!/usr/bin/env python3
"""Phase-1 template-arg promotion — L-measure (deliverable 7).

End-to-end honest measurement of the per-integral precision flip on the 11
dd-relevant integrals ({B10,B12,B13,B14} + {B15,B16,BIN0-4}).  Ties deliverables
1-5 together:

  1. clone the pristine snapshot (STOP #Z: snapshot never mutated),
  2. build the VANILLA baseline over the clone (all integrals at double),
  3. for each candidate integral's mass GROUP, emit + build a per-group NARROWING
     flip TU from the clone (deliverable 2 + 4; the instantiation gate = does the
     per-group dd TU compile?),  — built from the snapshot clone ALONE, never the
     ddfun_enabled oracle tree (Decision 2),
  4. build the DD ORACLE reference (QL_MODE=dd against ~/qcdloop/src/qcdloop) — used
     ONLY as the measurement reference (the Validator's oracle black box),
  5. measure per-integral min precise-digits: baseline (vanilla vs dd-ref) and
     candidate (flip vs dd-ref), over N samples,
  6. apply the uniform build-AND-lift>0 gate (deliverable 5).

Structural group discovery (agents.patcher.tu_emit.group_header_for_files); no
integral->group table baked in.  Run under the venv + module env:

    python runs/qcdloop/phase1_lmeasure.py --sample-count 2000
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from array import array
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))

from agents.patcher.flip_gate import GateInputs, evaluate            # noqa: E402
from agents.patcher.precision_flip import TargetPrecision            # noqa: E402
from agents.patcher.tu_emit import emit_flip_tu                       # noqa: E402
from agents.validator.coeffs import N_COMPONENTS, parse_component     # noqa: E402
from agents.validator.precise_digits import precise_digits_fast       # noqa: E402
from agents.validator import runner as _runner                        # noqa: E402

# The 11 dd-relevant integrals (design §3).  Group is discovered structurally at run
# time from the box header that DEFINES each — NOT hard-coded here beyond the target
# list the dispatch named.
TARGETS = ["B10", "B12", "B13", "B14", "B15", "B16",
           "BIN0", "BIN1", "BIN2", "BIN3", "BIN4"]

MODULE = "module use /soft/modulefiles && module load gcc/13.3.0 cmake/3.28.3"
RECIPES = HERE / "src"
VEND = REPO / "third_party" / "include"


def _bash(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-lc", f"{MODULE} && {cmd}"],
                          capture_output=True, text=True)


def _git_archive(repo: Path, ref: str, subpath: str, dest: Path) -> None:
    """Extract ``repo@ref:subpath`` into ``dest`` (repo stays on its branch).

    Mirrors agents.validator.validate._git_archive — the dd oracle is materialized from
    the pinned ddfun_enabled ref, never read from the repo's on-disk working tree."""
    proc = subprocess.run(["git", "-C", str(repo), "archive", ref, subpath],
                          capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git archive {ref}:{subpath} failed:\n"
                           f"{proc.stderr.decode()[-1500:]}")
    tar = subprocess.run(["tar", "-x", "-C", str(dest)], input=proc.stdout,
                         capture_output=True)
    if tar.returncode != 0:
        raise RuntimeError(f"tar extract failed:\n{tar.stderr.decode()[-1500:]}")


def _group_of(integral: str, tree: Path) -> str:
    """The ``box/B<k>m.h`` group header that DEFINES ``integral`` (structural scan)."""
    import re
    for hdr in sorted((tree / "box").glob("B*m.h")):
        text = hdr.read_text(errors="ignore")
        if re.search(rf"\bvoid\s+{re.escape(integral)}\s*\(", text):
            return f"box/{hdr.name}"
    raise RuntimeError(f"no box group header defines {integral} under {tree}")


def _build_flip(clone: Path, group_header: str, drv_dir: Path, build_dir: Path,
                kokkos: Path) -> tuple[Path | None, str]:
    """Emit + compile a per-group narrowing flip TU. Returns (binary|None, log_tail)."""
    tu = emit_flip_tu(clone, group_header, drv_dir, TargetPrecision.DD)
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / f"flip_{Path(group_header).stem}"
    inc = (f"-I{clone} -I{clone}/box -I{RECIPES} -I{VEND} -I{kokkos}/include")
    lib = f"-L{kokkos}/lib -L{kokkos}/lib64 -lkokkoscore -lkokkoscontainers -ldl"
    r = _bash(f"g++ -std=c++20 -O2 -w {inc} {tu.driver_path} -o {binary} {lib}")
    if r.returncode != 0 or not binary.is_file():
        return None, (r.stdout + r.stderr)[-3000:]
    return binary, ""


def _coeffs(binary: Path, total: int) -> dict:
    """Run a driver binary and fold RES lines into {integral:(hi,lo)} flat arrays."""
    r = _bash(f"{binary} --sample-count {total}")
    if r.returncode != 0:
        raise RuntimeError(f"run {binary} failed: {r.stderr[-1500:]}")
    out: dict = {}
    for line in r.stdout.splitlines():
        if not line.startswith("RES,"):
            continue
        parts = line.split(",")
        if len(parts) != 3 + N_COMPONENTS:
            continue
        integ, idx = parts[1], int(parts[2])
        if integ not in out:
            out[integ] = (array("d", bytes(8 * total * N_COMPONENTS)),
                          array("d", bytes(8 * total * N_COMPONENTS)))
        hi, lo = out[integ]
        base = idx * N_COMPONENTS
        for c, tok in enumerate(parts[3:]):
            h, l = parse_component(tok)
            hi[base + c] = h
            lo[base + c] = l
    return out


def _min_digits(cand: dict, ref: dict, integral: str, total: int) -> float | None:
    """min over samples/components of precise_digits(cand vs ref) for one integral."""
    if integral not in cand or integral not in ref:
        return None
    c_hi, c_lo = cand[integral]
    r_hi, r_lo = ref[integral]
    n = min(len(c_hi), len(r_hi)) // N_COMPONENTS
    worst = None
    for s in range(n):
        base = s * N_COMPONENTS
        for c in range(N_COMPONENTS):
            d = precise_digits_fast(c_hi[base + c], c_lo[base + c],
                                    r_hi[base + c], r_lo[base + c])
            if worst is None or d < worst:
                worst = d
    return worst


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample-count", type=int, default=2000)
    ap.add_argument("--out-dir", default=str(HERE / "phase1_lmeasure_out"))
    ap.add_argument("--kokkos-root", default=str(Path.home() / "kokkos-install"))
    ap.add_argument("--dd-repo", default=str(Path.home() / "qcdloop"))
    ap.add_argument("--dd-ref", default="ddfun_enabled")
    ap.add_argument("--margin", type=float, default=0.0)
    # Required, no silent default: the tolerance bar is the user's acceptance criterion
    # (StrategyConfig.tolerance).  Omitting it fails loud rather than assuming a value.
    ap.add_argument("--tolerance", type=float, required=True,
                    help="minimum precise-digit bar (StrategyConfig.tolerance)")
    args = ap.parse_args(argv)

    kokkos = Path(args.kokkos_root)
    out = Path(args.out_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    total = args.sample_count

    snapshot = REPO / "runs" / "qcdloop_headers_full"
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain", str(snapshot)],
                           capture_output=True, text=True).stdout.strip()
    if dirty:
        raise SystemExit(f"snapshot dirty; refusing to run (STOP #Z):\n{dirty}")

    clone = out / "tree"
    shutil.copytree(snapshot, clone)
    print(f"=== Phase-1 L-measure ({total} samples) ===", flush=True)
    print(f"  clone     : {clone}", flush=True)
    print(f"  dd oracle : {args.dd_repo}@{args.dd_ref}", flush=True)

    # 2. vanilla baseline over the clone.
    print("  building vanilla baseline ...", flush=True)
    van_bin = _runner.build_driver(clone, "vanilla", out / "van_build", kokkos)
    van = _coeffs(van_bin, total)

    # 4. dd oracle reference: materialize ddfun_enabled:src/qcdloop via git archive
    #    (the Validator's oracle black box — reference only; never a Phase-1 build input).
    print("  materializing + building dd oracle reference ...", flush=True)
    oracle_tree = out / "dd_oracle_tree"
    oracle_tree.mkdir()
    _git_archive(Path(args.dd_repo), args.dd_ref, "src/qcdloop", oracle_tree)
    # Repoint at third_party/include — the fork ships shadowing dd_/ff_ primitives.
    oracle_headers = _runner.stage_dd_headers(oracle_tree / "src" / "qcdloop")
    dd_bin = _runner.build_driver(oracle_headers, "dd", out / "dd_build", kokkos)
    ref = _coeffs(dd_bin, total)

    # 3. per-group narrowing flip TUs (one build per distinct group).
    group_by_integral = {i: _group_of(i, clone) for i in TARGETS}
    distinct_groups = sorted(set(group_by_integral.values()))
    print(f"  candidate groups: {distinct_groups}", flush=True)
    flip_bin: dict[str, Path] = {}
    flip_fail: dict[str, str] = {}
    flip_coeffs: dict[str, dict] = {}
    for grp in distinct_groups:
        print(f"  building flip TU {grp} ...", flush=True)
        binary, log = _build_flip(clone, grp, out / "flip_drv",
                                  out / f"flip_build_{Path(grp).stem}", kokkos)
        if binary is None:
            flip_fail[grp] = log
            (out / f"flip_build_{Path(grp).stem}.log").write_text(log)
            print(f"    BUILD FAILED (instantiation gate) — see log", flush=True)
            continue
        flip_bin[grp] = binary
        flip_coeffs[grp] = _coeffs(binary, total)

    # 5 + 6. measure per-integral lift + apply gate.
    rows = []
    for integ in TARGETS:
        grp = group_by_integral[integ]
        built = grp in flip_bin
        base_d = _min_digits(van, ref, integ, total)
        cand_d = _min_digits(flip_coeffs[grp], ref, integ, total) if built else None
        gd = evaluate(GateInputs(integ, built=built, baseline_digits=base_d,
                                 candidate_digits=cand_d, tolerance=args.tolerance),
                      margin=args.margin)
        rows.append(dict(integral=integ, group=grp, built=built,
                         baseline_digits=base_d, candidate_digits=cand_d,
                         lift=gd.lift, accept=gd.accept,
                         no_flip_needed=gd.no_flip_needed, reason=gd.reason))
        verdict = ("ACCEPT" if gd.accept else
                   "no_flip_needed" if gd.no_flip_needed else "reject")
        print(f"  {integ:5s} [{Path(grp).stem}] built={built} "
              f"base={base_d if base_d is None else round(base_d,3)} "
              f"cand={cand_d if cand_d is None else round(cand_d,3)} "
              f"lift={gd.lift if gd.lift is None else round(gd.lift,3)} "
              f"-> {verdict}", flush=True)

    result = dict(sample_count=total, margin=args.margin, tolerance=args.tolerance,
                  distinct_groups=distinct_groups,
                  flip_build_failed=sorted(flip_fail), rows=rows)
    (out / "phase1_lmeasure.json").write_text(json.dumps(result, indent=2))
    print(f"\n  wrote {out / 'phase1_lmeasure.json'}", flush=True)

    n_accept = sum(1 for r in rows if r["accept"])
    n_noflip = sum(1 for r in rows if r.get("no_flip_needed"))
    print(f"  accepted {n_accept}/{len(rows)} "
          f"(no_flip_needed {n_noflip}/{len(rows)})", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
