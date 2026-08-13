#!/usr/bin/env python3
"""T4 probe — emit + compile a per-group QF flip TU.

Drives the SAME machinery the tu_only walk uses (``tu_emit.emit_flip_tu`` into a clone
of the pristine snapshot, then the bare ``g++`` compile from
``runs/qcdloop/tu_provider``), but for a single group and with the QF profile forced
available.  Its whole purpose is to surface the real compile-error set for QF so the
vendored-header local patches are driven by evidence rather than by the dd/ff precedent.

Usage:
    python runs/qcdloop/qf_flip_probe.py [--group box/B1m.h ...] [--precision qf]

Prints, per group, either the run's first RES line or the g++ error tail.  Exits
non-zero if any group fails to build.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import replace
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from agents.patcher import tu_emit                          # noqa: E402
from agents.patcher.precision_flip import TargetPrecision   # noqa: E402
from runs.qcdloop import tu_provider as tp                  # noqa: E402

SNAPSHOT = _REPO / "runs" / "qcdloop_headers_full"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", action="append", default=None,
                    help="group header (repeatable); default = every box/B*m.h")
    ap.add_argument("--precision", default="qf")
    ap.add_argument("--out", default=str(_REPO / "runs" / "qcdloop" / "qf_probe_out"))
    ap.add_argument("--kokkos-root", default=str(Path.home() / "kokkos-install"))
    ap.add_argument("--sample-count", type=int, default=8)
    args = ap.parse_args()

    target = TargetPrecision(args.precision)
    # Force the profile available for the probe ONLY — the table stays unavailable until
    # a TU is proven to build (that is exactly what this script decides).
    prof = tu_emit.PROFILES[target]
    if not prof.available:
        tu_emit.PROFILES[target] = replace(prof, available=True)
        print(f"[probe] {target.value}: profile forced available for this run only")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clone = out / "tree"
    if clone.exists():
        shutil.rmtree(clone)
    shutil.copytree(SNAPSHOT, clone)

    kokkos = Path(args.kokkos_root)
    groups = args.group or [f"box/{p.name}" for p in sorted((clone / "box").glob("B*m.h"))]

    failures = 0
    for group in groups:
        stem = Path(group).stem
        tu = tu_emit.emit_flip_tu(clone, group, out / "flip_drv", target)
        build_dir = out / f"flip_build_{stem}_{target.value}"
        build_dir.mkdir(parents=True, exist_ok=True)
        binary = build_dir / f"flip_{stem}_{target.value}"
        inc = (f"-I{clone} -I{clone}/box -I{tp.RECIPES} -I{tp.VEND} "
               f"-I{kokkos}/include")
        lib = (f"-L{kokkos}/lib -L{kokkos}/lib64 "
               f"-lkokkoscore -lkokkoscontainers -ldl")
        r = tp._bash(f"g++ -std=c++20 -O2 -w {tp._backend_flags(kokkos)} {inc} "
                     f"{tu.driver_path} -o {binary} {lib}")
        log = r.stdout + r.stderr
        (build_dir / "compile.log").write_text(log)
        if r.returncode != 0 or not binary.is_file():
            failures += 1
            errs = [l for l in log.splitlines() if " error:" in l]
            print(f"\n=== {group} {target.value}: BUILD FAILED "
                  f"({len(errs)} error lines) ===")
            print("\n".join(errs[:25]) or log[-2500:])
            continue
        run = tp._bash(f"{binary} --sample-count {args.sample_count}")
        res = [l for l in run.stdout.splitlines() if l.startswith("RES,")]
        print(f"=== {group} {target.value}: BUILT OK, {len(res)} RES lines ===")
        if res:
            print(f"    {res[0][:150]}")
        else:
            failures += 1
            print(f"    !! ran but emitted no RES: {run.stderr[-600:]}")

    print(f"\n[probe] {len(groups) - failures}/{len(groups)} groups built+ran at "
          f"{target.value}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
