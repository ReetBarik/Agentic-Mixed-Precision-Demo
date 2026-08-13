#!/usr/bin/env python3
"""Phase-2 downshift — L-measure (deliverable 7), FLOAT then FF fallback.

End-to-end honest measurement of the per-integral double->narrower **downshift** on the 10
raw-double integrals (B1-B9, B11 — the Phase-1 non-candidates).  The mechanism is the same
per-group whole-TU flip as Phase-1, walking the downshift preference order
(:data:`agents.patcher.precision_flip.DOWNSHIFT_PREFERENCE` = ``FLOAT`` then ``FF``):

  * **FLOAT** is served by **pipeline-authored shim synthesis** (no static maths header, no
    source enrichment): ``emit_flip_tu`` generates ``kokkosMaths_float_shim.hpp`` into the
    clone alongside the double reference header, and the wrapper includes reference + shim.
  * **FF** is served by its **static enrichment header** ``kokkosMaths_ff.h`` (commit
    d0f5b35, custom ``ql::ffun::ffcomplex`` container — clears STOP #EEE): ``emit_flip_tu``
    at ``TargetPrecision.FF`` emits the dd-style static-header wrapper + driver, no shim.
  * the acceptance gate runs in the DOWNSHIFT direction (``flip_gate`` with
    ``LiftDirection.DOWNSHIFT``): a downshift is accepted iff its p100 candidate digits still
    **clear the tolerance bar** (``candidate_digits >= tolerance + margin``) — precision above
    the bar was headroom, so a negative lift is fine — and rejected iff it drops below the bar.
    The bar is ``--tolerance`` (required; the user's ``StrategyConfig.tolerance``), not the
    raw-double baseline.

Per integral the walk is: try FLOAT (accept iff it clears the bar); if FLOAT rejects,
try FF; the final routing is the first accepted precision, else raw double.  Both attempts
are measured and reported for every integral (FLOAT is expected to reject most/all — float is
too narrow for the box family; FF, at ~10 delivered digits, clears a moderate bar for many).

Everything else is identical to phase1_lmeasure.py: clone the pristine snapshot (STOP #Z),
build the vanilla baseline, build the dd oracle reference from ddfun_enabled via git archive
(reference only — never a build input), build the per-group flip TUs from the clone alone,
measure per-integral min precise-digits (baseline vs candidate, both vs dd-ref), and apply
the gate.

The 11 Phase-1 dd candidates are NOT measured here and NOT touched (STOP #ZZ) — this run's
target list is exactly the raw-double set.  Group discovery is structural.  Run under the
venv + module env:

    python runs/qcdloop/phase2_lmeasure.py --sample-count 2000
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

from agents.patcher.flip_gate import GateInputs, LiftDirection, evaluate     # noqa: E402
from agents.patcher.precision_flip import (                                  # noqa: E402
    DOWNSHIFT_PREFERENCE, TargetPrecision)
from agents.patcher.tu_emit import PROFILES, emit_flip_tu                    # noqa: E402
from agents.validator.coeffs import N_COMPONENTS, parse_component            # noqa: E402
from agents.validator.precise_digits import precise_digits_fast             # noqa: E402
from agents.validator import runner as _runner                              # noqa: E402

# The 10 raw-double integrals (Phase-1 non-candidates).  Group is discovered structurally
# at run time from the box header that DEFINES each — no integral->group table baked in.
TARGETS = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B9", "B11"]

MODULE = "module use /soft/modulefiles && module load gcc/13.3.0 cmake/3.28.3"
RECIPES = HERE / "src"
VEND = REPO / "third_party" / "include"


def _bash(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-lc", f"{MODULE} && {cmd}"],
                          capture_output=True, text=True)


def _fmt(x) -> str:
    """Compact console formatter: '-' for None, else 3-dp round."""
    return "-" if x is None else str(round(x, 3))


def _group_of(integral: str, tree: Path) -> str:
    """The ``box/B<k>m.h`` group header that DEFINES ``integral`` (structural scan)."""
    import re
    for hdr in sorted((tree / "box").glob("B*m.h")):
        text = hdr.read_text(errors="ignore")
        if re.search(rf"\bvoid\s+{re.escape(integral)}\s*\(", text):
            return f"box/{hdr.name}"
    raise RuntimeError(f"no box group header defines {integral} under {tree}")


def _build_flip(clone: Path, group_header: str, drv_dir: Path, build_dir: Path,
                kokkos: Path, target: TargetPrecision) -> tuple[Path | None, str]:
    """Emit + compile a per-group downshift flip TU at ``target``. Returns (binary|None, log).

    ``emit_flip_tu`` selects the emission shape from the profile table: FLOAT synthesizes the
    leaf shim + two-line wrapper (double reference + shim) at ``Kokkos::complex<float>``; FF
    emits the dd-style static-header wrapper (``kokkosMaths_ff.h``) + driver at
    ``ql::ffun::ffcomplex`` (no shim).  Either way the driver is the same per-group shape.
    The instantiation gate = does this target's TU compile?
    """
    tu = emit_flip_tu(clone, group_header, drv_dir, target)
    build_dir.mkdir(parents=True, exist_ok=True)
    binary = build_dir / f"flip_{Path(group_header).stem}_{target.value}"
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
    ap.add_argument("--out-dir", default=str(HERE / "phase2_lmeasure_out"))
    ap.add_argument("--kokkos-root", default=str(Path.home() / "kokkos-install"))
    ap.add_argument("--dd-repo", default=str(Path.home() / "qcdloop"))
    ap.add_argument("--dd-ref", default="ddfun_enabled")
    ap.add_argument("--margin", type=float, default=0.0)
    # Required, no silent default: the tolerance bar is the user's acceptance criterion
    # (StrategyConfig.tolerance).  Omitting it fails loud rather than assuming a value.
    ap.add_argument("--tolerance", type=float, required=True,
                    help="minimum precise-digit bar (StrategyConfig.tolerance)")
    # Override the target list.  Default = the 10 raw-double integrals.  Under a tolerance
    # gate, dd candidates that came back no_flip_needed (double already clears the bar) are
    # also legitimate downshift/speedup candidates and can be passed here explicitly.
    ap.add_argument("--targets", default=None,
                    help="comma-separated integral list (default: the raw-double set)")
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
    print(f"=== Phase-2 float downshift L-measure ({total} samples) ===", flush=True)
    print(f"  clone     : {clone}", flush=True)
    print(f"  dd oracle : {args.dd_repo}@{args.dd_ref}", flush=True)

    # 2. vanilla baseline over the clone (all integrals at double).
    print("  building vanilla baseline ...", flush=True)
    van_bin = _runner.build_driver(clone, "vanilla", out / "van_build", kokkos)
    van = _coeffs(van_bin, total)

    # 4. dd oracle reference (reference only; never a Phase-2 build input).
    print("  materializing + building dd oracle reference ...", flush=True)
    oracle_tree = out / "dd_oracle_tree"
    oracle_headers = _runner.materialize_dd_headers(
        Path(args.dd_repo), args.dd_ref, oracle_tree)
    dd_bin = _runner.build_driver(oracle_headers, "dd", out / "dd_build", kokkos)
    ref = _coeffs(dd_bin, total)

    # 3. per-group downshift flip TUs, one build per (group, precision) in the walk order.
    #    WALK = the downshift preference (FLOAT then FF), filtered to the emission stack's
    #    available targets — never hard-coded here (STOP #SS).
    available = {t for t in DOWNSHIFT_PREFERENCE if PROFILES[t].available}
    walk = [t for t in DOWNSHIFT_PREFERENCE if t in available]
    targets = ([t.strip() for t in args.targets.split(",") if t.strip()]
               if args.targets else TARGETS)
    group_by_integral = {i: _group_of(i, clone) for i in targets}
    distinct_groups = sorted(set(group_by_integral.values()))
    print(f"  candidate groups: {distinct_groups}", flush=True)
    print(f"  downshift walk  : {[t.value for t in walk]}", flush=True)

    # flip_bin[(grp, precision)] -> binary ; flip_coeffs[(grp, precision)] -> coeff dict
    flip_bin: dict[tuple[str, TargetPrecision], Path] = {}
    flip_fail: dict[str, str] = {}
    flip_coeffs: dict[tuple[str, TargetPrecision], dict] = {}
    for grp in distinct_groups:
        for target in walk:
            tag = f"{Path(grp).stem}_{target.value}"
            print(f"  building {target.value} flip TU {grp} ...", flush=True)
            binary, log = _build_flip(clone, grp, out / "flip_drv",
                                      out / f"flip_build_{tag}", kokkos, target)
            if binary is None:
                flip_fail[tag] = log
                (out / f"flip_build_{tag}.log").write_text(log)
                print(f"    BUILD FAILED (instantiation gate) — see log", flush=True)
                continue
            flip_bin[(grp, target)] = binary
            flip_coeffs[(grp, target)] = _coeffs(binary, total)

    # 5 + 6. measure per-integral lift for each precision + apply the DOWNSHIFT gate,
    #        walking FLOAT then FF: the final routing is the first ACCEPTED precision,
    #        else raw double.  Every attempt is recorded (per_precision) for the report.
    rows = []
    for integ in targets:
        grp = group_by_integral[integ]
        base_d = _min_digits(van, ref, integ, total)
        attempts: dict[str, dict] = {}
        final_target = None
        final_accept = False
        for target in walk:
            built = (grp, target) in flip_bin
            cand_d = (_min_digits(flip_coeffs[(grp, target)], ref, integ, total)
                      if built else None)
            gd = evaluate(GateInputs(integ, built=built, baseline_digits=base_d,
                                     candidate_digits=cand_d, tolerance=args.tolerance),
                          margin=args.margin, direction=LiftDirection.DOWNSHIFT)
            attempts[target.value] = dict(built=built, candidate_digits=cand_d,
                                          lift=gd.lift, accept=gd.accept,
                                          reason=gd.reason)
            if gd.accept and final_target is None:
                final_target, final_accept = target.value, True
        final_route = final_target if final_accept else "double"
        rows.append(dict(integral=integ, group=grp, baseline_digits=base_d,
                         per_precision=attempts,
                         final_route=final_route, final_accept=final_accept))
        fa = attempts.get("float", {})
        ff = attempts.get("ff", {})
        print(f"  {integ:5s} [{Path(grp).stem}] "
              f"base={base_d if base_d is None else round(base_d,3)} "
              f"float={_fmt(fa.get('candidate_digits'))}/"
              f"{_fmt(fa.get('lift'))}{'A' if fa.get('accept') else 'r'} "
              f"ff={_fmt(ff.get('candidate_digits'))}/"
              f"{_fmt(ff.get('lift'))}{'A' if ff.get('accept') else 'r'} "
              f"-> {final_route}", flush=True)

    result = dict(sample_count=total, margin=args.margin, tolerance=args.tolerance,
                  direction="downshift",
                  walk=[t.value for t in walk], distinct_groups=distinct_groups,
                  flip_build_failed=sorted(flip_fail), rows=rows)
    (out / "phase2_lmeasure.json").write_text(json.dumps(result, indent=2))
    print(f"\n  wrote {out / 'phase2_lmeasure.json'}", flush=True)

    for target in walk:
        n = sum(1 for r in rows if r["per_precision"].get(target.value, {}).get("accept"))
        print(f"  accepted at {target.value}: {n}/{len(rows)}", flush=True)
    n_final = sum(1 for r in rows if r["final_accept"])
    print(f"  final downshifted (any precision): {n_final}/{len(rows)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
