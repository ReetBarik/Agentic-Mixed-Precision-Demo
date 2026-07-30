#!/usr/bin/env python3
"""qcdloop whole-TU measure provider — the injectable ``tu_measure_fn`` for the
Strategy ``strategy_mode="tu_only"`` walk.

This is the L-measure recipe (``phase1_lmeasure.py`` + ``phase2_lmeasure.py``)
packaged as the provider contract the generic Strategy walk consumes:

    tu_measure_fn(integral, target) -> dict
        target == "baseline"          -> {"built", "baseline_digits"}
        target in {"dd","float","ff"} -> {"built", "baseline_digits",
                                          "candidate_digits", "log_tail"}

The qcdloop-specific build/oracle/measure lives HERE (runs/qcdloop), not in
agents/strategy — Strategy only calls the injected callable (keeps agents/ free of
app identifiers; feedback_no_placeholder_patterns).

Caching contract (built once, reused across integrals sharing a mass group):

* the pristine snapshot is cloned once into ``out_dir/tree`` (STOP #Z: snapshot
  never mutated),
* the VANILLA baseline binary + coeffs are built once,
* the DD ORACLE reference (git-archived ``ddfun_enabled:src/qcdloop``) is built
  once and used ONLY as the measurement reference (never a flip build input,
  Decision 2),
* each ``(group, precision)`` flip TU is emitted + compiled + measured once and
  memoized; integrals in the same mass group reuse it.

Group discovery is structural (the ``box/B<k>m.h`` header that DEFINES the
integral), matching the L-measure scripts exactly — no integral→group table.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from array import array
from pathlib import Path

from agents.patcher.precision_flip import TargetPrecision
from agents.patcher.tu_emit import emit_flip_tu
from agents.validator.coeffs import N_COMPONENTS, parse_component
from agents.validator.precise_digits import precise_digits_fast
from agents.validator import runner as _runner

_REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
MODULE = "module use /soft/modulefiles && module load gcc/13.3.0 cmake/3.28.3"
RECIPES = HERE / "src"
VEND = _REPO / "third_party" / "include"

_TARGET_BY_NAME = {
    "dd": TargetPrecision.DD,
    "float": TargetPrecision.FLOAT,
    "ff": TargetPrecision.FF,
}


def _bash(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-lc", f"{MODULE} && {cmd}"],
                          capture_output=True, text=True)


def _git_archive(repo: Path, ref: str, subpath: str, dest: Path) -> None:
    """Extract ``repo@ref:subpath`` into ``dest`` (repo stays on its branch)."""
    proc = subprocess.run(["git", "-C", str(repo), "archive", ref, subpath],
                          capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"git archive {ref}:{subpath} failed:\n"
                           f"{proc.stderr.decode()[-1500:]}")
    tar = subprocess.run(["tar", "-x", "-C", str(dest)], input=proc.stdout,
                         capture_output=True)
    if tar.returncode != 0:
        raise RuntimeError(f"tar extract failed:\n{tar.stderr.decode()[-1500:]}")


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


class TUMeasureProvider:
    """Stateful whole-TU measure provider.  One instance per Strategy run.

    Build everything lazily on the first ``__call__`` so construction is cheap and
    a run that never measures (e.g. an empty report) does no builds.
    """

    def __init__(self, *, out_dir: Path, kokkos_root: Path,
                 dd_repo: Path, dd_ref: str, sample_count: int,
                 snapshot: Path | None = None):
        self.out_dir = Path(out_dir)
        self.kokkos = Path(kokkos_root)
        self.dd_repo = Path(dd_repo)
        self.dd_ref = dd_ref
        self.total = int(sample_count)
        self.snapshot = (Path(snapshot) if snapshot is not None
                         else _REPO / "runs" / "qcdloop_headers_full")

        self.clone: Path | None = None
        self._van: dict | None = None
        self._ref: dict | None = None
        self._group_of: dict[str, str] = {}
        # (group, precision) -> coeff dict | None (None = build failed)
        self._flip_coeffs: dict[tuple[str, str], dict | None] = {}
        self._flip_log: dict[tuple[str, str], str] = {}

    # -- lazy one-time setup -------------------------------------------------
    def _ensure_setup(self) -> None:
        if self.clone is not None:
            return
        self.out_dir.mkdir(parents=True, exist_ok=True)
        clone = self.out_dir / "tree"
        if clone.exists():
            shutil.rmtree(clone)
        shutil.copytree(self.snapshot, clone)
        self.clone = clone

        van_bin = _runner.build_driver(clone, "vanilla",
                                       self.out_dir / "van_build", self.kokkos)
        self._van = _coeffs(van_bin, self.total)

        oracle_tree = self.out_dir / "dd_oracle_tree"
        if oracle_tree.exists():
            shutil.rmtree(oracle_tree)
        oracle_tree.mkdir(parents=True)
        _git_archive(self.dd_repo, self.dd_ref, "src/qcdloop", oracle_tree)
        dd_bin = _runner.build_driver(oracle_tree / "src" / "qcdloop", "dd",
                                      self.out_dir / "dd_build", self.kokkos)
        self._ref = _coeffs(dd_bin, self.total)

    def _group_for(self, integral: str) -> str:
        if integral in self._group_of:
            return self._group_of[integral]
        assert self.clone is not None
        for hdr in sorted((self.clone / "box").glob("B*m.h")):
            text = hdr.read_text(errors="ignore")
            if re.search(rf"\bvoid\s+{re.escape(integral)}\s*\(", text):
                grp = f"box/{hdr.name}"
                self._group_of[integral] = grp
                return grp
        raise RuntimeError(f"no box group header defines {integral} under {self.clone}")

    def _flip(self, group: str, precision: str) -> tuple[dict | None, str]:
        """Emit+compile+measure a per-group flip TU (memoized)."""
        key = (group, precision)
        if key in self._flip_coeffs:
            return self._flip_coeffs[key], self._flip_log.get(key, "")
        assert self.clone is not None
        target = _TARGET_BY_NAME[precision]
        tu = emit_flip_tu(self.clone, group, self.out_dir / "flip_drv", target)
        stem = Path(group).stem
        build_dir = self.out_dir / f"flip_build_{stem}_{precision}"
        build_dir.mkdir(parents=True, exist_ok=True)
        binary = build_dir / f"flip_{stem}_{precision}"
        inc = (f"-I{self.clone} -I{self.clone}/box -I{RECIPES} -I{VEND} "
               f"-I{self.kokkos}/include")
        lib = (f"-L{self.kokkos}/lib -L{self.kokkos}/lib64 "
               f"-lkokkoscore -lkokkoscontainers -ldl")
        r = _bash(f"g++ -std=c++20 -O2 -w {inc} {tu.driver_path} -o {binary} {lib}")
        if r.returncode != 0 or not binary.is_file():
            log = (r.stdout + r.stderr)[-3000:]
            (build_dir.parent / f"flip_build_{stem}_{precision}.log").write_text(log)
            self._flip_coeffs[key] = None
            self._flip_log[key] = log
            return None, log
        coeffs = _coeffs(binary, self.total)
        self._flip_coeffs[key] = coeffs
        self._flip_log[key] = ""
        return coeffs, ""

    # -- provider contract ---------------------------------------------------
    def __call__(self, integral: str, target: str) -> dict:
        self._ensure_setup()
        base_d = _min_digits(self._van, self._ref, integral, self.total)
        if target == "baseline":
            return {"built": True, "baseline_digits": base_d}

        if target not in _TARGET_BY_NAME:
            raise ValueError(f"unknown TU target {target!r}")
        group = self._group_for(integral)
        coeffs, log = self._flip(group, target)
        if coeffs is None:
            return {"built": False, "baseline_digits": base_d,
                    "candidate_digits": None, "log_tail": log}
        cand_d = _min_digits(coeffs, self._ref, integral, self.total)
        return {"built": True, "baseline_digits": base_d,
                "candidate_digits": cand_d, "log_tail": ""}


def make_tu_measure_fn(*, out_dir, kokkos_root, dd_repo, dd_ref, sample_count,
                       snapshot=None):
    """Convenience factory returning a ready ``tu_measure_fn`` callable."""
    return TUMeasureProvider(
        out_dir=Path(out_dir), kokkos_root=Path(kokkos_root),
        dd_repo=Path(dd_repo), dd_ref=dd_ref, sample_count=sample_count,
        snapshot=snapshot)
