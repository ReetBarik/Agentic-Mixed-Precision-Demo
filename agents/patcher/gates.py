"""P5 build + runtime gates — the only thing the Patcher runs to prove a candidate.

The Patcher builds **only the vanilla driver** (``runs/qcdloop/src/boxGPU_vanilla.cpp``
via the app CMake project) against the candidate's header tree, then runs a tiny
deterministic smoke test — 1 sample per integral (21 rows), ``srand(12345)`` — and
checks three things (design §P5):

* it compiles (else ``build_failed``);
* it runs to completion emitting ≥ ``expected_rows`` result rows (else
  ``runtime_crashed``);
* no coefficient is NaN/Inf (else ``runtime_nan``).

No DD build, no full run, no divergence check — numerical correctness over
meaningful statistics is the Validator's judgment.  Timeouts (build 5 min, smoke
30 s) catch runaway loops and surface as ``timeout``.  The HPC module chain is
sourced via the same env-overridable wrapper as ``build_and_run``
(``PIPELINE_MODULE_LIST`` / ``PIPELINE_MODULE_USE_PATH``); set
``PIPELINE_MODULE_LIST=""`` to build unwrapped on a non-cluster host.
"""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from agents.build_run.agent import _module_settings
from agents.integrator_base.cache import hash_header_dir
from agents.patcher import result as R

# Source extensions scanned for variant reference-counting (the fan-out wiring gate).
_SRC_EXTS = (".h", ".hpp", ".hh", ".hxx", ".cuh", ".inl", ".ipp", ".cpp", ".cc", ".cxx")
_IDENT_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")

_REPO = Path(__file__).resolve().parents[2]
DEFAULT_APP_CMAKE_DIR = _REPO / "runs" / "qcdloop" / "app"
DEFAULT_KOKKOS_ROOT = Path.home() / "kokkos-install"

EXPECTED_ROWS = 21
BUILD_TIMEOUT_SEC = 300
SMOKE_TIMEOUT_SEC = 30

_NAN_RE = re.compile(r"\b(nan|inf)\b", re.IGNORECASE)


@dataclass
class GateResult:
    status: str                      # R.OK | R.BUILD_FAILED | R.RUNTIME_* | R.TIMEOUT
    err_kind: str | None = None
    detail: str | None = None
    build_log_path: Path | None = None
    runtime_log_path: Path | None = None
    # Build-fuse (CALIBRATION.md §Bug 5): on OK the gate binary + a content hash of
    # the header tree it was built against, so the Validator can reuse this binary
    # for its candidate run instead of rebuilding the same monolithic TU.
    binary_path: Path | None = None
    tree_hash: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == R.OK


def _wrap(cmd: str) -> list[str]:
    """Wrap ``cmd`` in the module-sourcing login shell (or run it bare)."""
    modules, use_path = _module_settings()
    if not modules:
        return ["bash", "-lc", cmd]
    prelude = f"module use {use_path} && module load {' '.join(modules)}"
    return ["bash", "-lc", f"{prelude} && {cmd}"]


def run_gate(headers_dir: Path, build_dir: Path, logs_dir: Path, iter_id,
             *, app_cmake_dir: Path = DEFAULT_APP_CMAKE_DIR,
             kokkos_root: Path = DEFAULT_KOKKOS_ROOT,
             expected_rows: int = EXPECTED_ROWS,
             build_timeout: int = BUILD_TIMEOUT_SEC,
             smoke_timeout: int = SMOKE_TIMEOUT_SEC) -> GateResult:
    """Build the vanilla driver against ``headers_dir`` and smoke-run it."""
    headers_dir = Path(headers_dir).resolve()
    build_dir = Path(build_dir).resolve()
    logs_dir = Path(logs_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    build_log = logs_dir / f"iter_{iter_id}_build.log"
    runtime_log = logs_dir / f"iter_{iter_id}_runtime.log"

    # ---- build (configure + compile) ----
    cfg_cmd = (f"cmake -S {app_cmake_dir} -B {build_dir} "
               f"-DCMAKE_PREFIX_PATH={Path(kokkos_root).resolve()} "
               f"-DQL_MODE=vanilla -DQL_HEADERS={headers_dir} "
               f"-DCMAKE_BUILD_TYPE=Release")
    build_cmd = f"cmake --build {build_dir} -j"
    try:
        cfg = subprocess.run(_wrap(cfg_cmd), capture_output=True, text=True,
                             timeout=build_timeout)
        bld = subprocess.run(_wrap(build_cmd), capture_output=True, text=True,
                             timeout=build_timeout)
    except subprocess.TimeoutExpired as exc:
        build_log.write_text(f"BUILD TIMEOUT after {build_timeout}s\n{exc}")
        return GateResult(R.TIMEOUT, R.ERR_TIMEOUT,
                          f"build exceeded {build_timeout}s", build_log)

    build_log.write_text(
        f"$ {cfg_cmd}\n{cfg.stdout}\n{cfg.stderr}\n"
        f"$ {build_cmd}\n{bld.stdout}\n{bld.stderr}\n")
    if cfg.returncode != 0 or bld.returncode != 0:
        return GateResult(R.BUILD_FAILED, R.ERR_COMPILE,
                          "cmake configure/build failed", build_log)

    binary = build_dir / "boxGPU_app"
    if not binary.is_file():
        return GateResult(R.BUILD_FAILED, R.ERR_COMPILE,
                          f"no binary produced at {binary}", build_log)

    # ---- smoke run (1 sample per integral) ----
    smoke_cmd = f"{binary} --sample-count 1 --sample-offset 0"
    try:
        run = subprocess.run(_wrap(smoke_cmd), capture_output=True, text=True,
                             timeout=smoke_timeout)
    except subprocess.TimeoutExpired as exc:
        runtime_log.write_text(f"SMOKE TIMEOUT after {smoke_timeout}s\n{exc}")
        return GateResult(R.TIMEOUT, R.ERR_TIMEOUT,
                          f"smoke run exceeded {smoke_timeout}s",
                          build_log, runtime_log)

    runtime_log.write_text(f"$ {smoke_cmd}\n{run.stdout}\n---stderr---\n{run.stderr}\n")
    result = _scan_smoke(run, expected_rows, build_log, runtime_log)
    if result.ok:
        # Publish the binary + the hash of the tree it was built against so the
        # Validator can reuse it (the candidate tree it would build is byte-identical
        # to this one — same header set + shims — so its hash_header_dir matches).
        result.binary_path = binary
        result.tree_hash = hash_header_dir(headers_dir)
    return result


def _scan_smoke(run: subprocess.CompletedProcess, expected_rows: int,
                build_log: Path, runtime_log: Path) -> GateResult:
    """Classify a completed smoke run into ok / nan / crashed."""
    res_lines = [ln for ln in run.stdout.splitlines() if ln.startswith("RES,")]
    if run.returncode != 0:
        return GateResult(R.RUNTIME_CRASHED, R.ERR_CRASH,
                          f"driver exited {run.returncode}", build_log, runtime_log)
    for ln in res_lines:
        # only inspect the coefficient columns (skip "RES,<integral>,<idx>")
        parts = ln.split(",")
        if _NAN_RE.search(",".join(parts[3:])):
            return GateResult(R.RUNTIME_NAN, R.ERR_NAN,
                              "NaN/Inf in coefficient output", build_log, runtime_log)
    if len(res_lines) < expected_rows:
        return GateResult(R.RUNTIME_CRASHED, R.ERR_CRASH,
                          f"only {len(res_lines)} result rows (<{expected_rows})",
                          build_log, runtime_log)
    return GateResult(R.OK, None, None, build_log, runtime_log)


# ===========================================================================
# Phase 2a — variant-wiring gate (rename-cascade guard) + nm telemetry
# ===========================================================================
#
# The fan-out (:mod:`agents.patcher.fanout`) turns a region intent into per-caller-
# path function *variants* and reroutes the call chain up to the entry point.  A
# rename-cascade bug leaves a variant DEFINED but never CALLED (its parent still
# calls the original), which compiles fine (both exist) yet silently bypasses the
# precision change — the very "shim exists but nothing routes to it" failure fan-out
# is meant to eliminate, one level up.
#
# The reliable detector is source-level, not ``nm``.  qcdloop's variants are
# ``KOKKOS_INLINE_FUNCTION`` *templates*; once instantiated they are usually inlined
# away, so a correctly-wired variant legitimately produces NO defined symbol — an
# ``nm``-only check would false-fail on every real variant.  So the HARD gate is
# "every declared variant is referenced (called) somewhere in the tree source"
# (an orphan variant is defined but appears only at its own definition site), and
# ``nm`` is kept as *supplementary telemetry* (which variants did emit a symbol, and
# what the candidate binary carries that the baseline does not).


def _tree_sources(tree_root: Path) -> list[Path]:
    return [p for p in sorted(Path(tree_root).rglob("*"))
            if p.suffix.lower() in _SRC_EXTS and p.is_file() and ".git" not in p.parts]


def _callable_occurrences(text: str, name: str) -> int:
    """Count whole-word ``name`` tokens immediately followed by ``(`` or ``<``.

    Comment/string-agnostic *over-count* is fine here — we only compare against the
    single guaranteed definition-site occurrence, and any extra (a call) proves the
    variant is referenced.  Uses a cheap whole-word regex; the definition's
    ``name(`` param list always contributes exactly one.
    """
    count = 0
    for m in re.finditer(r"(?<![\w])" + re.escape(name) + r"\s*[<(]", text):
        # guard the left boundary: the char before must not be an identifier char
        start = m.start()
        if start > 0 and text[start - 1] in _IDENT_CHARS:
            continue
        count += 1
    return count


def referenced_variants(tree_root: Path, declared: list[str]) -> set[str]:
    """Subset of ``declared`` variants that are *called* somewhere in the tree.

    A declared variant is "referenced" if its callable-occurrence count across all
    tree sources exceeds one (one occurrence is its own definition signature; two or
    more means at least one call site).
    """
    if not declared:
        return set()
    counts = {v: 0 for v in declared}
    for src in _tree_sources(tree_root):
        try:
            text = src.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for v in declared:
            if v in text:                       # cheap prefilter
                counts[v] += _callable_occurrences(text, v)
    return {v for v, c in counts.items() if c >= 2}


def check_variant_wiring(tree_root: Path, declared: list[str]) -> str | None:
    """HARD rename-cascade guard: return an error detail if any declared variant is
    orphaned (defined but never called), else ``None``.

    Robust to inlining (source-level).  The returned detail is prefixed
    ``rename_cascade_incomplete:`` so the manifest can classify the new failure mode.
    """
    if not declared:
        return None
    referenced = referenced_variants(tree_root, declared)
    orphans = sorted(set(declared) - referenced)
    if orphans:
        return (f"rename_cascade_incomplete: {len(orphans)} declared variant(s) are "
                f"defined but never called (orphaned by a broken reroute): {orphans}. "
                f"A caller was left pointing at the original, silently bypassing the "
                f"precision change.")
    return None


def check_no_silent_bypass(tree_root: Path, reroutes: dict[str, str]) -> str | None:
    """Assert each expected reroute actually landed: the variant is referenced and,
    for each ``original -> variant`` pair, the variant's call count is non-zero.

    ``reroutes`` maps an original callee name to the variant it should now route to
    (collected from the fan-out manifest / result).  Returns an error detail
    (prefixed ``silent_bypass:``) if a variant that should be called is not, else
    ``None``.  Complements :func:`check_variant_wiring` (which guards orphans) by
    naming the specific reroute that failed.
    """
    if not reroutes:
        return None
    variants = sorted({v for v in reroutes.values()})
    referenced = referenced_variants(tree_root, variants)
    missed = [f"{orig}->{var}" for orig, var in sorted(reroutes.items())
              if var not in referenced]
    if missed:
        return (f"silent_bypass: expected reroute(s) not wired in — {missed}; a caller "
                f"still calls the original instead of the variant.")
    return None


def nm_defined_symbols(*paths: Path, demangle: bool = True) -> set[str]:
    """Defined symbols in the given object/binary files via ``nm --defined-only``.

    Best-effort telemetry: returns the raw (optionally C++-demangled) symbol name
    strings.  Any ``nm`` failure (missing tool, unreadable file) yields an empty set
    rather than raising — the variant-wiring gate does not depend on this.
    """
    flags = ["--defined-only"]
    if demangle:
        flags.append("-C")
    syms: set[str] = set()
    for p in paths:
        p = Path(p)
        if not p.is_file():
            continue
        try:
            r = subprocess.run(["nm", *flags, str(p)],
                               capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError):
            continue
        for line in r.stdout.splitlines():
            parts = line.split(" ", 2)
            if len(parts) == 3 and parts[1] in "TtWwVvDdBbRr":
                syms.add(parts[2].strip())
    return syms


def variant_symbols_present(binary: Path, declared: list[str]) -> tuple[list[str], list[str]]:
    """Split ``declared`` into (present, absent) by substring match against the
    binary's defined (demangled) symbols.

    Telemetry only — an *absent* variant is expected for an inlined template and is
    NOT a failure (the hard guard is :func:`check_variant_wiring`).  ``present`` is
    the honest "the pipeline produced real, emitted new code" signal the Phase-2a
    sizing asks for.
    """
    syms = nm_defined_symbols(binary)
    blob = "\n".join(syms)
    present = [v for v in declared if v in blob]
    absent = [v for v in declared if v not in blob]
    return present, absent
