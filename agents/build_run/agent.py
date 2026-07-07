"""Build/run stub — deterministic subprocess wrapper, no LLM.

Compiles a micro-driver against Tracked headers (and optionally Kokkos),
runs it, and returns a RunResult.  The real LLM-driven build/run agent will
replace this with the same interface.
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agents.config import PipelineConfig

# TODO(build-run-agent): these are HPC-cluster-specific plumbing values.
# When build_and_run becomes LLM-driven, environment/toolchain detection
# should be part of the agent's responsibility, not hardcoded here.
DEFAULT_CLUSTER_MODULES = ["gcc/13.3.0", "cmake/3.28.3"]
DEFAULT_MODULE_USE_PATH = "/soft/modulefiles"


def _module_settings() -> tuple[list[str], str]:
    """Resolve the module list + `module use` path, honoring env overrides.

    ``PIPELINE_MODULE_LIST`` is colon-separated (e.g. "gcc/13.3.0:cmake/3.28.3").
    Setting it to the empty string disables module loading entirely (useful for
    CI / non-cluster hosts where ``module`` isn't defined).
    """
    raw = os.environ.get("PIPELINE_MODULE_LIST")
    if raw is None:
        modules = list(DEFAULT_CLUSTER_MODULES)
    else:
        modules = [m for m in raw.split(":") if m]
    use_path = os.environ.get("PIPELINE_MODULE_USE_PATH", DEFAULT_MODULE_USE_PATH)
    return modules, use_path


def _run_build_step(cmd: list[str], cwd: str) -> subprocess.CompletedProcess:
    """Run one build/run subprocess with the HPC module chain sourced.

    Every cmake/make/ctest (and the built binary) is wrapped in
    ``bash -lc 'module use <path> && module load <m...> && <cmd>'`` so the
    toolchain and its runtime libraries are on PATH/LD_LIBRARY_PATH.  The module
    state lives only in the subprocess — it is never exported into this Python
    process.  When no modules are configured (``PIPELINE_MODULE_LIST=""``) the
    command runs directly, unwrapped.
    """
    modules, use_path = _module_settings()
    if not modules:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)

    inner = " ".join(shlex.quote(part) for part in cmd)
    prelude = f"module use {shlex.quote(use_path)} && module load {' '.join(shlex.quote(m) for m in modules)}"
    wrapped = ["bash", "-lc", f"{prelude} && {inner}"]
    return subprocess.run(wrapped, cwd=cwd, capture_output=True, text=True)


@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    journal_path: Path | None   # None if run failed before the flush
    work_dir: Path              # kept for debugging
    # Which stage the result reflects.  "ok" = ran cleanly and produced a
    # journal; "run" = binary ran but exited non-zero (or no journal).
    # Retry discriminator: phase in {"configure", "build"}.
    phase: Literal["configure", "build", "run", "ok"] = "ok"


def build_and_run(
    driver_source: str,
    framework: str,
    cfg: PipelineConfig,
    work_dir: Path | None = None,
    clean_build: bool = True,
    use_tracked: bool = False,
    target_library_headers: Path | None = None,
    existing_shim: Path | None = None,
) -> RunResult:
    """Write driver_source to a temp directory, cmake-build, and execute.

    When ``clean_build`` is True (the default), ``work_dir/build/`` is wiped
    before configuring so a stale ``CMakeCache.txt`` from a prior attempt can't
    poison the next configure.  The retry loop relies on this.

    When ``use_tracked`` is True and ``target_library_headers`` is provided, the
    tracked-integrator shared service runs first: it (re)generates the
    ``<app>_interop.hpp`` interop shim that makes the target library callable
    with ``Tracked<T>``, unless an up-to-date shim already exists (SOURCE_HASH
    match).  This keeps the "does this target need Tracked?" decision in one
    place — any caller that opts in benefits automatically (task revision #2,
    option (a)).  With ``use_tracked`` False (the default) the flow is exactly
    the pre-existing compile/run path, so current characterizer callers are
    unaffected.
    """

    if work_dir is None:
        _tmp = tempfile.mkdtemp(prefix="micro_driver_")
        work_dir = Path(_tmp)

    src_dir = work_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    driver_cpp = src_dir / "micro_driver.cpp"
    driver_cpp.write_text(driver_source, encoding="utf-8")

    # Prerequisite: ensure a Tracked interop shim exists for the target library
    # before compiling.  Imported locally so build_run stays import-light (the
    # integrator pulls in the LLM client in Part 2) and to avoid any cycle.
    if use_tracked and target_library_headers is not None:
        from agents.tracked_integrator import agent as tracked_integrator

        tracked_integrator.integrate(
            target_library_headers=target_library_headers,
            driver_source_path=driver_cpp,
            tracked_repo_path=cfg.tracked_root,
            existing_shim=existing_shim,
            cfg=cfg,
        )

    cmake_content = _render_cmake(framework, cfg)
    (work_dir / "CMakeLists.txt").write_text(cmake_content, encoding="utf-8")

    build_dir = work_dir / "build"
    if clean_build and build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir(exist_ok=True)

    # --- Configure ---
    configure_cmd = [
        "cmake", "..",
        f"-DCMAKE_BUILD_TYPE=Release",
    ]
    if framework == "kokkos-serial" and cfg.kokkos_root:
        configure_cmd.append(f"-DCMAKE_PREFIX_PATH={cfg.kokkos_root}")

    configure_result = _run_build_step(configure_cmd, cwd=str(build_dir))
    if configure_result.returncode != 0:
        return RunResult(
            returncode=configure_result.returncode,
            stdout=configure_result.stdout,
            stderr=configure_result.stderr,
            journal_path=None,
            work_dir=work_dir,
            phase="configure",
        )

    # --- Build ---
    build_result = _run_build_step(["cmake", "--build", ".", "-j"], cwd=str(build_dir))
    if build_result.returncode != 0:
        combined_stderr = configure_result.stderr + "\n" + build_result.stderr
        return RunResult(
            returncode=build_result.returncode,
            stdout=build_result.stdout,
            stderr=combined_stderr,
            journal_path=None,
            work_dir=work_dir,
            phase="build",
        )

    # --- Run ---
    # Wrapped in the module chain too: the binary is linked against the module
    # gcc's libstdc++, so it needs that lib on LD_LIBRARY_PATH at runtime.
    exe = build_dir / "micro_driver"
    run_result = _run_build_step([str(exe)], cwd=str(work_dir))

    # Tracked flushes a JSONL whose name is passed via journal::flush("<name>.jsonl").
    # The driver is generated to write to work_dir/journal.jsonl.
    journal_path = work_dir / "journal.jsonl"
    if not journal_path.exists():
        journal_path = None

    # "ok" only when the binary exited cleanly AND produced a journal; any
    # other runtime outcome is "run" (not retried — see PLAN_retry_loop.md).
    phase = "ok" if (run_result.returncode == 0 and journal_path is not None) else "run"

    return RunResult(
        returncode=run_result.returncode,
        stdout=run_result.stdout,
        stderr=run_result.stderr,
        journal_path=journal_path,
        work_dir=work_dir,
        phase=phase,
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _render_cmake(framework: str, cfg: PipelineConfig) -> str:
    template_path = Path(__file__).parent / "cmake_template.cmake"
    template = template_path.read_text(encoding="utf-8")

    tracked_include = ""
    if cfg.tracked_root:
        tracked_include = str(cfg.tracked_root / "include")

    find_package_lines = ""
    link_libs = ""
    extra_include_dirs = ""
    if framework == "kokkos-serial":
        find_package_lines = "find_package(Kokkos REQUIRED)"
        link_libs = "Kokkos::kokkos"
        if cfg.kokkos_root:
            extra_include_dirs = str(cfg.kokkos_root / "include")

    return template.format(
        cxx_standard=cfg.cxx_standard,
        find_package_lines=find_package_lines,
        tracked_include_dir=tracked_include,
        extra_include_dirs=extra_include_dirs,
        link_libs=link_libs,
    )
