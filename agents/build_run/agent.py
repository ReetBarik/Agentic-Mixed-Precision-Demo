"""Build/run stub — deterministic subprocess wrapper, no LLM.

Compiles a micro-driver against Tracked headers (and optionally Kokkos),
runs it, and returns a RunResult.  The real LLM-driven build/run agent will
replace this with the same interface.
"""

from __future__ import annotations

import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from agents.config import PipelineConfig


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
) -> RunResult:
    """Write driver_source to a temp directory, cmake-build, and execute.

    When ``clean_build`` is True (the default), ``work_dir/build/`` is wiped
    before configuring so a stale ``CMakeCache.txt`` from a prior attempt can't
    poison the next configure.  The retry loop relies on this.
    """

    if work_dir is None:
        _tmp = tempfile.mkdtemp(prefix="micro_driver_")
        work_dir = Path(_tmp)

    src_dir = work_dir / "src"
    src_dir.mkdir(parents=True, exist_ok=True)

    driver_cpp = src_dir / "micro_driver.cpp"
    driver_cpp.write_text(driver_source, encoding="utf-8")

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

    configure_result = subprocess.run(
        configure_cmd,
        cwd=str(build_dir),
        capture_output=True,
        text=True,
    )
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
    build_result = subprocess.run(
        ["cmake", "--build", ".", "-j"],
        cwd=str(build_dir),
        capture_output=True,
        text=True,
    )
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
    exe = build_dir / "micro_driver"
    run_result = subprocess.run(
        [str(exe)],
        cwd=str(work_dir),
        capture_output=True,
        text=True,
    )

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
