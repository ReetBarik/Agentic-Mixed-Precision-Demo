"""Unit tests for the build_run per-line (`line=`) injection plumbing.

The full generate→apply→rebuild→reset flow needs a real compile (validated
end-to-end for qcdloop in runs/qcdloop/VALIDATION.md); here we cover the two new
pure helpers — the companion patch path and the module-wrapped gcc include-search
parse — plus the guard that ``inject_line_scopes`` defaults off so existing
callers are unaffected.
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace

from agents.build_run import agent as br


def test_companion_lines_patch_path():
    shim = Path("/x/runs/qcdloop/src/ql_tracked_interop.hpp")
    assert br._companion_lines_patch_path(shim).name == "ql_tracked_lines.patch"
    # falls back to stem for a non-standard shim name
    assert br._companion_lines_patch_path(Path("/x/foo.hpp")).name == "foo_lines.patch"


def test_module_gcc_search_dirs_parses_search_block(monkeypatch, tmp_path):
    # two real dirs (kept) and one bogus dir (dropped: not on disk)
    d1 = tmp_path / "inc1"; d1.mkdir()
    d2 = tmp_path / "c++" / "13"; d2.mkdir(parents=True)
    fake_v = (
        "ignored preamble\n"
        "#include <...> search starts here:\n"
        f" {d1}\n"
        f" {d2} (framework directory)\n"
        " /does/not/exist\n"
        "End of search list.\n"
        "trailing ignored\n"
    )
    monkeypatch.setattr(br, "_run_build_step",
                        lambda cmd, cwd: SimpleNamespace(stdout="", stderr=fake_v, returncode=0))
    dirs = br._module_gcc_search_dirs()
    assert dirs == [str(d1), str(d2)]          # bogus dir filtered, framework suffix stripped


def test_inject_line_scopes_defaults_off():
    sig = inspect.signature(br.build_and_run)
    assert sig.parameters["inject_line_scopes"].default is False
