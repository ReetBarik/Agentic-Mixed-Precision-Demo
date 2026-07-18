"""Unit + integration tests for dd_integrator.integrate_region (regional DD).

Structural twin of tests/ff_integrator/test_regional.py; the DD-specific check is
the hex-encoded (hi, lo) constant-table instruction in the user turn / ruleset.
The whole-app integrate() stub is exercised by tests/... (unchanged) — not here.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from agents.dd_integrator import agent as dd

_FILE = (
    "#pragma once\n"
    "double g(double a, double b) {\n"
    "    double r = a * b;\n"
    "    return r;\n"
    "}\n"
)


def _git(root, *args):
    return subprocess.run(["git", "-C", str(root), *args],
                          capture_output=True, text=True, check=True)


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "cand"
    root.mkdir()
    (root / "kernel.h").write_text(_FILE)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t.t")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    sha = _git(root, "rev-parse", "HEAD").stdout.strip()
    return root, sha


class _CannedLLM:
    def __init__(self):
        self.calls = []

    def __call__(self, system, user, attempt):
        self.calls.append((system, user, attempt))
        return (
            "#pragma once\n"
            "// SOURCE_HASH: PENDING\n"
            "#include <dd_math.hpp>\n"
            "#include <dd_complex.hpp>\n"
            "// Rule 2: region computes in ddouble; vendored ops suffice\n"
        )


def _call(repo, out_dir, llm, attempt=0):
    root, sha = repo
    return dd.integrate_region(
        file="kernel.h", line_start=3, line_end=3, variables=["a", "b"],
        working_tree=sha, out_dir=out_dir, attempt=attempt,
        repo_path=str(root), llm_fn=llm,
    )


def test_generates_dd_shim_and_boundary(repo, tmp_path):
    root, _ = repo
    out = tmp_path / "shims"
    res = _call(repo, out, _CannedLLM())

    assert res.ok
    shim = Path(res.shim_paths[0])
    assert shim.name.startswith("kernel_dd_") and shim.name.endswith(".h")
    assert "#include <dd_math.hpp>" in shim.read_text()
    assert (root / shim.name).exists()

    patch = res.boundary_patch
    assert "quad::ddfun::ddouble a__ff = quad::ddfun::ddouble(a);" in patch
    assert "quad::ddfun::ddouble r__ext = a__ff * b__ff;" in patch
    assert "double r = static_cast<double>(r__ext.hi) + static_cast<double>(r__ext.lo);" in patch


def test_user_message_carries_hex_constant_note(repo, tmp_path):
    llm = _CannedLLM()
    _call(repo, tmp_path / "shims", llm)
    user = llm.calls[0][1]
    assert "make_dd(0x" in user
    assert "quad::ddfun::ddouble" in user
    assert "quad::ddfun::ddcomplex" in user


def test_cache_hit_skips_llm(repo, tmp_path):
    out = tmp_path / "shims"
    llm = _CannedLLM()
    _call(repo, out, llm, attempt=0)
    _call(repo, out, llm, attempt=0)
    assert len(llm.calls) == 1


def test_retry_bypasses_cache(repo, tmp_path):
    out = tmp_path / "shims"
    llm = _CannedLLM()
    _call(repo, out, llm, attempt=0)
    _call(repo, out, llm, attempt=1)
    assert len(llm.calls) == 2
    assert "regeneration attempt 1" in llm.calls[1][1]


def test_ruleset_forbids_decimal_constants():
    # The DD ruleset must codify the hex-encoding requirement (Rule R3).
    assert "make_dd(0x" in dd._SYSTEM_PROMPT
    assert "decimal literal" in dd._SYSTEM_PROMPT


# --------------------------------------------------------------------------- #
# Integration: a real LLM call (skipped without the Argo proxy).
# --------------------------------------------------------------------------- #

@pytest.mark.llm
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_AUTH_TOKEN"),
                    reason="requires the Argo LLM proxy")
def test_real_llm_generates_parseable_dd_shim(repo, tmp_path):
    res = _call(repo, tmp_path / "shims", None)
    assert res.ok, res.error
    shim = Path(res.shim_paths[0]).read_text()
    assert "#pragma once" in shim
    assert "dd_math.hpp" in shim
    assert res.boundary_patch and "quad::ddfun::ddouble" in res.boundary_patch
    assert res.llm_tokens > 0
