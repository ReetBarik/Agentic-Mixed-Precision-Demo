"""Unit + integration tests for chain_integrator.integrate_region (Phase 2f).

Structural twin of tests/dd_integrator/test_regional.py — same shared engine,
same ql_shim_dd.h family shim — plus the C9 chain-boundary rule assertions.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from agents.chain_integrator import agent as chain

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
            "// Rule 2 / C9: chain-internal value stays DoubleDouble\n"
        )


def _call(repo, out_dir, llm, attempt=0):
    root, sha = repo
    return chain.integrate_region(
        file="kernel.h", line_start=3, line_end=3, variables=["a", "b"],
        working_tree=sha, out_dir=out_dir, attempt=attempt,
        repo_path=str(root), llm_fn=llm,
    )


def test_generates_dd_shim_and_boundary_into_family_shim(repo, tmp_path):
    root, _ = repo
    res = _call(repo, tmp_path / "shims", _CannedLLM())
    assert res.ok
    shim = Path(res.shim_paths[0])
    assert shim.name.startswith("kernel_dd_") and shim.name.endswith(".h")
    # merges into the SAME canonical dd family shim as dd_integrator (shim_prefix=dd)
    canonical = root / "ql_shim_dd.h"
    assert canonical.exists()
    assert '#include "ql_shim_dd.h"' in res.boundary_patch
    patch = res.boundary_patch
    assert "Kokkos::Experimental::DoubleDouble a__ff = Kokkos::Experimental::DoubleDouble(a);" in patch
    assert "Kokkos::Experimental::DoubleDouble r__ext = a__ff * b__ff;" in patch


def test_spec_targets_double_double():
    assert chain.SPEC.cpp_scalar == "Kokkos::Experimental::DoubleDouble"
    assert chain.SPEC.cpp_complex == "Kokkos::Experimental::DoubleDoubleComplex"
    assert chain.SPEC.shim_prefix == "dd"          # shares ql_shim_dd.h
    assert chain.SPEC.two_limb is True             # extended scalar


def test_ruleset_carries_c9_chain_boundary_rule():
    # The distinguishing rule of the chain integrator.
    assert "C9" in chain._SYSTEM_PROMPT
    assert "chain-boundary" in chain._SYSTEM_PROMPT.lower()
    assert "chain-internal" in chain._SYSTEM_PROMPT.lower()
    # still carries the inherited dd discipline (R3 hex constants, C1 include set)
    assert "DoubleDouble::from_bits(0x" in chain._SYSTEM_PROMPT
    assert "C1." in chain._SYSTEM_PROMPT and "app-source" in chain._SYSTEM_PROMPT


def test_user_message_carries_hex_and_c9_notes(repo, tmp_path):
    llm = _CannedLLM()
    _call(repo, tmp_path / "shims", llm)
    user = llm.calls[0][1]
    assert "DoubleDouble::from_bits(0x" in user
    assert "Kokkos::Experimental::DoubleDouble" in user
    assert "C9" in user or "Chain-boundary" in user


def test_cache_hit_skips_llm(repo, tmp_path):
    out = tmp_path / "shims"
    llm = _CannedLLM()
    _call(repo, out, llm, attempt=0)
    _call(repo, out, llm, attempt=0)
    assert len(llm.calls) == 1


def test_integrate_whole_app_not_implemented():
    with pytest.raises(NotImplementedError):
        chain.integrate("headers", "driver")


class _BadIncludeLLM:
    def __init__(self):
        self.calls = []

    def __call__(self, system, user, attempt):
        self.calls.append((system, user, attempt))
        return (
            "#pragma once\n"
            "// SOURCE_HASH: PENDING\n"
            "#include <dd_math.hpp>\n"
            '#include "ql/constants.h"\n'
            "// Rule 5\n"
        )


def test_forbidden_include_is_rejected_as_misgen(repo, tmp_path):
    res = _call(repo, tmp_path / "shims", _BadIncludeLLM())
    assert not res.ok and res.status == "llm_failed"
    assert "C1 include lint" in (res.error or "")


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
    assert res.boundary_patch and "Kokkos::Experimental::DoubleDouble" in res.boundary_patch
    assert res.llm_tokens > 0
