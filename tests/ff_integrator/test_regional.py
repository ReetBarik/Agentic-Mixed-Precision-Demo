"""Unit + integration tests for ff_integrator.integrate_region (regional ff)."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from agents.ff_integrator import agent as ff

_FILE = (
    "#pragma once\n"
    "double f(double a, double b) {\n"
    "    double r = a + b;\n"
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
    """Records (system, user, attempt) calls; returns a fixed shim body."""

    def __init__(self):
        self.calls = []

    def __call__(self, system, user, attempt):
        self.calls.append((system, user, attempt))
        return (
            "#pragma once\n"
            "// SOURCE_HASH: PENDING\n"
            "#include <ff_math.hpp>\n"
            "#include <ff_complex.hpp>\n"
            "// Rule 2: region computes in ffloat; vendored ops suffice\n"
        )


def _call(repo, out_dir, llm, attempt=0):
    root, sha = repo
    return ff.integrate_region(
        file="kernel.h", line_start=3, line_end=3, variables=["a", "b"],
        working_tree=sha, out_dir=out_dir, attempt=attempt,
        repo_path=str(root), llm_fn=llm,
    )


def test_generates_shim_and_boundary(repo, tmp_path):
    root, _ = repo
    out = tmp_path / "shims"
    res = _call(repo, out, _CannedLLM())

    assert res.ok
    # shim written to out_dir, named <stem>_ff_<key8>.h, with a stamped hash
    assert len(res.shim_paths) == 1
    shim = Path(res.shim_paths[0])
    assert shim.exists() and shim.parent == out
    assert shim.name.startswith("kernel_ff_") and shim.name.endswith(".h")
    text = shim.read_text()
    assert "SOURCE_HASH: PENDING" not in text
    assert "#include <ff_math.hpp>" in text
    # Wave-3 dedup: the per-family canonical shim is installed in the tree (the
    # per-region file stays an out_dir artifact), and the boundary #includes it.
    canonical = root / "ql_shim_ff.h"
    assert canonical.exists()
    assert "#include <ff_math.hpp>" in canonical.read_text()

    # boundary patch: promote reads on entry, retype/rename local, demote on exit
    patch = res.boundary_patch
    assert patch is not None
    assert "quad::ffun::ffloat a__ff = quad::ffun::ffloat(a);" in patch
    assert "quad::ffun::ffloat b__ff = quad::ffun::ffloat(b);" in patch
    assert "quad::ffun::ffloat r__ext = a__ff + b__ff;" in patch
    assert "double r = static_cast<double>(r__ext.hi) + static_cast<double>(r__ext.lo);" in patch
    assert '#include "ql_shim_ff.h"' in patch


def test_cache_hit_skips_llm(repo, tmp_path):
    out = tmp_path / "shims"
    llm = _CannedLLM()
    first = _call(repo, out, llm, attempt=0)
    second = _call(repo, out, llm, attempt=0)
    assert first.ok and second.ok
    assert first.shim_paths == second.shim_paths     # same cache_key → same file
    assert len(llm.calls) == 1                        # second call was a cache hit
    assert second.llm_tokens == 0                     # cache hit reports no tokens


def test_retry_bypasses_cache_and_varies_message(repo, tmp_path):
    out = tmp_path / "shims"
    llm = _CannedLLM()
    _call(repo, out, llm, attempt=0)
    _call(repo, out, llm, attempt=1)
    # attempt>0 bypasses the cache → llm invoked again
    assert len(llm.calls) == 2
    msg0 = llm.calls[0][1]
    msg1 = llm.calls[1][1]
    assert msg0 != msg1
    assert "regeneration attempt 1" in msg1
    assert llm.calls[1][2] == 1                        # attempt forwarded


def test_scalar_change_changes_cache_key(repo, tmp_path):
    # different scalar spelling in the spec would change the key; here we assert
    # the ff key differs from a dd key over the same region (guards the tag).
    from agents.integrator_base import cache
    region = "    double r = a + b;"
    k_ff = cache.compute_region_hash(region, ff._SYSTEM_PROMPT, "quad::ffun::ffloat", [])
    k_dd = cache.compute_region_hash(region, ff._SYSTEM_PROMPT, "quad::ddfun::ddouble", [])
    assert k_ff != k_dd


def test_bad_region_range_returns_failed(repo, tmp_path):
    root, sha = repo
    res = ff.integrate_region(
        file="kernel.h", line_start=99, line_end=100, variables=[],
        working_tree=sha, out_dir=tmp_path / "s", attempt=0,
        repo_path=str(root), llm_fn=_CannedLLM(),
    )
    assert not res.ok
    assert "out of range" in (res.error or "")


def test_integrate_whole_app_not_implemented():
    with pytest.raises(NotImplementedError):
        ff.integrate("headers", "driver")


# --------------------------------------------------------------------------- #
# Integration: a real LLM call (skipped without the Argo proxy).
# --------------------------------------------------------------------------- #

@pytest.mark.llm
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_AUTH_TOKEN"),
                    reason="requires the Argo LLM proxy")
def test_real_llm_generates_parseable_shim(repo, tmp_path):
    res = _call(repo, tmp_path / "shims", None)   # llm_fn=None → real call
    assert res.ok, res.error
    shim = Path(res.shim_paths[0]).read_text()
    assert "#pragma once" in shim
    assert "ff_math.hpp" in shim
    assert res.boundary_patch and "quad::ffun::ffloat" in res.boundary_patch
    assert res.llm_tokens > 0
