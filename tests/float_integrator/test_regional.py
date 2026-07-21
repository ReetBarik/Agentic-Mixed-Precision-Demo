"""Unit + integration tests for float_integrator.integrate_region (regional float).

Wave 2: the first extension of the regional integrator ruleset to a NEW target
type.  Float is a *native* single-limb scalar, so the load-bearing differences
from the ff/dd twins are:

* the boundary patch widens region writes with a PLAIN cast (``static_cast<T>(w)``)
  — NOT two-limb ``.hi``/``.lo`` reconstruction (a plain ``float`` has no limbs);
* the shim uses builtin ``float`` / ``std::complex<float>`` (no vendored header,
  no ``make_ff``);
* the escape hatch (Rule 6 / R4): an un-classifiable float-vs-double narrowing is
  surfaced as a hard-build-failure ``#error`` rather than a silent slip.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from agents.float_integrator import agent as fl

# A template-typed region (no bare `double` token) — exactly the surface the
# plain-edit float rung cannot touch, so the regional path owns it.
_FILE = (
    "#pragma once\n"
    "template <class T> T f(T a, T b) {\n"
    "    T r = a * b;\n"
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
    """Returns a fixed float shim body; records calls."""

    def __init__(self, body=None):
        self.calls = []
        self._body = body or (
            "#pragma once\n"
            "// SOURCE_HASH: PENDING\n"
            "// Rule 2: region computes in float; builtin float ops suffice\n"
        )

    def __call__(self, system, user, attempt):
        self.calls.append((system, user, attempt))
        return self._body


def _call(repo, out_dir, llm, attempt=0, line_start=3, line_end=3, variables=("a", "b")):
    root, sha = repo
    return fl.integrate_region(
        file="kernel.h", line_start=line_start, line_end=line_end,
        variables=list(variables), working_tree=sha, out_dir=out_dir,
        attempt=attempt, repo_path=str(root), llm_fn=llm,
    )


# --------------------------------------------------------------------------- #
# happy path: template-typed region → compilable float shim + float boundary
# --------------------------------------------------------------------------- #

def test_template_region_generates_float_shim_and_boundary(repo, tmp_path):
    root, _ = repo
    out = tmp_path / "shims"
    res = _call(repo, out, _CannedLLM())

    assert res.ok, res.error
    # shim written + installed, tagged `_float_`, hash stamped
    shim = Path(res.shim_paths[0])
    assert shim.exists() and shim.parent == out
    assert shim.name.startswith("kernel_float_") and shim.name.endswith(".h")
    # Wave-3 dedup: the per-family canonical shim is installed in the tree.
    assert (root / "ql_shim_float.h").exists()
    assert "SOURCE_HASH: PENDING" not in shim.read_text()

    # boundary: demote reads to float, retype the local to float, WIDEN the write
    # back with a PLAIN cast (native float has no .hi/.lo).
    patch = res.boundary_patch
    assert patch is not None
    assert "float a__ff = float(a);" in patch
    assert "float b__ff = float(b);" in patch
    assert "float r__ext = a__ff * b__ff;" in patch
    assert "T r = static_cast<T>(r__ext);" in patch
    # never two-limb reconstruction for a native float
    assert ".hi" not in patch and ".lo" not in patch
    # never promoted to an extended type
    assert "ffloat" not in patch and "ddouble" not in patch
    assert '#include "ql_shim_float.h"' in patch


def test_float_shim_uses_no_vendored_header(repo, tmp_path):
    # The include-set lint allows stdlib only for float (vendored_headers=[]); an
    # app-source include is still rejected as a retryable misgen.
    out = tmp_path / "shims"
    bad = _CannedLLM(
        "#pragma once\n// SOURCE_HASH: PENDING\n"
        '#include "ql/constants.h"\n'   # forbidden app-source header
    )
    res = _call(repo, out, bad)
    assert not res.ok
    assert "C1 include lint" in (res.error or "")


def test_float_shim_allows_stdlib_complex(repo, tmp_path):
    out = tmp_path / "shims"
    ok = _CannedLLM(
        "#pragma once\n// SOURCE_HASH: PENDING\n"
        "#include <complex>\n#include <cmath>\n"
        "// Rule 3: complex-at-float is std::complex<float>\n"
    )
    res = _call(repo, out, ok)
    assert res.ok, res.error


# --------------------------------------------------------------------------- #
# adversarial: mixed double driver constant → #error escape hatch (pinned choice)
# --------------------------------------------------------------------------- #

def test_adversarial_mixed_double_literal_escape_hatch(repo, tmp_path):
    # A region reading a double driver constant (M_PI) where float-vs-double is
    # ambiguous.  We PIN the behavior to the Rule R4 escape hatch: the model emits
    # a `#error` and the integrator passes it through faithfully (res.ok — a shim
    # WAS generated), so the ambiguity becomes a hard BUILD failure downstream
    # rather than a silent precision slip.  The integrator must NOT strip #error.
    root, sha = repo
    (root / "kernel.h").write_text(
        "#pragma once\n"
        "template <class T> T g(T a) {\n"
        "    T r = a * M_PI;\n"   # M_PI is a double driver constant → ambiguous
        "    return r;\n"
        "}\n"
    )
    _git(root, "commit", "-aqm", "mixed")
    sha = _git(root, "rev-parse", "HEAD").stdout.strip()

    escape_shim = (
        "#pragma once\n// SOURCE_HASH: PENDING\n"
        "// UNCLASSIFIED: M_PI\n"
        "// Rule 6 unclear because: double driver constant, float-vs-double ambiguous\n"
        '#error "Float Regional Integrator: M_PI requires manual classification"\n'
    )
    res = fl.integrate_region(
        file="kernel.h", line_start=3, line_end=3, variables=["a"],
        working_tree=sha, out_dir=tmp_path / "shims", attempt=0,
        repo_path=str(root), llm_fn=_CannedLLM(escape_shim),
    )
    assert res.ok, res.error                       # shim generated (build fails later)
    shim_text = Path(res.shim_paths[0]).read_text()
    assert "#error" in shim_text                    # escape hatch preserved
    assert "UNCLASSIFIED: M_PI" in shim_text


# --------------------------------------------------------------------------- #
# cache + retry + cache-key parity with the ff/dd twins
# --------------------------------------------------------------------------- #

def test_cache_hit_skips_llm(repo, tmp_path):
    out = tmp_path / "shims"
    llm = _CannedLLM()
    first = _call(repo, out, llm, attempt=0)
    second = _call(repo, out, llm, attempt=0)
    assert first.ok and second.ok
    assert first.shim_paths == second.shim_paths
    assert len(llm.calls) == 1
    assert second.llm_tokens == 0


def test_float_scalar_distinct_cache_key_from_ff_dd():
    from agents.integrator_base import cache
    region = "    T r = a * b;"
    k_float = cache.compute_region_hash(region, fl._SYSTEM_PROMPT, "float", [])
    k_ff = cache.compute_region_hash(region, fl._SYSTEM_PROMPT, "quad::ffun::ffloat", [])
    assert k_float != k_ff


def test_integrate_whole_app_not_implemented():
    with pytest.raises(NotImplementedError):
        fl.integrate("headers", "driver")


# --------------------------------------------------------------------------- #
# Integration: a real LLM call (skipped without the Argo proxy).
# --------------------------------------------------------------------------- #

@pytest.mark.llm
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_AUTH_TOKEN"),
                    reason="requires the Argo LLM proxy")
def test_real_llm_generates_parseable_float_shim(repo, tmp_path):
    res = _call(repo, tmp_path / "shims", None)   # llm_fn=None → real call
    assert res.ok, res.error
    shim = Path(res.shim_paths[0]).read_text()
    assert "#pragma once" in shim
    # native float target: never a vendored extended header / two-limb factory
    assert "ff_math.hpp" not in shim and "dd_math.hpp" not in shim
    assert res.boundary_patch and "float" in res.boundary_patch
    assert ".hi" not in res.boundary_patch
