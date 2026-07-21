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
    # Wave-3 dedup: the region's shim is merged into the canonical per-family shim
    # installed in the tree (not a per-region file), and the boundary #includes it.
    canonical = root / "ql_shim_dd.h"
    assert canonical.exists()
    assert "#include <dd_math.hpp>" in canonical.read_text()
    assert '#include "ql_shim_dd.h"' in res.boundary_patch

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


def test_ruleset_forbids_app_source_includes():
    # The DD ruleset must codify the closed include set (C1) — the fix for the
    # 2026-07-18 shakedown's dominant dd_untested cause.
    assert "C1." in dd._SYSTEM_PROMPT
    assert "app-source" in dd._SYSTEM_PROMPT


class _BadIncludeLLM:
    """Emits a shim that hallucinates an app-source header (the shakedown bug)."""

    def __init__(self):
        self.calls = []

    def __call__(self, system, user, attempt):
        self.calls.append((system, user, attempt))
        return (
            "#pragma once\n"
            "// SOURCE_HASH: PENDING\n"
            "#include <dd_math.hpp>\n"
            "#include <dd_complex.hpp>\n"
            '#include "ql/constants.h"\n'   # <-- forbidden app-source header
            "// Rule 5: Constants<ddouble> specialization\n"
        )


def test_forbidden_include_is_rejected_as_misgen(repo, tmp_path):
    # The include-set lint must turn an app-source #include into a retryable
    # llm_failed (so the Patcher re-rolls), NOT let it through to a doomed build.
    out = tmp_path / "shims"
    llm = _BadIncludeLLM()
    res = _call(repo, out, llm)
    assert not res.ok
    assert res.status == "llm_failed"
    assert "C1 include lint" in (res.error or "")
    assert "ql/constants.h" in (res.error or "")
    # A rejected shim must not be persisted into the candidate tree.
    root, _ = repo
    assert not list(root.glob("kernel_dd_*.h"))


# --------------------------------------------------------------------------- #
# Integration: a real LLM call (skipped without the Argo proxy).
# --------------------------------------------------------------------------- #

# Exact region sources that triggered app-source-include hallucinations in run
# 20260718_194556_67dbcf37 (all reference ql:: symbols that tempt an app header).
_PREVIOUSLY_FAILING_REGIONS = {
    "B2m.h:65": (
        "        const TOutput k34c = TOutput(k34 - ql::Max(ql::kAbs(k34), "
        "TMass(ql::Constants<TMass>::_one())) * "
        "ql::Constants<TScale>::template _ieps50<TOutput, TMass, TScale>()) / k13c;"
    ),
    "B4m.h:163": (
        "        const TScale gamma = ql::Sign(ql::Real(a*(x[1][3] - x[0][3])) "
        "+ ql::Constants<TScale>::_reps());"
    ),
}


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


@pytest.mark.llm
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_AUTH_TOKEN"),
                    reason="requires the Argo LLM proxy")
@pytest.mark.parametrize("loc", sorted(_PREVIOUSLY_FAILING_REGIONS))
def test_real_llm_previously_failing_region_has_clean_includes(loc, tmp_path):
    """Regenerate a region that hallucinated an app-source #include in the
    2026-07-18 shakedown; with C1 in the prompt (+ lint net) the shim must build
    a clean include set (res.ok — the lint would have flipped it to llm_failed).
    """
    region_src = _PREVIOUSLY_FAILING_REGIONS[loc]
    root = tmp_path / "cand"
    root.mkdir()
    # A header carrying the exact failing region line — the generation trigger is
    # the region text (ql:: symbols), not the surrounding tree.
    (root / "region.h").write_text("#pragma once\n" + region_src + "\n")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t.t")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    sha = _git(root, "rev-parse", "HEAD").stdout.strip()

    res = dd.integrate_region(
        file="region.h", line_start=2, line_end=2, variables=[],
        working_tree=sha, out_dir=tmp_path / "shims",
        repo_path=str(root), llm_fn=None,
    )
    assert res.ok, f"{loc}: {res.error}"
    shim = Path(res.shim_paths[0]).read_text()
    from agents.integrator_base import regional as _r
    bad = _r._lint_include_set(shim, _r._allowed_include_set(dd._SPEC))
    assert bad is None, f"{loc}: {bad}"


# --------------------------------------------------------------------------- #
# Gap B (source-derivable constants) — real-LLM reproduction of the _ieps50 R4.
# --------------------------------------------------------------------------- #

# A minimal kokkosMaths-shaped header defining _ieps50 as a source double literal,
# co-located so the derivation helper can resolve its RHS (as in the full tree).
_IEPS50_HEADER = (
    "#pragma once\n"
    "namespace ql {\n"
    "template<class T> struct Constants {\n"
    "  static constexpr T _one() { return T(1); }\n"
    "  template<class TOutput, class TMass, class TScale>\n"
    "  static TOutput _ieps50() { return TOutput{Constants<TScale>::_zero(), TScale(1e-50)}; }\n"
    "};\n"
    "template<class T> T kAbs(T const& x);\n"
    "template<class T> T Max(T const& a, T const& b);\n"
    "}\n"
)

_IEPS50_REGION_LINE = (
    "        const TOutput k34c = TOutput(k34 - ql::Max(ql::kAbs(k34), "
    "TMass(ql::Constants<TMass>::_one())) * "
    "ql::Constants<TScale>::template _ieps50<TOutput, TMass, TScale>()) / k13c;"
)


@pytest.mark.llm
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_AUTH_TOKEN"),
                    reason="requires the Argo LLM proxy")
def test_real_llm_ieps50_derived_not_r4(tmp_path):
    """B0m.h:69 / B2m.h:65 shape: the region reads `_ieps50` (source `1e-50`).

    Pre-Gap-B this tripped the Rule R4 #error (and even guessed wrong bits).  With
    the R3 cascade + the pre-derived hint, the shim must generate cleanly (no
    #error) and carry the CORRECT double-double bits for 1e-50 (hi 0x358d…, lo 0).
    """
    root = tmp_path / "cand"
    root.mkdir()
    (root / "kokkosMaths.h").write_text(_IEPS50_HEADER)
    (root / "region.h").write_text("#pragma once\n" + _IEPS50_REGION_LINE + "\n")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t.t")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    sha = _git(root, "rev-parse", "HEAD").stdout.strip()

    res = dd.integrate_region(
        file="region.h", line_start=2, line_end=2, variables=[],
        working_tree=sha, out_dir=tmp_path / "shims",
        repo_path=str(root), llm_fn=None,
    )
    assert res.ok, res.error
    shim = Path(res.shim_paths[0]).read_text()
    assert "#error" not in shim, f"still hit R4:\n{shim}"
    # the exact bits the model previously got wrong — proof the derivation was used
    assert "358dee7a4ad4b81f" in shim.lower()


@pytest.mark.llm
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_AUTH_TOKEN"),
                    reason="requires the Argo LLM proxy")
def test_real_llm_synthetic_derivable_constant_no_r4(tmp_path):
    """Non-qcdloop Gap B: a plain `constexpr double MY_TINY = 1e-40` must derive,
    never R4 — exercises the generic path with zero qcdloop symbols."""
    root = tmp_path / "cand"
    root.mkdir()
    (root / "consts.h").write_text("#pragma once\nconstexpr double MY_TINY = 1e-40;\n")
    (root / "kernel.h").write_text(
        '#pragma once\n#include "consts.h"\n'
        "double g(double a) {\n    double c = MY_TINY * a;\n    return c;\n}\n")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t.t")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    sha = _git(root, "rev-parse", "HEAD").stdout.strip()

    res = dd.integrate_region(
        file="kernel.h", line_start=4, line_end=4, variables=["a"],
        working_tree=sha, out_dir=tmp_path / "shims",
        repo_path=str(root), llm_fn=None,
    )
    assert res.ok, res.error
    shim = Path(res.shim_paths[0]).read_text()
    assert "#error" not in shim, f"unexpected R4:\n{shim}"


# --------------------------------------------------------------------------- #
# Gap A (namespace-qualified bridge) — real-LLM synthetic std::sqrt(promoted).
# --------------------------------------------------------------------------- #

@pytest.mark.llm
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_AUTH_TOKEN"),
                    reason="requires the Argo LLM proxy")
def test_real_llm_qualified_call_gets_bridge(tmp_path):
    """A namespace-qualified `std::sqrt(x)` on a promoted read must get a bridge
    overload in `namespace std` (or a using-decl) — otherwise the C3 bridge lint
    would have flipped the result to llm_failed.  Generic, no qcdloop symbols."""
    root = tmp_path / "cand"
    root.mkdir()
    (root / "kernel.h").write_text(
        "#pragma once\n#include <cmath>\n"
        "double g(double x) {\n    double r = std::sqrt(x) + std::fabs(x);\n"
        "    return r;\n}\n")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t.t")
    _git(root, "config", "user.name", "t")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    sha = _git(root, "rev-parse", "HEAD").stdout.strip()

    res = dd.integrate_region(
        file="kernel.h", line_start=4, line_end=4, variables=["x"],
        working_tree=sha, out_dir=tmp_path / "shims",
        repo_path=str(root), llm_fn=None,
    )
    assert res.ok, res.error
    shim = Path(res.shim_paths[0]).read_text()
    from agents.integrator_base import regional as _r
    region_line = "double r = std::sqrt(x) + std::fabs(x);"
    assert _r._lint_qualified_bridges(region_line, shim, frozenset({"x", "r"})) is None, shim
