"""End-to-end Patcher: real strategy branch + real git commit + real vanilla
build + real smoke/NaN scan.

Two flavors:

* ``test_e2e_regional_ff_real_build`` — a *mocked* regional integrator returning a
  hand-written qcdloop shim (scope decision (b)); proves the Patcher plumbing
  offline (no LLM).
* ``test_e2e_regional_ff_real_llm`` — the *real* ff integrator (no injection): a
  real LLM generates the ff shim, the deterministic boundary patch promotes a real
  qcdloop region, and the candidate really builds + smokes + commits.  This is the
  agentic loop's forward slice on real code.  Marked ``llm`` (skipped without the
  Argo proxy) in addition to ``kokkos``.

Both are marked ``kokkos`` — they compile the qcdloop vanilla driver, so they need
the HPC toolchain + a Kokkos install.  Skipped automatically where those are absent.
"""

from __future__ import annotations

import difflib
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from agents.integrator_base.region import RegionIntegrationResult
from agents.patcher import result as R
from agents.patcher.agent import make_patcher_fn

_REPO = Path(__file__).resolve().parents[2]
_HEADERS = _REPO / "runs" / "qcdloop_headers_full"
_KOKKOS = Path.home() / "kokkos-install"

pytestmark = pytest.mark.kokkos

_needs_kokkos = pytest.mark.skipif(
    not _KOKKOS.is_dir() or not _HEADERS.is_dir(),
    reason="requires ~/kokkos-install and the qcdloop headers tree")


def _git(repo, *args):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True)


def _append_patch(repo_root: Path, relpath: str, added_line: str) -> str:
    """A valid `git apply -p1` diff appending one line to a tracked file."""
    original = (repo_root / relpath).read_text()
    patched = original + added_line
    diff = difflib.unified_diff(
        original.splitlines(keepends=True), patched.splitlines(keepends=True),
        fromfile=f"a/{relpath}", tofile=f"b/{relpath}")
    return "".join(diff)


@pytest.fixture
def qcdloop_repo(tmp_path):
    """A git repo whose root is a copy of the qcdloop header tree (boxGPU.h + box/)."""
    root = tmp_path / "cand"
    shutil.copytree(_HEADERS, root)
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "t@t.t")
    _git(root, "config", "user.name", "t")
    _git(root, "config", "commit.gpgsign", "false")
    _git(root, "add", "-A")
    _git(root, "commit", "-q", "-m", "base")
    start = _git(root, "rev-parse", "HEAD").stdout.strip()
    _git(root, "checkout", "-q", "-B", "strategy/e2e", start)
    return root, start


def _hand_written_shim_integrator(repo_root: Path):
    """Mock integrator: write a self-contained shim + a harmless boundary patch."""
    def _integ(**kw):
        shim = Path(kw["out_dir"]) / "region_ff.h"
        Path(kw["out_dir"]).mkdir(parents=True, exist_ok=True)
        shim.write_text("#pragma once\n// hand-written regional ff shim (e2e test)\n")
        # copy the shim into the tree so it is committed with the candidate
        (repo_root / "region_ff.h").write_text(shim.read_text())
        boundary = _append_patch(repo_root, "boxGPU.h",
                                 "// regional ff boundary marker (e2e)\n")
        return RegionIntegrationResult(status="ok", shim_paths=[str(shim)],
                                       boundary_patch=boundary, llm_tokens=7)
    return _integ


@_needs_kokkos
def test_e2e_regional_ff_real_build(qcdloop_repo, tmp_path):
    root, start = qcdloop_repo
    run_dir = tmp_path / "run"
    ctx = {"run_id": "e2e", "branch": "strategy/e2e", "repo_path": str(root),
           "parent_sha": start, "run_dir": str(run_dir), "iter_id": 1}

    fn = make_patcher_fn(
        integrators={"ff": _hand_written_shim_integrator(root)},
        build_config={"headers_dir": str(root), "kokkos_root": str(_KOKKOS)})

    intent = {
        "target": {"file": "box/B2m.h", "line_start": 20, "line_end": 22,
                   "variables": []},
        "kind": "double-to-ff", "intent": "speedup",
        "current_precision": "double", "rationale_id": "iter_1"}

    resp = fn(intent, ctx)

    # -- P2 ok --
    assert resp["status"] == R.OK, resp.get("error")
    assert resp["candidate_sha"] and resp["parent_sha"] == start
    assert resp["llm_tokens"] == 7
    art = resp["artifacts"]
    assert art["shim_paths"] and Path(art["shim_paths"][0]).exists()
    assert art["boundary_patch_path"] and Path(art["boundary_patch_path"]).exists()
    assert Path(art["build_log_path"]).exists()
    assert Path(art["runtime_log_path"]).exists()

    # -- branch state: exactly one commit on top of start, tree carries the shim + marker --
    count = _git(root, "rev-list", "--count", f"{start}..HEAD").stdout.strip()
    assert count == "1"
    committed = _git(root, "show", f"{resp['candidate_sha']}:boxGPU.h").stdout
    assert "regional ff boundary marker (e2e)" in committed
    assert subprocess.run(["git", "-C", str(root), "cat-file", "-e",
                           f"{resp['candidate_sha']}:region_ff.h"],
                          capture_output=True).returncode == 0

    # -- commit message follows the Q3 machine-parseable schema --
    subject = _git(root, "log", "-1", "--format=%s", resp["candidate_sha"]).stdout.strip()
    assert subject.startswith("[iter_1] double-to-ff box/B2m.h:20-22")

    # -- smoke run really produced >= 21 result rows, no NaN --
    runtime = Path(art["runtime_log_path"]).read_text()
    assert runtime.count("RES,") >= 21


@_needs_kokkos
@pytest.mark.llm
@pytest.mark.skipif(not os.environ.get("ANTHROPIC_AUTH_TOKEN"),
                    reason="requires the Argo LLM proxy")
def test_e2e_regional_ff_real_llm(qcdloop_repo, tmp_path):
    """The forward slice on real code: real LLM ff shim + deterministic boundary
    patch → real vanilla build + smoke + commit, with NO injected integrator.

    The region is a real single-statement double computation
    (``kokkosUtils.h:312`` — ``TMass arg = x1 * x2;`` inside ``Li2omx``); the local
    is declared through the ``TMass`` template alias (``double`` at the vanilla
    instantiation), exercising the boundary's dataflow-based promotion + demote-to-
    original-type even though the Patcher passes ``caller_type="double"``.
    """
    root, start = qcdloop_repo
    run_dir = tmp_path / "run"
    ctx = {"run_id": "e2e-llm", "branch": "strategy/e2e", "repo_path": str(root),
           "parent_sha": start, "run_dir": str(run_dir), "iter_id": 1}

    # No integrators override → the real ff_integrator.integrate_region is used.
    fn = make_patcher_fn(
        build_config={"headers_dir": str(root), "kokkos_root": str(_KOKKOS)})

    intent = {
        "target": {"file": "kokkosUtils.h", "line_start": 312, "line_end": 312,
                   "variables": ["x1", "x2"]},
        "kind": "double-to-ff", "intent": "speedup",
        "current_precision": "double", "rationale_id": "iter_1"}

    resp = fn(intent, ctx)

    # -- P2 ok: candidate committed, real LLM tokens spent --
    assert resp["status"] == R.OK, resp.get("error")
    assert resp["candidate_sha"] and resp["parent_sha"] == start
    assert resp["llm_tokens"] > 0
    art = resp["artifacts"]
    shim = Path(art["shim_paths"][0])
    assert shim.exists() and shim.name.startswith("kokkosUtils_ff_")
    assert art["boundary_patch_path"] and Path(art["boundary_patch_path"]).exists()

    # -- the real shim is a valid header referencing the vendored ff type --
    shim_text = shim.read_text()
    assert "#pragma once" in shim_text
    assert "ff_math.hpp" in shim_text
    assert "SOURCE_HASH: PENDING" not in shim_text

    # -- the committed region carries the deterministic boundary edits, demoting to
    #    the local's own template-alias type (TMass), not the passed caller "double" --
    committed = _git(root, "show", f"{resp['candidate_sha']}:kokkosUtils.h").stdout
    assert "Kokkos::Experimental::FloatFloat x1__ff = Kokkos::Experimental::FloatFloat(x1);" in committed
    assert "Kokkos::Experimental::FloatFloat arg__ext = x1__ff * x2__ff;" in committed
    assert "TMass arg = static_cast<TMass>(arg__ext.hi)" in committed
    # Wave-3 dedup: the region #includes the canonical per-family shim (not a
    # per-region file); that canonical shim is committed and carries the ff type.
    assert '#include "ql_shim_ff.h"' in committed
    canonical = _git(root, "show", f"{resp['candidate_sha']}:ql_shim_ff.h").stdout
    assert "ff_math.hpp" in canonical and "SOURCE_HASH: PENDING" not in canonical

    # -- exactly one commit, schema-conformant subject, smoke produced 21 rows --
    assert _git(root, "rev-list", "--count", f"{start}..HEAD").stdout.strip() == "1"
    subject = _git(root, "log", "-1", "--format=%s", resp["candidate_sha"]).stdout.strip()
    assert subject.startswith("[iter_1] double-to-ff kokkosUtils.h:312")
    runtime = Path(art["runtime_log_path"]).read_text()
    assert runtime.count("RES,") >= 21
