"""Shared fixtures for the Patcher unit tests.

A tiny real git repo (so commits/resets/reverts are exercised for real) plus
helpers to build intents, mock integrators, and a mock build/smoke gate — so no
unit test needs libclang, a live LLM, or a real compiler.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from agents.patcher import gates, result as R


def git(repo, *args, **kw):
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True, **kw)


# A header with real float/double declarations and a comment/string that a naive
# sed would corrupt (the region-local edit must leave those alone).
HEADER = """\
#pragma once
namespace ql {
    // this float comment must not change and neither must "double" in a string
    inline double compute(double a, float b) {
        double result = a * b;   // float here is only in a comment
        return result;
    }
    struct float_traits { int x; };   // 'float_traits' identifier must survive
}
"""


@pytest.fixture(autouse=True)
def sleep_calls(monkeypatch):
    """Record (and never actually perform) the Patcher's inter-attempt backoff
    sleeps.  Autouse so every retry test runs instantly instead of waiting the
    real 2s/4s backoff; the backoff test inspects the returned list of delays."""
    calls: list[float] = []
    monkeypatch.setattr("agents.patcher.agent.time.sleep", calls.append)
    return calls


@pytest.fixture
def repo(tmp_path):
    root = tmp_path / "tree"
    root.mkdir()
    (root / "region.h").write_text(HEADER)
    git(root, "init", "-q")
    git(root, "config", "user.email", "t@t.t")
    git(root, "config", "user.name", "t")
    git(root, "config", "commit.gpgsign", "false")
    git(root, "add", "-A")
    git(root, "commit", "-q", "-m", "base")
    start = git(root, "rev-parse", "HEAD").stdout.strip()
    return root, start


@pytest.fixture
def make_ctx(tmp_path):
    def _make(repo_root, start, iter_id=1):
        run_dir = tmp_path / "run"
        run_dir.mkdir(exist_ok=True)
        return {"run_id": "testrun", "branch": "strategy/testrun",
                "repo_path": str(repo_root), "parent_sha": start,
                "run_dir": str(run_dir), "iter_id": iter_id}
    return _make


def intent(kind, *, file="region.h", line_start=4, line_end=6,
           variables=("result",), flavor="correctness",
           current_precision="double", identity=None, rationale_id="iter_1",
           via=None):
    payload = {
        "target": {"file": file, "line_start": line_start, "line_end": line_end,
                   "variables": list(variables)},
        "kind": kind, "intent": flavor,
        "current_precision": current_precision, "rationale_id": rationale_id,
    }
    if identity is not None:
        payload["identity"] = identity
    if via is not None:
        payload["via"] = via
    return payload


# -- gate mocks -------------------------------------------------------------

def ok_gate(*a, **k):
    logs = Path(k.get("logs_dir") or a[2])
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "b.log").write_text("ok")
    return gates.GateResult(R.OK, None, None, logs / "b.log", logs / "r.log")


def gate_returning(status, err_kind=None, detail="x"):
    def _g(*a, **k):
        return gates.GateResult(status, err_kind, detail)
    return _g


def flaky_gate(fail_times, fail_status=R.BUILD_FAILED, err_kind=R.ERR_COMPILE):
    """Fail the first ``fail_times`` calls, then return ok."""
    state = {"n": 0}

    def _g(*a, **k):
        state["n"] += 1
        if state["n"] <= fail_times:
            return gates.GateResult(fail_status, err_kind, "flaky")
        return ok_gate(*a, **k)
    return _g


# -- integrator mocks -------------------------------------------------------

from agents.integrator_base.region import RegionIntegrationResult


def make_shim_integrator(repo_root, *, fail_times=0, shim_name="region_ff.h"):
    """Integrator that writes a shim into the tree, succeeding after ``fail_times``."""
    state = {"n": 0}

    def _integ(**kw):
        state["n"] += 1
        if state["n"] <= fail_times:
            return RegionIntegrationResult.failed("mock integrator misgen")
        shim = Path(repo_root) / shim_name
        shim.write_text(f"// regional {kw['scalar_type']} shim (test)\n")
        return RegionIntegrationResult(status="ok", shim_paths=[str(shim)],
                                       boundary_patch=None, llm_tokens=42)
    return _integ
