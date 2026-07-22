"""Fixtures/helpers for per-integral orchestrator tests (deterministic, no LLM)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest


def _region(integral: str, *, signal_class="stable", max_cond=1.0,
            max_rel_err=1e-16, pred_float=1e-16, pred_ff=None,
            value_range_ok_for_float=True, n=100, ops=None,
            variables=None, integral_tag=None) -> dict:
    """A v2 report region entry (the value under a ``"file:line"`` key)."""
    return {
        # Phase A explicit per-record integral tag; overridable to inject a
        # fidelity violation (integral_tag != container integral).
        "integral": integral if integral_tag is None else integral_tag,
        "signal_class": signal_class,
        "non_localizable": False,
        "max_cond": max_cond,
        "max_rel_err": max_rel_err,
        "predicted_rel_err_if_float": pred_float,
        "predicted_rel_err_if_ff": pred_float if pred_ff is None else pred_ff,
        "value_range_ok_for_float": value_range_ok_for_float,
        "n": n,
        "ops": dict(ops) if ops is not None else {"mul": 1},
        "region_local_vars": list(variables or ["x"]),
    }


def _chain(integral: str, chain_id: str, spans: list[tuple[str, int, int]], *,
           max_rel_err=1e-3, integral_tag=None) -> dict:
    return {
        "integral": integral if integral_tag is None else integral_tag,
        "chain_id": chain_id,
        "chain": [{"file": f, "line_start": a, "line_end": b}
                  for (f, a, b) in spans],
        "signal_class": "cancellation_cascade",
        "max_cond": 1e12,
        "max_rel_err": max_rel_err,
        "predicted_rel_err_if_float": 1e-3,
        "predicted_rel_err_if_ff": 1e-3,
        "n": 50,
        "ops": {"add": 3},
        "region_local_vars": ["a", "b"],
    }


def make_report_dict(integrals: dict) -> dict:
    """Wrap ``{name: {regions, cascade_chains}}`` into a v2 stability_report."""
    return {
        "schema_version": 2,
        "kind": "stability_report",
        "samples_seen": {name: 5000 for name in integrals},
        "no_id_records": 0,
        "integrals": {
            name: {
                "samples": 5000,
                "class_counts": {},
                "top_regions_by_rel_err": [],
                "regions": idata.get("regions", {}),
                "variables": [],
                "cascade_chains": idata.get("cascade_chains", []),
            } for name, idata in integrals.items()
        },
    }


@pytest.fixture
def synth_report(tmp_path):
    """A 3-integral v2 report on disk; returns (path, dict).

    * B1  — 2 regions, 0 chains
    * B4  — 3 regions, 1 two-line cascade chain
    * X9  — 1 region, 0 chains
    """
    integrals = {
        "B1": {"regions": {
            "boxGPU.h:99": _region("B1", signal_class="local_cancellation"),
            "B0m.h:405": _region("B1"),
        }},
        "B4": {"regions": {
            "boxGPU.h:99": _region("B4"),
            "B2m.h:65": _region("B4", signal_class="local_cancellation",
                                max_cond=1e10, max_rel_err=1e-4, pred_float=1e-4),
            "kokkosUtils.h:140": _region("B4"),
        }, "cascade_chains": [
            _chain("B4", "B4::c0", [("box/B4m.h", 10, 12), ("box/B4m.h", 20, 22)]),
        ]},
        "X9": {"regions": {
            "boxGPU.h:104": _region("X9"),
        }},
    }
    doc = make_report_dict(integrals)
    p = tmp_path / "report.json"
    p.write_text(json.dumps(doc, indent=2))
    return p, doc


def _git(repo: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args],
                       capture_output=True, text=True)
    assert r.returncode == 0, f"git {' '.join(args)} failed: {r.stderr}"
    return r.stdout.strip()


@pytest.fixture
def base_repo(tmp_path):
    """A minimal flat headers git repo (root holds boxGPU.h); returns (path, sha)."""
    repo = tmp_path / "base_repo"
    repo.mkdir()
    (repo / "boxGPU.h").write_text("// base boxGPU.h\n")
    (repo / "box").mkdir()
    (repo / "box" / "B4m.h").write_text("// base B4m.h\n")
    _git(repo, "init", "-q")
    _git(repo, "config", "user.name", "test")
    _git(repo, "config", "user.email", "test@local")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", "base")
    sha = _git(repo, "rev-parse", "HEAD")
    return repo, sha
