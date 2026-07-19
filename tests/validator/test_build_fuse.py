"""Build-fuse: the Validator reuses the Patcher's gate binary for the candidate
run when the tree hash matches, instead of rebuilding the monolithic TU
(WI2 / CALIBRATION.md §Bug 5).

Drives ``_run_vanilla`` directly with ``runner.build_driver`` /
``run_and_aggregate`` monkeypatched, so the test needs no compiler or cluster
modules — it asserts only *whether the build command was invoked*.
"""

from pathlib import Path

import pytest

import sys

import agents.validator.validate  # noqa: F401  (load the submodule)
from agents.integrator_base import cache as _hashcache
from agents.validator import runner as _runner

# `agents.validator.__init__` rebinds the name `validate` to the function, so
# `import ... as V` would grab the function; reach the module via sys.modules.
V = sys.modules["agents.validator.validate"]


@pytest.fixture
def vanilla_tree(tmp_path):
    """A tiny header tree standing in for qcdloop_headers_full."""
    root = tmp_path / "vanilla"
    (root / "box").mkdir(parents=True)
    (root / "boxGPU.h").write_text("#pragma once\n// boxGPU\n")
    (root / "box" / "B2m.h").write_text("l1\n  double x;\nl3\n")
    return root


def _patch_runner(monkeypatch, *, built):
    """Record build_driver calls; stub run_and_aggregate to a sentinel."""
    def fake_build(tree, mode, build_dir, kokkos_root):
        built.append(Path(tree))
        return Path(build_dir) / "boxGPU_app"

    def fake_run(binary, total, *, chunk=0, workers=1):
        return {"_binary": str(binary), "_total": total}

    monkeypatch.setattr(_runner, "build_driver", fake_build)
    monkeypatch.setattr(_runner, "run_and_aggregate", fake_run)


def test_reuse_skips_build_when_hash_matches(tmp_path, vanilla_tree, monkeypatch):
    built: list = []
    _patch_runner(monkeypatch, built=built)

    # No candidate patch → the candidate tree == a copy of vanilla_tree, whose
    # header-content hash equals hash_header_dir(vanilla_tree).
    tree_hash = _hashcache.hash_header_dir(vanilla_tree)
    reuse_binary = tmp_path / "gate" / "boxGPU_app"
    reuse_binary.parent.mkdir(parents=True)
    reuse_binary.write_text("#!/bin/true\n")   # just needs to exist

    out = V._run_vanilla(
        vanilla_tree, [], None, tmp_path / "kokkos",
        tmp_path / "scratch", total=1000, chunk=0, workers=1,
        reuse_binary=str(reuse_binary), reuse_tree_hash=tree_hash)

    assert built == []                              # build_driver was NOT invoked
    assert out["_binary"] == str(reuse_binary)      # ran the reused gate binary
    assert out["_total"] == 1000


def test_falls_back_to_build_on_hash_mismatch(tmp_path, vanilla_tree, monkeypatch):
    built: list = []
    _patch_runner(monkeypatch, built=built)

    reuse_binary = tmp_path / "gate" / "boxGPU_app"
    reuse_binary.parent.mkdir(parents=True)
    reuse_binary.write_text("#!/bin/true\n")

    out = V._run_vanilla(
        vanilla_tree, [], None, tmp_path / "kokkos",
        tmp_path / "scratch", total=1000, chunk=0, workers=1,
        reuse_binary=str(reuse_binary), reuse_tree_hash="deadbeef_wrong")

    assert len(built) == 1                          # mismatch → rebuilt
    assert out["_binary"].endswith("boxGPU_app")


def test_falls_back_to_build_when_no_artifact(tmp_path, vanilla_tree, monkeypatch):
    built: list = []
    _patch_runner(monkeypatch, built=built)

    out = V._run_vanilla(
        vanilla_tree, [], None, tmp_path / "kokkos",
        tmp_path / "scratch", total=1000, chunk=0, workers=1)

    assert len(built) == 1                          # no reuse artifact → build
    assert out["_total"] == 1000
