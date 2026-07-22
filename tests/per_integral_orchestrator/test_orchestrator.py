"""Tree isolation + manifest completeness with a fake (no-LLM) pipeline."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from agents.per_integral_orchestrator import run_per_integral_pass
from tests.per_integral_orchestrator.conftest import _git


def _make_fake_pipeline(accepts, rejects, *, marker=None):
    """Build a ``pipeline_fn`` that fabricates Strategy's on-disk artifacts.

    Writes a fat report.json + iterations.jsonl (``accepts`` accepted rows +
    ``rejects`` rejected rows) + a cumulative diff, and returns the
    ``strategy_result`` bundle — exactly the shape ``build_manifest`` consumes.
    Optionally drops a ``marker`` file into the cloned tree (isolation probe).
    """
    def pipeline_fn(filtered_report: Path, tree: Path, out_dir: Path) -> dict:
        if marker is not None:
            (tree / marker).write_text(f"touched by {tree.name}\n")

        run_dir = out_dir / "strategy" / "run0"
        run_dir.mkdir(parents=True, exist_ok=True)
        iter_log = run_dir / "iterations.jsonl"
        report_json = run_dir / "report.json"
        diff_path = run_dir / "cumulative.diff"

        rows, assigns, dist = [], [], {}
        iid = 0
        for i in range(accepts):
            file = f"f{i}.h"
            rows.append({"iter_id": iid, "phase": "correctness",
                         "target": {"file": file, "line_start": 100 + i,
                                    "line_end": 100 + i},
                         "kind": "double-to-dd", "intent": "dd",
                         "current_precision": "double", "patcher_status": "ok",
                         "validator_verdict": "accept", "accepted": True,
                         "log_tag": "ok", "rationale": "r"})
            assigns.append({"file": file, "line_start": 100 + i,
                            "line_end": 100 + i, "precision": "dd",
                            "required_by": [], "rationale_id": f"iter_{iid}",
                            "phase": "correctness"})
            dist["dd"] = dist.get("dd", 0) + 1
            iid += 1
        for i in range(rejects):
            file = f"g{i}.h"
            rows.append({"iter_id": iid, "phase": "speedup",
                         "target": {"file": file, "line_start": 200 + i,
                                    "line_end": 200 + i},
                         "kind": "double-to-float", "intent": "float",
                         "current_precision": "double", "patcher_status": "ok",
                         "validator_verdict": "regressed", "accepted": False,
                         "log_tag": "reject", "rationale": "r"})
            iid += 1

        iter_log.write_text("".join(json.dumps(r) + "\n" for r in rows))
        report_json.write_text(json.dumps({
            "status": "success", "run_id": "run0",
            "iterations": len(rows), "duration_sec": 1.5,
            "precision_assignment": assigns,
            "precision_distribution": dist,
            "correctness_summary": {"ceiling_regions": []},
            "speedup_summary": {},
            "iteration_log_path": str(iter_log),
        }))
        # a cumulative diff whose +++ headers name the accepted files
        diff_lines = []
        for a in assigns:
            diff_lines += [f"--- a/{a['file']}", f"+++ b/{a['file']}",
                           "@@ -1 +1 @@", "-x", "+y"]
        diff_path.write_text("\n".join(diff_lines) + "\n")

        return {
            "status": "success", "run_id": "run0", "final_branch": "strategy/run0",
            "report_json_path": str(report_json),
            "report_md_path": str(run_dir / "report.md"),
            "cumulative_diff_path": str(diff_path),
        }

    return pipeline_fn


def test_tree_isolation_two_passes(base_repo, synth_report, tmp_path):
    repo, base_sha = base_repo
    report_path, _ = synth_report

    out_a = tmp_path / "out" / "B1"
    out_b = tmp_path / "out" / "B4"
    run_per_integral_pass("B1", report_path, repo, out_a,
                          pipeline_fn=_make_fake_pipeline(1, 0, marker="A.txt"))
    run_per_integral_pass("B4", report_path, repo, out_b,
                          pipeline_fn=_make_fake_pipeline(1, 0, marker="B.txt"))

    tree_a = out_a / "tree_B1"
    tree_b = out_b / "tree_B4"
    # two independent trees, each with only its own marker
    assert (tree_a / "A.txt").is_file() and not (tree_a / "B.txt").exists()
    assert (tree_b / "B.txt").is_file() and not (tree_b / "A.txt").exists()

    # mutating one tree does not affect the other
    (tree_a / "extra.txt").write_text("only in A\n")
    assert not (tree_b / "extra.txt").exists()

    # the base repo is untouched: same HEAD, clean status, no markers
    assert _git(repo, "rev-parse", "HEAD") == base_sha
    assert _git(repo, "status", "--porcelain") == ""
    assert not (repo / "A.txt").exists() and not (repo / "B.txt").exists()

    # clone preserved the base SHA (validator's starting_sha stays valid)
    assert _git(tree_a, "rev-parse", "HEAD") == base_sha


def test_manifest_completeness(base_repo, synth_report, tmp_path):
    repo, _ = base_repo
    report_path, _ = synth_report
    n_accept, m_reject = 3, 2

    out = tmp_path / "out" / "B1"
    manifest = run_per_integral_pass(
        "B1", report_path, repo, out,
        pipeline_fn=_make_fake_pipeline(n_accept, m_reject))

    # persisted next to the tree
    mpath = out / "manifest_B1.json"
    assert mpath.is_file()
    on_disk = json.loads(mpath.read_text())

    assert manifest["integral"] == "B1"
    assert manifest["counts"] == {"intents_attempted": n_accept + m_reject,
                                  "accepted": n_accept, "rejected": m_reject}
    # every attempt is listed with the full (file, line, precision, verdict, integral)
    assert len(manifest["decisions"]) == n_accept + m_reject
    for d in manifest["decisions"]:
        assert d["file"] and d["line"] is not None
        assert d["precision"] is not None
        assert d["verdict"] in ("accept", "regressed", "reject")
        assert d["integral"] == "B1"
    n_acc = sum(1 for d in manifest["decisions"] if d["verdict"] == "accept")
    assert n_acc == n_accept
    # accepted rows carry the settled precision from precision_assignment
    assert all(d["precision"] == "dd"
               for d in manifest["decisions"] if d["accepted"])

    # modified_files reconstructed from the cumulative diff
    assert manifest["modified_files"] == [f"f{i}.h" for i in range(n_accept)]
    assert manifest["timing"]["wall_sec"] >= 0
    assert on_disk["counts"] == manifest["counts"]


def test_build_gate_runs_before_pipeline(base_repo, synth_report, tmp_path):
    """A failing build gate aborts the pass before the pipeline runs."""
    repo, _ = base_repo
    report_path, _ = synth_report
    calls = {"pipeline": 0}

    def pipeline_fn(*a):
        calls["pipeline"] += 1
        return {}

    def failing_gate(tree):
        raise RuntimeError("vanilla build broken")

    import pytest
    with pytest.raises(RuntimeError, match="vanilla build broken"):
        run_per_integral_pass("B1", report_path, repo, tmp_path / "o",
                              pipeline_fn=pipeline_fn, build_gate_fn=failing_gate)
    assert calls["pipeline"] == 0   # pipeline never reached
