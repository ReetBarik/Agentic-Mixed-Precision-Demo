"""Smoke tests — SOLVER_STAGE markdown renders for both outcomes (Phase 2e).

Gate is regression-relative (margin 0.5 vs baseline); the stopped case is now an
*unscoreable* baseline, not a low absolute floor.
"""

from agents.solver.queue import build_queue
from agents.solver.report import build_markdown
from agents.solver.solver import ApplyResult, ValidateResult, solve


def _rows():
    return [
        {"region_id": "A.h:10", "rung": "float", "status": "measured",
         "delta_effective": 1e-7, "baseline_delta_effective": 1e-13,
         "patcher_metadata": {"kind": "double-to-float", "intent": "speedup",
                              "via": "regional"}},
        {"region_id": "A.h:10", "rung": "ff", "status": "measured",
         "delta_effective": 1e-14, "baseline_delta_effective": 1e-13,
         "patcher_metadata": {"kind": "double-to-ff", "intent": "speedup",
                              "via": "regional"}},
        {"region_id": "B.h:20", "rung": "float", "status": "measured",
         "delta_effective": 2.5e-4, "baseline_delta_effective": 2.5e-4,   # INERT
         "patcher_metadata": {"kind": "double-to-float", "intent": "speedup",
                              "via": "regional"}},
    ]


def _run(cand_min_by_key, baseline=8.84, margin=0.5):
    qb = build_queue(_rows())
    head = ["s0"]; n = [0]; applied = []

    def apply_fn(c, parent):
        applied.append((c.region_id, c.rung))
        n[0] += 1; head[0] = f"s{n[0]}"
        return ApplyResult(ok=True, candidate_sha=head[0], patcher_status="ok",
                           wall_sec=8.0)

    def validate_fn(sha, gb, gth):
        cm = cand_min_by_key.get(applied[-1])
        return ValidateResult(cand_min=cm, curr_min=baseline,
                              combined_cand_min=cm, verdict="?", wall_sec=5.0)

    def revert_fn(p): head[0] = p

    res = solve(qb.queue, apply_fn=apply_fn, validate_fn=validate_fn,
                revert_fn=revert_fn, head_fn=lambda: head[0], margin=margin,
                all_region_ids={"A.h:10", "B.h:20"})
    return res, qb


def _md(res, qb, **kw):
    return build_markdown(res, qb, integral="B12", tree_path="/t", diff_path="/d",
                          manifest_path="/m", report_regions={}, margin=0.5,
                          solve_wall_sec=13.0, snapshot={"seed": 1, "sample_count": 5000},
                          **kw)


def test_report_renders_accept_case():
    res, qb = _run({("A.h:10", "float"): 8.5})   # float holds (>= 8.34) -> ff skipped
    md = _md(res, qb)
    assert "Solver Stage — B12" in md
    assert "Candidate queue" in md
    assert "Reet review" in md
    assert "Blocking finding" not in md           # normal termination
    assert "regression-relative gate" in md.lower()
    assert "→ **float**" in md                    # A.h:10 landed float


def test_report_renders_stopped_case_with_finding():
    res, qb = _run({("A.h:10", "float"): 3.0}, baseline=None)  # unscoreable baseline
    md = _md(res, qb,
             baseline_hotspot={"integral": "B12", "sample_idx": 3868,
                               "component": "c1", "precise_digits": 3.5})
    assert "Blocking finding" in md
    assert "unscoreable" in md.lower()
    assert "regression-relative gate" in md.lower()


def test_report_lists_inert_exclusions():
    res, qb = _run({("A.h:10", "float"): 8.5})
    md = _md(res, qb)
    assert "measured-INERT" in md.lower() or "INERT" in md
    assert "`B.h:20`" in md    # the inert cell is enumerated
