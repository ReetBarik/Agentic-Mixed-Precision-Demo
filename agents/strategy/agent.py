"""Strategy agent — the remediation loop (design: docs/strategy_patcher_design.md).

Strategy owns the whole loop internally (Q5): it reads a fixed characterization
report, ranks regions into a correctness queue and a speedup queue, and for each
target drives a mechanical vocabulary walk through the Patcher and Validator
(supplied as callables on the state).  Correctness mode drains first; speedup
mode second.  Every iteration — accepted or rejected — is logged; per-patch
commits live on ``strategy/<run_id>`` (the Patcher makes them), and the run ends
by writing ``report.json``, ``report.md`` and ``final.diff``.

``run(state) -> dict`` returns the thin state-delta bundle (Q5); the fat
artifacts live on disk under ``runs/qcdloop/strategy/<run_id>/``.

Patcher/Validator callable contracts (mocked in tests; real adapters deferred):

    patcher_fn(intent: dict, ctx: dict) -> P2 response dict
    validator_fn(candidate_sha: str, ctx: dict) -> verdict dict
        verdict dict: {"verdict": "accept"|"reject",
                       "candidate": {"min_precise_digits": float}, ...}
"""

from __future__ import annotations

import datetime
import hashlib
import time
from pathlib import Path

from agents.config import StrategyConfig
from agents.state import PipelineState
from agents.strategy.characterization import load_regions
from agents.strategy.dispatch import dispatch
from agents.strategy.gitops import GitRepo
from agents.strategy.iteration_log import IterationLogger
from agents.strategy.models import INTENT_CORRECTNESS, INTENT_SPEEDUP, LADDER
from agents.strategy.ranking import build_queues
from agents.strategy.report import write_reports
from agents.strategy.walk import RetryWalk

_REPO = Path(__file__).resolve().parents[2]


def _new_run_id(seed_material: str) -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    h = hashlib.sha1((seed_material + ts).encode()).hexdigest()[:8]
    return f"{ts}_{h}"


def run(state: PipelineState) -> dict:
    """LangGraph node entry point — see module docstring."""
    return StrategyRun(state).execute()


class StrategyRun:
    """One Strategy invocation.  Holds all mutable run state."""

    def __init__(self, state: PipelineState):
        cfg = state.get("strategy_config") or StrategyConfig()
        self.cfg = cfg
        self.tolerance = float(cfg.tolerance)
        self.snapshot = dict(cfg.snapshot)

        self.report_path = state.get("characterization_report_path")
        self.repo_path = state.get("strategy_repo_path")
        self.starting_sha = state.get("strategy_starting_sha")
        self.patcher_fn = state.get("patcher_fn")
        self.validator_fn = state.get("validator_fn")
        if self.patcher_fn is None or self.validator_fn is None:
            raise ValueError("Strategy requires patcher_fn and validator_fn on the state")
        if not self.report_path:
            raise ValueError("Strategy requires characterization_report_path on the state")

        runs_root = Path(cfg.runs_root) if cfg.runs_root else _REPO / "runs" / "qcdloop"
        self.run_id = _new_run_id(str(self.starting_sha) + str(self.report_path))
        self.branch = f"strategy/{self.run_id}"
        self.run_dir = runs_root / "strategy" / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.repo = GitRepo(self.repo_path) if self.repo_path else None
        self.logger = IterationLogger(self.run_dir)

        # -- budget / stop bookkeeping --
        self.t0 = time.monotonic()
        self.budget_iters = 0
        self.tokens = 0
        self.dr_streak = 0
        self.stop_status: str | None = None   # set → hard/soft stop requested

        # -- accumulators for the report --
        self.precision_assignment: list[dict] = []
        self.rewrites: list[dict] = []
        self.ceiling_regions: list[dict] = []
        self.region_final: dict[tuple, str] = {}
        self.region_status: dict[tuple, str] = {}

    # ------------------------------------------------------------------
    def execute(self) -> dict:
        regions, meta = load_regions(self.report_path)
        correctness_q, speedup_q = build_queues(regions, self.tolerance)

        # every localizable region starts at the double baseline
        for r in regions:
            self.region_final[r.key] = "double"
            self.region_status[r.key] = "baseline_ok"
        for r in correctness_q:
            self.region_status[r.key] = "unworked"

        if self.repo and self.starting_sha:
            self.repo.create_branch(self.branch, self.starting_sha)

        # -- correctness mode drains first --
        for record in correctness_q:
            if self.stop_status:
                break
            self._process_target(record, INTENT_CORRECTNESS)

        # -- speedup mode second (only if correctness fully drained) --
        if not self.stop_status:
            for record in speedup_q:
                if self.stop_status:
                    break
                self._process_target(record, INTENT_SPEEDUP)

        status = self.stop_status or "success"
        return self._finalize(status, meta, len(correctness_q))

    # ------------------------------------------------------------------
    def _process_target(self, record, mode: str) -> None:
        """Drive one target's retry walk to termination (or until a stop fires)."""
        walk = RetryWalk(record, mode, self.tolerance, baseline="double")
        walk_digits: float | None = None
        timed_out_kinds: set[str] = set()

        while True:
            intent = walk.propose("")
            if intent is None:
                break
            iter_id = self.logger.next_iter_id()
            intent.rationale_id = self.logger.rationale_id(iter_id)

            resp = self._invoke_patcher(intent, iter_id, timed_out_kinds)
            status = resp.get("status")
            entry = dispatch(status)

            # ---- fatal: commit failure aborts the whole run (Q3) ----
            if entry.action == "fatal":
                self.logger.write(
                    iter_id=iter_id, target=intent.target.as_dict(), kind=intent.kind,
                    intent=intent.intent, current_precision=intent.current_precision,
                    patcher_status=status, validator_verdict=None, accepted=False,
                    log_tag=entry.log_tag, rationale="commit_failed → internal_error",
                    extra={"candidate_sha": resp.get("candidate_sha")})
                self.stop_status = "internal_error"
                return

            patcher_ok = status == "ok"
            validator_verdict = None
            verdict_digits = None
            if patcher_ok:
                verdict = self._invoke_validator(resp.get("candidate_sha"), iter_id)
                validator_verdict = verdict.get("verdict")
                verdict_digits = (verdict.get("candidate") or {}).get("min_precise_digits")
                if verdict_digits is not None:
                    walk_digits = verdict_digits

            accepted = patcher_ok and validator_verdict == "accept"
            genuine_reject = patcher_ok and validator_verdict == "reject"

            is_rewrite = intent.kind.startswith("reformulate")
            is_dd_promo = intent.kind.endswith("-to-dd")

            # ---- git: keep or revert the candidate (option (a)) ----
            retain = accepted or (genuine_reject and is_dd_promo)
            if patcher_ok and not retain and self.repo and resp.get("parent_sha"):
                self.repo.reset_hard(resp["parent_sha"])

            # ---- record accepted / retained remediations for the report ----
            self._record_remediation(intent, accepted, genuine_reject, is_rewrite, is_dd_promo)

            # ---- iteration log ----
            self.logger.write(
                iter_id=iter_id, target=intent.target.as_dict(), kind=intent.kind,
                intent=intent.intent, current_precision=intent.current_precision,
                patcher_status=status, validator_verdict=validator_verdict,
                accepted=accepted, log_tag=entry.log_tag,
                rationale=self._rationale(intent, entry.log_tag, accepted),
                strategy_bug=(entry.log_tag == "strategy_bug"),
                extra=self._log_extra(intent, resp, verdict_digits))

            # ---- budget / diminishing-returns accounting ----
            if entry.counts_budget:
                self.budget_iters += 1
            self.tokens += int(resp.get("llm_tokens", 0) or 0)
            if accepted:
                self.dr_streak = 0
            elif entry.log_tag != "strategy_bug":
                self.dr_streak += 1

            # ---- advance the walk state machine ----
            walk.resolve(accepted=accepted, genuine_reject=genuine_reject)

            # ---- stopping conditions (checked every iteration) ----
            if self._check_stops():
                # walk interrupted mid-flight: leave region provisional
                self.region_final[record.key] = walk.installed
                self.region_status[record.key] = "unresolved"
                return

        self._finish_walk(record, walk.result(), walk_digits)

    # ------------------------------------------------------------------
    def _invoke_patcher(self, intent, iter_id: int, timed_out_kinds: set) -> dict:
        """Call Patcher; implement the P6 timeout retry-once → fold-to-advance."""
        ctx = self._patcher_ctx(iter_id)
        resp = self.patcher_fn(intent.to_patcher(), ctx)
        if resp.get("status") == "timeout" and intent.kind not in timed_out_kinds:
            timed_out_kinds.add(intent.kind)
            resp = self.patcher_fn(intent.to_patcher(), {**ctx, "retry": True})
            if resp.get("status") == "timeout":
                # second timeout → fold into a Bucket-A reject (build_failed-equivalent)
                resp = {**resp, "status": "build_failed",
                        "error": {"kind": "timeout", "detail": "second timeout folded to build_failed"}}
        return resp

    def _invoke_validator(self, candidate_sha, iter_id: int) -> dict:
        ctx = {"run_id": self.run_id, "branch": self.branch,
               "repo_path": str(self.repo_path) if self.repo_path else None,
               "tolerance": self.tolerance, "snapshot": self.snapshot,
               "iter_id": iter_id}
        return self.validator_fn(candidate_sha, ctx)

    def _patcher_ctx(self, iter_id: int) -> dict:
        return {"run_id": self.run_id, "branch": self.branch,
                "repo_path": str(self.repo_path) if self.repo_path else None,
                "parent_sha": self.repo.head() if self.repo else None,
                "run_dir": str(self.run_dir), "iter_id": iter_id}

    def _record_remediation(self, intent, accepted, genuine_reject, is_rewrite, is_dd_promo):
        if accepted and not is_rewrite:
            level = intent.kind.split("-to-")[-1]
            self.precision_assignment.append(
                {**intent.target.as_dict(), "precision": level,
                 "rationale_id": intent.rationale_id})
        elif accepted and is_rewrite:
            self.rewrites.append(
                {**intent.target.as_dict(), "kind": intent.kind,
                 "identity": intent.identity, "rationale_id": intent.rationale_id,
                 "accepted": True})
        elif genuine_reject and is_dd_promo:
            # DD retained on the branch as the ceiling candidate (Q2 / P6a)
            self.precision_assignment.append(
                {**intent.target.as_dict(), "precision": "dd",
                 "rationale_id": intent.rationale_id})

    def _finish_walk(self, record, res, walk_digits) -> None:
        self.region_final[record.key] = res.final_precision
        if res.status == "cleared":
            self.region_status[record.key] = "cleared"
        elif res.status == "settled":
            self.region_status[record.key] = "speedup_demoted"
        elif res.status == "dd_ceiling":
            self.region_status[record.key] = "dd_ceiling"
            self.ceiling_regions.append({
                "location": f"{record.integral} {record.target.location}",
                "final_min_digits": walk_digits,
                "signal_class": record.signal_class,
                "ceiling_kind": "dd_ceiling",
                "attempted_rewrites": list(res.attempted_rewrites),
            })
        elif res.status == "dd_untested":
            self.region_status[record.key] = "dd_untested"
            self.ceiling_regions.append({
                "location": f"{record.integral} {record.target.location}",
                "final_min_digits": None,
                "signal_class": record.signal_class,
                "ceiling_kind": "dd_untested",
                "reason": "Patcher failure at the double-to-dd rung (P6a)",
            })
        else:  # exhausted
            self.region_status[record.key] = "unresolved"

    # ------------------------------------------------------------------
    def _check_stops(self) -> bool:
        """Return True if a budget or diminishing-returns stop just fired."""
        b = self.cfg.budget
        if self.budget_iters >= b.max_iters:
            self.stop_status = "budget_exhausted"
            return True
        if (time.monotonic() - self.t0) >= b.max_wall_clock_sec:
            self.stop_status = "budget_exhausted"
            return True
        if self.tokens >= b.max_llm_tokens:
            self.stop_status = "budget_exhausted"
            return True
        if self.cfg.diminishing_returns_k > 0 and self.dr_streak >= self.cfg.diminishing_returns_k:
            self.stop_status = "partial"
            return True
        return False

    def _rationale(self, intent, log_tag: str, accepted: bool) -> str:
        if accepted:
            return f"accepted {intent.kind} at {intent.target.location}"
        if log_tag:
            return f"{intent.kind} at {intent.target.location}: {log_tag}"
        return f"{intent.kind} at {intent.target.location}: reject"

    def _log_extra(self, intent, resp, digits) -> dict:
        extra = {"candidate_sha": resp.get("candidate_sha"),
                 "parent_sha": resp.get("parent_sha")}
        if intent.identity is not None:
            extra["identity"] = intent.identity
        if digits is not None:
            extra["candidate_min_precise_digits"] = digits
        return extra

    # ------------------------------------------------------------------
    def _finalize(self, status: str, meta: dict, n_correctness: int) -> dict:
        self.logger.close()

        # cumulative diff (best-effort; never mask the primary result on git error)
        diff_path = self.run_dir / "final.diff"
        if self.repo and self.starting_sha and status != "internal_error":
            try:
                self.repo.write_cumulative_diff(self.starting_sha, diff_path)
            except Exception as exc:  # noqa: BLE001 - forensic best-effort
                diff_path.write_text(f"# diff unavailable: {exc}\n")
        elif not diff_path.exists():
            diff_path.write_text("")

        dist = {p: 0 for p in LADDER}
        for prec in self.region_final.values():
            dist[prec] = dist.get(prec, 0) + 1

        n_ceiling = sum(1 for c in self.ceiling_regions if c["ceiling_kind"] == "dd_ceiling")
        n_untested = sum(1 for c in self.ceiling_regions if c["ceiling_kind"] == "dd_untested")
        n_unresolved = sum(1 for s in self.region_status.values() if s == "unresolved")
        n_threshold = len(self.region_status) - n_ceiling - n_untested - n_unresolved

        report = {
            "status": status,
            "run_id": self.run_id,
            "final_branch": self.branch,
            "final_working_tree": self.repo.head() if self.repo else None,
            "starting_sha": self.starting_sha,
            "tolerance": self.tolerance,
            "duration_sec": round(time.monotonic() - self.t0, 2),
            "iterations": self.logger._next_id,
            "budget_iters_used": self.budget_iters,
            "llm_tokens_used": self.tokens,
            "correctness_queue_len": n_correctness,
            "precision_assignment": self.precision_assignment,
            "algorithmic_rewrites": self.rewrites,
            "correctness_summary": {
                "regions_at_threshold": n_threshold,
                "regions_at_dd_ceiling": n_ceiling,
                "regions_dd_untested": n_untested,
                "regions_unresolved": n_unresolved,
                "ceiling_regions": self.ceiling_regions,
            },
            "precision_distribution": dist,
            "region_meta": meta,
            "iteration_log_path": str(self.logger.path),
        }
        json_path, md_path = write_reports(self.run_dir, report)

        return {
            "strategy_result": {
                "status": status,
                "run_id": self.run_id,
                "final_branch": self.branch,
                "report_json_path": str(json_path),
                "report_md_path": str(md_path),
                "cumulative_diff_path": str(diff_path),
            }
        }
