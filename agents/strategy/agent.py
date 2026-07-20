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
from agents.strategy.characterization import load_chains, load_regions
from agents.strategy.dispatch import dispatch
from agents.strategy.gitops import GitRepo
from agents.strategy.iteration_log import IterationLogger
from agents.strategy.models import (
    INTENT_CORRECTNESS, INTENT_SPEEDUP, LADDER, VIA_PLAIN, VIA_REGIONAL,
)
from agents.strategy.ranking import build_queues, error_threshold, load_flop_weights
from agents.strategy.report import write_reports
from agents.strategy.source_probe import region_has_bare_double
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
        self.stop_status: str | None = None   # set → HARD stop (ends the whole run)

        # -- two-phase walk (correctness → speedup) --
        # Phase-1 hitting its cap is a SOFT stop: it ends the correctness phase and
        # hands off to speedup; only a global ceiling (wall/tokens/DR/fatal) or the
        # phase-2 cap is a HARD run stop.  Overall status = phase-2's status.
        self.cap_correctness, self.cap_speedup = self.cfg.budget.phase_caps()
        self.phase = INTENT_CORRECTNESS       # current walk phase
        self.phase_iters = 0                  # counting iters in the current phase
        self.phase_exhausted = False          # soft: phase-1 cap reached
        self.phase_stats = {
            INTENT_CORRECTNESS: {"iterations": 0, "accepts": 0},
            INTENT_SPEEDUP: {"iterations": 0, "accepts": 0},
        }
        self.dd_promoted_keys: set[tuple] = set()   # phase-1 dd promotions (skip in phase 2)

        # -- accumulators for the report --
        self.precision_assignment: list[dict] = []
        self.rewrites: list[dict] = []
        self.ceiling_regions: list[dict] = []
        self.region_final: dict[tuple, str] = {}
        self.region_status: dict[tuple, str] = {}

        # -- Wave-3 report-prune telemetry (never silent: if a prune fires it is
        #    counted here and surfaced in the report's speedup_summary) --
        self.report_prunes = bool(getattr(cfg, "report_prunes", True))
        self.n_skipped_range_unsafe = 0     # WI1: float rung skipped, range-unsafe
        self.n_skipped_pred_float = 0       # WI2: float rung skipped, pred_float > thr
        self.speedup_queue_flop_weighted = False  # WI3: flop-weight table available
        self._flop_weights: dict | None = None    # loaded lazily in execute()

        # -- required_by ledger (cascade-chain per-line precision floor) --
        # keyed by line target key (file, line_start, line_end):
        #   chain_ids   -> the chain_id(s) requiring this line
        #   floor_idx   -> the max ladder index any chain requires (max precision)
        #   rationale   -> the rationale_id that set the floor
        self.line_chain_ids: dict[tuple, set[str]] = {}
        self.line_floor_idx: dict[tuple, int] = {}
        self.line_rationale: dict[tuple, str] = {}
        self._retain_rationale: str | None = None   # set per walk iteration on retain
        self._source_cache: dict = {}   # resolved-path -> file lines (float-rung probe)

    # ------------------------------------------------------------------
    def execute(self) -> dict:
        regions, meta = load_regions(self.report_path)
        chains, chain_meta = load_chains(self.report_path)
        meta.update(chain_meta)
        # WI3: load the flop-weight table (unless the prunes are killed) and rank
        # the speedup queue by flop-weighted throughput.  Missing table → op_count
        # fallback with a warning (handled in load_flop_weights).
        if self.report_prunes:
            self._flop_weights = load_flop_weights(self._ratio_multipliers_path())
            self.speedup_queue_flop_weighted = self._flop_weights is not None
        correctness_q, speedup_q = build_queues(
            regions, self.tolerance, flop_weights=self._flop_weights)
        chain_q = self._rank_chains(chains)

        # every localizable region starts at the double baseline
        for r in regions:
            self.region_final[r.key] = "double"
            self.region_status[r.key] = "baseline_ok"
        for r in correctness_q:
            self.region_status[r.key] = "unworked"
        for c in chain_q:
            self.region_status[c.chain_id] = "unworked"

        if self.repo and self.starting_sha:
            self.repo.create_branch(self.branch, self.starting_sha)

        # == Phase 1 — correctness walk to termination (or its cap) ============
        # tier-1/3/4 regions, then the cascade chains (tier 2's concrete
        # population).  All finish before speedup, so the required_by ledger is
        # fully populated when speedup consults it.
        self.phase = INTENT_CORRECTNESS
        for record in correctness_q:
            if self._phase_over():
                break
            self._process_target(record, INTENT_CORRECTNESS)
        self._process_chains(chain_q)

        # == Phase 2 — speedup walk on the phase-1 accepted state ==============
        # Only a HARD stop (global ceiling / fatal) skips phase 2; a phase-1 cap
        # (soft) hand-off still runs speedup on its reserved + spilled budget.
        if not self.stop_status:
            self._enter_speedup_phase()
            speedup_q2 = [r for r in speedup_q if r.key not in self.dd_promoted_keys]
            for record in speedup_q2:
                if self.stop_status:
                    break
                self._process_target(record, INTENT_SPEEDUP)

        status = self.stop_status or "success"
        return self._finalize(status, meta, len(correctness_q))

    def _phase_over(self) -> bool:
        """True when the current phase must stop — a hard run stop OR the soft
        phase-1 cap hand-off."""
        return self.stop_status is not None or self.phase_exhausted

    def _enter_speedup_phase(self) -> None:
        """Cross the phase boundary: spill unused phase-1 budget into phase 2.

        The phase-2 effective cap = its own knob + whatever counting iterations
        phase 1 left on the table.  Unused speedup budget does NOT spill back
        (phase 1 is already terminated).  Snapshots the phase-1 dd promotions
        (region_final == dd, plus chain-floored lines) so phase 2 skips them.
        """
        spill = max(0, self.cap_correctness - self.phase_iters)
        self.cap_speedup = self.cap_speedup + spill
        self.dd_promoted_keys = self._dd_promoted_keys()
        self.phase = INTENT_SPEEDUP
        self.phase_iters = 0
        self.phase_exhausted = False

    def _dd_promoted_keys(self) -> set[tuple]:
        """Region keys sitting at dd after phase 1 — direct promotions plus lines
        a cascade chain floored at dd (speedup can't move a dd-floored region)."""
        dd_idx = LADDER.index("dd")
        keys = {k for k, p in self.region_final.items() if p == "dd"}
        keys |= {k for k, idx in self.line_floor_idx.items() if idx == dd_idx}
        return keys

    def _rank_chains(self, chains) -> list:
        """Tier-2 population: cascade chains above the tolerance bar, worst
        conditioning first (chain_id breaks ties for determinism)."""
        thr = error_threshold(self.tolerance)
        eligible = [c for c in chains if c.max_rel_err > thr]
        return sorted(eligible, key=lambda c: (-c.max_cond, c.chain_id))

    # ------------------------------------------------------------------
    def _process_target(self, record, mode: str) -> None:
        """Drive one region target's retry walk to termination."""
        floor = self._floor_for(record.key) if mode == INTENT_SPEEDUP else None
        float_via = VIA_PLAIN
        float_ok = True
        if mode == INTENT_SPEEDUP:
            # A region with a bare `double` token reaches float via the Patcher's
            # plain-edit rung (VIA_PLAIN); a template-typed region has no such token
            # and reaches float only via the LLM/regional float integrator
            # (VIA_REGIONAL) — Wave 2 makes that path reachable instead of settling.
            has_bare = region_has_bare_double(
                self.repo_path, record.target.file,
                record.target.line_start, record.target.line_end,
                cache=self._source_cache)
            float_via = VIA_PLAIN if has_bare else VIA_REGIONAL
            float_ok = self._float_rung_ok(record)
        walk = RetryWalk(record, mode, self.tolerance, baseline="double",
                         floor=floor, float_via=float_via, float_ok=float_ok)
        stopped, walk_digits = self._drive_walk(walk, record, chain=None)
        if stopped:
            self.region_final[record.key] = walk.installed
            self.region_status[record.key] = "unresolved"
            return
        self._finish_walk(record, walk.result(), walk_digits)

    def _ratio_multipliers_path(self) -> Path:
        """WI3 weight-table location: the config override, else the qcdloop
        default (``runs/qcdloop/ratio_multipliers.json`` under the repo root)."""
        override = getattr(self.cfg, "ratio_multipliers_path", None)
        if override:
            return Path(override)
        return _REPO / "runs" / "qcdloop" / "ratio_multipliers.json"

    def _float_rung_ok(self, record) -> bool:
        """Wave-3 float-rung admission: both the range guard (WI1) and the
        pred-float error gate (WI2) must pass for the walk to attempt ``->float``.

        Order (per the inventory design): (1) range guard — a range-unsafe region
        settles at ff; (2) error gate — pred_float > 10^-tol settles at ff; (3)
        both pass — float is attempted.  Each skip is counted (never silent).
        The kill-switch (``report_prunes=False``) disables both gates: float stays
        admissible (Wave-2 behavior).
        """
        if not self.report_prunes:
            return True
        if not getattr(record, "value_range_ok_for_float", True):
            self.n_skipped_range_unsafe += 1
            return False
        if record.predicted_rel_err_if_float > error_threshold(self.tolerance):
            self.n_skipped_pred_float += 1
            return False
        return True

    def _process_chains(self, chain_q) -> None:
        """Drive the cascade-chain correctness phase with representative dedup.

        The walk drives on a chain's *representative* line (``lines[0]``); with
        many chains at scale sharing representatives, walking each chain
        independently re-drives the same line dozens of times — wasting budget and
        tripping diminishing-returns before speedup (CALIBRATION.md §Bug 2).  So:

        * **group** chains by representative line and walk each group **once**
          (the highest-ranked chain drives; its result is distributed to the rest);
        * **skip** a group whose representative is already at/above the target
          precision (``dd``) — no walk enqueued.

        Every chain (driver, dedup-sibling, or already-at-target) still flows
        through the required_by ledger and gets a ``region_status`` so accounting /
        telemetry account for all of them; only the driver spends walk iterations.
        """
        dd_idx = LADDER.index("dd")
        groups: dict[tuple, list] = {}
        order: list[tuple] = []
        for chain in chain_q:
            rep_key = chain.lines[0].key
            if rep_key not in groups:
                groups[rep_key] = []
                order.append(rep_key)
            groups[rep_key].append(chain)

        for rep_key in order:
            if self._phase_over():
                break
            group = groups[rep_key]
            # Fix (a): representative already at/above the target — no walk fires.
            if self._line_precision_idx(rep_key) >= dd_idx:
                precision = LADDER[self._line_precision_idx(rep_key)]
                rationale = self.line_rationale.get(rep_key, "")
                for chain in group:
                    self._skip_chain_dedup(chain, precision, rationale)
                continue
            # Fix (b): walk the driver once, distribute its result to the siblings.
            driver = group[0]
            res = self._process_chain(driver)
            if res is None:                     # walk interrupted by a stop
                break
            for chain in group[1:]:
                self._skip_chain_dedup(chain, res.final_precision,
                                       self._retain_rationale or "")

    def _process_chain(self, chain):
        """Drive one cascade chain's correctness walk; distribute the promoted
        precision across every line in the chain via the required_by ledger.

        Returns the terminal :class:`WalkResult` (used to distribute precision to
        dedup-siblings sharing the representative line), or ``None`` if a budget /
        DR stop interrupted the walk mid-flight.
        """
        record = chain.walk_record()
        walk = RetryWalk(record, INTENT_CORRECTNESS, self.tolerance, baseline="double")
        stopped, walk_digits = self._drive_walk(walk, record, chain=chain)
        if stopped:
            return None
        res = walk.result()
        self._finish_chain(chain, res, walk_digits)
        return res

    def _line_precision_idx(self, key: tuple) -> int:
        """Ladder index of the highest precision already assigned to ``key`` —
        max of its baseline/final precision and any cascade-chain floor."""
        idx = self.line_floor_idx.get(key, -1)
        prec = self.region_final.get(key)
        if prec is not None:
            idx = max(idx, LADDER.index(prec))
        return idx

    def _skip_chain_dedup(self, chain, precision: str, rationale: str) -> None:
        """Record a chain whose representative was walked (or already resolved) by
        another chain: distribute the determined precision across this chain's own
        lines (so its unique tail lines still get floored and it appears in every
        line's ``required_by``), then mark it ``chain_dedup_skipped`` — it drove no
        independent walk iteration."""
        if LADDER.index(precision) > LADDER.index("double"):
            for line in chain.lines:
                self._require_line(line.key, chain.chain_id, precision, rationale)
        self.region_status[chain.chain_id] = "chain_dedup_skipped"

    def _drive_walk(self, walk, record, chain) -> tuple[bool, float | None]:
        """Run a walk to termination (or until a stop fires).

        Returns ``(stopped, walk_digits)`` — ``stopped`` True if a budget/DR stop
        interrupted the walk mid-flight (result not available).
        """
        walk_digits: float | None = None
        timed_out_kinds: set[str] = set()
        self._retain_rationale = None

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
                return True, walk_digits

            patcher_ok = status == "ok"
            validator_verdict = None
            verdict_digits = None
            if patcher_ok:
                verdict = self._invoke_validator(resp, iter_id)
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
            self._record_remediation(intent, accepted, genuine_reject, is_rewrite,
                                     is_dd_promo, chain=chain)

            # ---- iteration log ----
            self.logger.write(
                iter_id=iter_id, target=intent.target.as_dict(), kind=intent.kind,
                intent=intent.intent, current_precision=intent.current_precision,
                patcher_status=status, validator_verdict=validator_verdict,
                accepted=accepted, log_tag=entry.log_tag, phase=self.phase,
                rationale=self._rationale(intent, entry.log_tag, accepted),
                strategy_bug=(entry.log_tag == "strategy_bug"),
                extra=self._log_extra(intent, resp, verdict_digits))

            # ---- per-phase accounting (for the report's phase grouping) ----
            self.phase_stats[self.phase]["iterations"] += 1
            if accepted:
                self.phase_stats[self.phase]["accepts"] += 1

            # ---- budget / diminishing-returns accounting ----
            if entry.counts_budget:
                self.budget_iters += 1
                self.phase_iters += 1
            self.tokens += int(resp.get("llm_tokens", 0) or 0)
            if accepted:
                self.dr_streak = 0
            elif entry.log_tag not in ("strategy_bug", "patch_inapplicable"):
                self.dr_streak += 1

            # ---- advance the walk state machine ----
            walk.resolve(accepted=accepted, genuine_reject=genuine_reject)

            # ---- stopping conditions (checked every iteration) ----
            if self._check_stops():
                return True, walk_digits     # walk interrupted mid-flight

        return False, walk_digits

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

    def _invoke_validator(self, resp: dict, iter_id: int) -> dict:
        artifacts = resp.get("artifacts") or {}
        ctx = {"run_id": self.run_id, "branch": self.branch,
               "repo_path": str(self.repo_path) if self.repo_path else None,
               "tolerance": self.tolerance, "snapshot": self.snapshot,
               "iter_id": iter_id,
               # Build-fuse: hand the gate binary + tree hash to the Validator so it
               # reuses the just-built candidate binary instead of rebuilding it.
               "gate_binary": artifacts.get("gate_binary"),
               "gate_tree_hash": artifacts.get("gate_tree_hash")}
        return self.validator_fn(resp.get("candidate_sha"), ctx)

    def _patcher_ctx(self, iter_id: int) -> dict:
        return {"run_id": self.run_id, "branch": self.branch,
                "repo_path": str(self.repo_path) if self.repo_path else None,
                "parent_sha": self.repo.head() if self.repo else None,
                "run_dir": str(self.run_dir), "iter_id": iter_id}

    def _record_remediation(self, intent, accepted, genuine_reject, is_rewrite,
                            is_dd_promo, chain=None):
        retain_precision = accepted and not is_rewrite
        ceiling_retain = genuine_reject and is_dd_promo
        if (retain_precision or ceiling_retain) and chain is not None:
            # A chain's precision is distributed across all its lines from the
            # ledger at _finish_chain — remember which iteration drove it here.
            self._retain_rationale = intent.rationale_id
        elif retain_precision:
            level = intent.kind.split("-to-")[-1]
            self.precision_assignment.append(
                {**intent.target.as_dict(), "precision": level,
                 "required_by": [], "rationale_id": intent.rationale_id,
                 "phase": self.phase})
        elif accepted and is_rewrite:
            self.rewrites.append(
                {**intent.target.as_dict(), "kind": intent.kind,
                 "identity": intent.identity, "rationale_id": intent.rationale_id,
                 "accepted": True, "phase": self.phase})
        elif ceiling_retain:
            # DD retained on the branch as the ceiling candidate (Q2 / P6a)
            self.precision_assignment.append(
                {**intent.target.as_dict(), "precision": "dd",
                 "required_by": [], "rationale_id": intent.rationale_id,
                 "phase": self.phase})

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

    def _finish_chain(self, chain, res, walk_digits) -> None:
        """Distribute a chain's promoted precision across its lines (required_by)."""
        precision = res.final_precision
        rationale = self._retain_rationale or ""
        # register the floor only if the chain was promoted above the baseline
        if LADDER.index(precision) > LADDER.index("double"):
            for line in chain.lines:
                self._require_line(line.key, chain.chain_id, precision, rationale)

        loc = f"{chain.integral} {chain.chain_id}"
        if res.status == "cleared":
            self.region_status[chain.chain_id] = "cleared"
        elif res.status == "dd_ceiling":
            self.region_status[chain.chain_id] = "dd_ceiling"
            self.ceiling_regions.append({
                "location": loc, "final_min_digits": walk_digits,
                "signal_class": chain.signal_class, "ceiling_kind": "dd_ceiling",
                "attempted_rewrites": list(res.attempted_rewrites),
                "chain_id": chain.chain_id})
        elif res.status == "dd_untested":
            self.region_status[chain.chain_id] = "dd_untested"
            self.ceiling_regions.append({
                "location": loc, "final_min_digits": None,
                "signal_class": chain.signal_class, "ceiling_kind": "dd_untested",
                "reason": "Patcher failure at the double-to-dd rung (P6a)",
                "chain_id": chain.chain_id})
        else:  # exhausted
            self.region_status[chain.chain_id] = "unresolved"

    # -- required_by ledger --------------------------------------------
    def _require_line(self, key: tuple, chain_id: str, precision: str,
                      rationale: str) -> None:
        self.line_chain_ids.setdefault(key, set()).add(chain_id)
        idx = LADDER.index(precision)
        if idx > self.line_floor_idx.get(key, -1):     # max precision wins
            self.line_floor_idx[key] = idx
            self.line_rationale[key] = rationale

    def _floor_for(self, key: tuple) -> str | None:
        idx = self.line_floor_idx.get(key)
        return LADDER[idx] if idx is not None else None

    def _emit_chain_assignments(self) -> None:
        """One resolved precision_assignment per chain-claimed line (deterministic).

        Overlap rule: a line in chains X (dd) and Y (ff) gets ONE entry at the max
        precision (dd) with both chain_ids in required_by (design "Overlap rule").
        """
        for key in sorted(self.line_chain_ids):
            file, ls, le = key
            precision = LADDER[self.line_floor_idx[key]]
            self.precision_assignment.append({
                "file": file, "line_start": ls, "line_end": le,
                "precision": precision,
                "required_by": sorted(self.line_chain_ids[key]),
                "rationale_id": self.line_rationale.get(key, ""),
                "phase": INTENT_CORRECTNESS})   # cascade chains are always phase 1
            # reflect the floor in the per-line final precision (distribution)
            cur = self.region_final.get(key)
            if cur is None or LADDER.index(cur) < self.line_floor_idx[key]:
                self.region_final[key] = precision

    # ------------------------------------------------------------------
    def _check_stops(self) -> bool:
        """Return True if a stop just fired (hard run stop OR soft phase-1 cap).

        Global ceilings (wall-clock, tokens, diminishing-returns) are HARD stops
        in either phase.  The per-phase iteration cap is a SOFT hand-off in phase 1
        (``phase_exhausted`` → move to speedup) and a HARD stop in phase 2
        (``budget_exhausted`` — nothing runs after speedup).
        """
        b = self.cfg.budget
        # -- global ceilings (both phases) --
        if (time.monotonic() - self.t0) >= b.max_wall_clock_sec:
            self.stop_status = "budget_exhausted"
            return True
        if self.tokens >= b.max_llm_tokens:
            self.stop_status = "budget_exhausted"
            return True
        if self.cfg.diminishing_returns_k > 0 and self.dr_streak >= self.cfg.diminishing_returns_k:
            self.stop_status = "partial"
            return True
        # -- per-phase iteration cap --
        if self.phase == INTENT_CORRECTNESS:
            if self.phase_iters >= self.cap_correctness:
                self.phase_exhausted = True     # soft: end phase 1, hand off to speedup
                return True
        else:  # speedup phase
            if self.phase_iters >= self.cap_speedup:
                self.stop_status = "budget_exhausted"
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

        # resolve cascade-chain per-line assignments (required_by + overlap) and
        # reflect their precision floor in region_final before the distribution.
        self._emit_chain_assignments()

        dist = {p: 0 for p in LADDER}
        for prec in self.region_final.values():
            dist[prec] = dist.get(prec, 0) + 1

        n_ceiling = sum(1 for c in self.ceiling_regions if c["ceiling_kind"] == "dd_ceiling")
        n_untested = sum(1 for c in self.ceiling_regions if c["ceiling_kind"] == "dd_untested")
        n_unresolved = sum(1 for s in self.region_status.values() if s == "unresolved")
        n_dedup = sum(1 for s in self.region_status.values() if s == "chain_dedup_skipped")
        n_threshold = (len(self.region_status) - n_ceiling - n_untested
                       - n_unresolved - n_dedup)

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
            "phase_summary": {
                "correctness": {
                    **self.phase_stats[INTENT_CORRECTNESS],
                    "iter_cap": self.cap_correctness,
                },
                "speedup": {
                    **self.phase_stats[INTENT_SPEEDUP],
                    "iter_cap": self.cap_speedup,   # effective (incl. phase-1 spill)
                    "skipped_dd_promoted": len(self.dd_promoted_keys),
                },
            },
            "correctness_queue_len": n_correctness,
            "precision_assignment": self.precision_assignment,
            "algorithmic_rewrites": self.rewrites,
            "correctness_summary": {
                "regions_at_threshold": n_threshold,
                "regions_at_dd_ceiling": n_ceiling,
                "regions_dd_untested": n_untested,
                "regions_unresolved": n_unresolved,
                "regions_chain_dedup_skipped": n_dedup,
                "ceiling_regions": self.ceiling_regions,
            },
            # Wave-3 report-prune telemetry (never silent): WI1 range guard, WI2
            # pred-float gate, WI3 flop-weight availability.  Surfaced by
            # runs/qcdloop/analyze_calibration.py without extra wiring.
            "speedup_summary": {
                "report_prunes_enabled": self.report_prunes,
                "regions_skipped_range_unsafe": self.n_skipped_range_unsafe,
                "regions_skipped_pred_float": self.n_skipped_pred_float,
                "speedup_queue_flop_weighted": self.speedup_queue_flop_weighted,
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
