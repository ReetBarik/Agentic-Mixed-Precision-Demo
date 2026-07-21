"""Patcher agent — mechanical translation of a remediation intent to a candidate.

Strategy drives the Patcher as a callable (Q5): ``patcher_fn(intent, ctx) -> P2``.
This module builds that callable (:func:`make_patcher_fn`) and implements the
per-intent flow locked in design §P1–§P7:

    parse (P1) → pre-checks (P4) → dispatch (P3) → bounded retry over
    generate + build/smoke gate (P4/P5) → commit on strategy/<run_id> (P2)

On ``ok`` the Patcher has committed the shim(s) + boundary patch (parent = branch
HEAD) and returns the new ``candidate_sha``; on any non-``ok`` status it makes no
commit and resets the working tree back to ``parent_sha`` so the next intent
starts clean (Strategy only resets on a Validator *reject* of an ``ok`` candidate).

``ctx`` (from Strategy's ``_patcher_ctx``) carries: ``run_id``, ``branch``,
``repo_path``, ``parent_sha``, ``run_dir``, ``iter_id`` (and ``retry`` on the P6
timeout re-issue).  Optional injectables — ``integrators`` (``{"ff","dd"}``),
``llm_call``, ``gate_fn``, ``build`` config — let tests drive the paths without a
live LLM or a real compiler; the defaults use the real integrators, the Argo LLM,
and the vanilla-driver build gate.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Callable

from agents.config import PipelineConfig
from agents.patcher import dispatch, gates, gitops, result as R
from agents.patcher.dispatch import PatchDeps
from agents.patcher.intent import IntentError, parse_intent, precheck, resolve_in_tree
from agents.state import PipelineState

MAX_INTEGRATOR_RETRIES = 3      # P4b — single shared budget: integrator + build

# Wave-2 backoff — space out the (unchanged) MAX_INTEGRATOR_RETRIES attempts on a
# retryable llm-driven failure so a transient LLM/service hiccup gets a fresh roll
# instead of a back-to-back re-hit.  This is SPACING ONLY: the retry budget is not
# widened, and a deterministic (non-llm) path never sleeps.  Delay is exponential
# in the (0-based) attempt index — 2s after attempt 0, 4s after attempt 1 — plus a
# small uniform jitter so concurrent Patchers don't retry in lockstep.
BACKOFF_BASE_SEC = 2.0
BACKOFF_JITTER_SEC = 0.5

_REPO = Path(__file__).resolve().parents[2]


def _backoff_delay(attempt: int) -> float:
    """Seconds to wait AFTER a failed ``attempt`` (0-based) before the next one.

    Exponential (``BASE * 2**attempt``) plus uniform jitter in
    ``[0, BACKOFF_JITTER_SEC)``.  Monotonic non-decreasing across attempts even
    with jitter (2.0–2.5 < 4.0–4.5), so the retry cadence always widens.
    """
    return BACKOFF_BASE_SEC * (2 ** attempt) + random.uniform(0.0, BACKOFF_JITTER_SEC)


# ---------------------------------------------------------------------------
# LangGraph node (vestigial — Strategy drives the Patcher as a callable now)
# ---------------------------------------------------------------------------

def run(state: PipelineState) -> dict:
    """No-op graph node.  The remediation loop lives inside Strategy (Q5)."""
    return {}


# ---------------------------------------------------------------------------
# Adapter Strategy injects
# ---------------------------------------------------------------------------

def make_patcher_fn(*, integrators: dict | None = None,
                    llm_call: Callable[[str, str, int], str] | None = None,
                    gate_fn: Callable | None = None,
                    build_config: dict | None = None,
                    config: PipelineConfig | None = None):
    """Build ``patcher_fn(intent, ctx) -> P2`` for Strategy's state.

    All heavy dependencies are injectable; unset ones fall back to the real
    integrators, the Argo LLM (via :mod:`agents.integrator_base.llm`), and the
    real vanilla-driver build gate.
    """
    def patcher_fn(intent: dict, ctx: dict) -> dict:
        return _Patcher(integrators=integrators, llm_call=llm_call,
                        gate_fn=gate_fn, build_config=build_config or {},
                        config=config).patch(intent, ctx)
    return patcher_fn


# ---------------------------------------------------------------------------
# Core
# ---------------------------------------------------------------------------

class _Patcher:
    def __init__(self, *, integrators, llm_call, gate_fn, build_config, config):
        self.integrators = integrators if integrators is not None else _default_integrators()
        self.llm_call = llm_call if llm_call is not None else _default_llm_call(config)
        self.gate_fn = gate_fn if gate_fn is not None else gates.run_gate
        self.build_config = build_config

    # -- entry --------------------------------------------------------------
    def patch(self, intent_dict: dict, ctx: dict) -> dict:
        repo_root = Path(ctx["repo_path"]).resolve()
        parent = ctx.get("parent_sha") or gitops.head(repo_root)

        # ---- P1 parse ----
        try:
            intent = parse_intent(intent_dict)
        except IntentError as exc:
            return R.failure(R.PATCH_APPLY_FAILED, parent,
                             err_kind=R.ERR_APPLY, detail=str(exc))

        # ---- P4 pre-checks ----
        pre_err = precheck(intent, repo_root)
        if pre_err is not None:
            return R.failure(R.PATCH_APPLY_FAILED, parent,
                             err_kind=R.ERR_APPLY, detail=pre_err)

        run_dir = Path(ctx["run_dir"])
        dirs = _run_dirs(run_dir)
        iter_id = ctx.get("iter_id", 0)
        deps = PatchDeps(
            repo_root=repo_root, parent_sha=parent,
            target_path=resolve_in_tree(repo_root, intent.target.file),
            shims_dir=dirs["shims"], patches_dir=dirs["patches"],
            integrators=self.integrators, llm_call=self.llm_call)

        path = dispatch.dispatch_path(intent.kind, intent.via)
        llm_driven = dispatch.is_llm_driven(path)
        attempts = MAX_INTEGRATOR_RETRIES if llm_driven else 1

        # Per-attempt forensic trail (Wave-2 backoff attribution): one JSONL record
        # per LLM-driven attempt, so the report can answer "how many llm_gen_failed
        # regions accepted on retry attempt 2/3?".  Sibling to Strategy's
        # iterations.jsonl in the same run_dir; deterministic paths don't log.
        attempts_log = run_dir / "patcher_attempts.jsonl"

        def _record_attempt(attempt, outcome, status, elapsed, backoff):
            if not llm_driven:
                return
            _append_attempt(attempts_log, {
                "iter_id": iter_id,
                "rationale_id": intent.rationale_id,
                "target": intent.target.location,
                "kind": intent.kind,
                "attempt": attempt,              # 0-based; also the seed-line variation
                "outcome": outcome,
                "status": status,
                "elapsed_sec": round(elapsed, 3),
                "backoff_sec": round(backoff, 3),
            })

        last_detail = None
        for attempt in range(attempts):
            t0 = time.monotonic()
            gitops.reset_hard(repo_root, parent)          # clean slate per attempt
            gen = dispatch.generate(intent, deps, attempt, path)

            if not gen.ok:
                # deterministic gen failure (apply/edit) → terminal, no retry
                if not llm_driven:
                    gitops.reset_hard(repo_root, parent)
                    return R.failure(gen.status, parent, err_kind=gen.err_kind,
                                     detail=gen.detail,
                                     excerpt_path=_excerpt(dirs, iter_id, gen.detail),
                                     llm_tokens=gen.llm_tokens)
                last_detail = gen.detail
                if attempt < attempts - 1:
                    delay = _backoff_delay(attempt)
                    _record_attempt(attempt, "gen_failed", gen.status,
                                    time.monotonic() - t0, delay)
                    time.sleep(delay)
                    continue
                _record_attempt(attempt, "gen_failed", R.LLM_GEN_FAILED,
                                time.monotonic() - t0, 0.0)
                gitops.reset_hard(repo_root, parent)
                return R.failure(R.LLM_GEN_FAILED, parent, err_kind=gen.err_kind,
                                 detail=f"generation failed after {attempts} attempts: {gen.detail}",
                                 excerpt_path=_excerpt(dirs, iter_id, gen.detail),
                                 llm_tokens=gen.llm_tokens)

            # ---- P5 build + smoke gate ----
            gate = self._gate(repo_root, run_dir, iter_id, ctx)
            if gate.ok:
                _record_attempt(attempt, "ok", R.OK, time.monotonic() - t0, 0.0)
                return self._commit(intent, deps, gen, gate, parent, repo_root)

            # timeout is kept standalone so Strategy's P6 timeout-retry can act.
            if gate.status == R.TIMEOUT:
                _record_attempt(attempt, "timeout", R.TIMEOUT,
                                time.monotonic() - t0, 0.0)
                gitops.reset_hard(repo_root, parent)
                return R.failure(R.TIMEOUT, parent, err_kind=R.ERR_TIMEOUT,
                                 detail=gate.detail,
                                 build_log_path=gate.build_log_path,
                                 runtime_log_path=gate.runtime_log_path,
                                 llm_tokens=gen.llm_tokens)

            last_detail = gate.detail
            retryable = dispatch.is_retryable_misgen(gate)
            if llm_driven and retryable and attempt < attempts - 1:
                delay = _backoff_delay(attempt)
                _record_attempt(attempt, "build_failed", gate.status,
                                time.monotonic() - t0, delay)
                time.sleep(delay)
                continue
            if llm_driven and retryable:
                _record_attempt(attempt, "build_failed", R.LLM_GEN_FAILED,
                                time.monotonic() - t0, 0.0)
                # P4a: exhausted retries on a retryable failure → llm_gen_failed
                # (P6a: at the DD rung Strategy treats this as dd_untested, not a
                # physics ceiling).
                gitops.reset_hard(repo_root, parent)
                return R.failure(R.LLM_GEN_FAILED, parent, err_kind=R.ERR_LLM,
                                 detail=f"{gate.status} after {attempts} attempts: {gate.detail}",
                                 build_log_path=gate.build_log_path,
                                 runtime_log_path=gate.runtime_log_path,
                                 llm_tokens=gen.llm_tokens)
            # deterministic path OR non-retryable → Bucket-A status verbatim
            _record_attempt(attempt, "build_failed", gate.status,
                            time.monotonic() - t0, 0.0)
            gitops.reset_hard(repo_root, parent)
            return R.failure(gate.status, parent, err_kind=gate.err_kind,
                             detail=gate.detail,
                             build_log_path=gate.build_log_path,
                             runtime_log_path=gate.runtime_log_path,
                             llm_tokens=gen.llm_tokens)

        # unreachable (loop always returns), but keep a well-formed fallback
        gitops.reset_hard(repo_root, parent)
        return R.failure(R.LLM_GEN_FAILED, parent, err_kind=R.ERR_LLM,
                         detail=last_detail or "exhausted retries")

    # -- gate + commit ------------------------------------------------------
    def _gate(self, repo_root: Path, run_dir: Path, iter_id, ctx) -> gates.GateResult:
        cfg = self.build_config
        headers_dir = _resolve_headers_dir(repo_root, cfg)
        kwargs = {
            "app_cmake_dir": Path(cfg["app_cmake_dir"]) if cfg.get("app_cmake_dir") else gates.DEFAULT_APP_CMAKE_DIR,
            "kokkos_root": Path(cfg["kokkos_root"]) if cfg.get("kokkos_root") else gates.DEFAULT_KOKKOS_ROOT,
        }
        for k in ("expected_rows", "build_timeout", "smoke_timeout"):
            if k in cfg:
                kwargs[k] = cfg[k]
        return self.gate_fn(headers_dir, run_dir / "build", run_dir / "logs",
                            iter_id, **kwargs)

    def _commit(self, intent, deps, gen, gate, parent, repo_root) -> dict:
        try:
            sha = gitops.commit_all(repo_root, _commit_message(intent))
        except gitops.NothingToCommitError as exc:
            # Benign: gen+build succeeded but the candidate == parent (no distinct
            # remediation produced — e.g. a shared shim already covers it).  Advance
            # the walk instead of aborting the run (Strategy: empty_candidate).
            gitops.reset_hard(repo_root, parent)
            return R.failure(R.EMPTY_CANDIDATE, parent, err_kind=R.ERR_EMPTY,
                             detail=str(exc), llm_tokens=gen.llm_tokens)
        except gitops.GitError as exc:
            gitops.reset_hard(repo_root, parent)
            return R.failure(R.COMMIT_FAILED, parent, err_kind=R.ERR_COMMIT,
                             detail=str(exc), llm_tokens=gen.llm_tokens)
        return R.ok(sha, parent, shim_paths=gen.shim_paths,
                    boundary_patch_path=gen.boundary_patch_path,
                    build_log_path=gate.build_log_path,
                    runtime_log_path=gate.runtime_log_path,
                    gate_binary=gate.binary_path,
                    gate_tree_hash=gate.tree_hash,
                    llm_tokens=gen.llm_tokens)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_dirs(run_dir: Path) -> dict:
    dirs = {name: run_dir / name for name in ("shims", "patches", "logs", "errors")}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


def _excerpt(dirs: dict, iter_id, detail: str | None) -> Path | None:
    if not detail:
        return None
    p = dirs["errors"] / f"iter_{iter_id}.txt"
    p.write_text(detail + "\n")
    return p


def _append_attempt(log_path: Path, record: dict) -> None:
    """Append one per-attempt JSON record to ``patcher_attempts.jsonl``.

    Best-effort forensic trail — a logging failure must never break a patch, so
    any I/O error is swallowed (the run continues; only the attribution line is
    lost).
    """
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(log_path, "a", buffering=1) as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        pass


def _commit_message(intent) -> str:
    """Q3 machine-parseable commit message (subject + one field per line)."""
    t = intent.target
    loc = f"{t.file}:{t.line_start}" if t.line_start == t.line_end \
        else f"{t.file}:{t.line_start}-{t.line_end}"
    body = [
        f"[{intent.rationale_id}] {intent.kind} {loc}",
        "",
        f"kind: {intent.kind}",
        f"intent: {intent.intent}",
        f"variables: {', '.join(t.variables)}",
        f"current_precision: {intent.current_precision}",
        f"rationale_id: {intent.rationale_id}",
    ]
    if intent.identity:
        body.append(f"identity: {intent.identity}")
    return "\n".join(body) + "\n"


def _resolve_headers_dir(repo_root: Path, cfg: dict) -> Path:
    """The QL_HEADERS dir (contains boxGPU.h + box/) inside the candidate tree."""
    if cfg.get("headers_dir"):
        return Path(cfg["headers_dir"])
    if (repo_root / "boxGPU.h").is_file():
        return repo_root
    conventional = repo_root / "runs" / "qcdloop_headers_full"
    if (conventional / "boxGPU.h").is_file():
        return conventional
    for cand in sorted(repo_root.rglob("boxGPU.h")):
        if ".git" not in cand.parts:
            return cand.parent
    return repo_root


def _default_integrators() -> dict:
    from agents.dd_integrator import agent as dd_integrator
    from agents.ff_integrator import agent as ff_integrator
    from agents.float_integrator import agent as float_integrator
    return {"ff": ff_integrator.integrate_region,
            "dd": dd_integrator.integrate_region,
            "float": float_integrator.integrate_region}


def _default_llm_call(config: PipelineConfig | None):
    cfg = config or PipelineConfig()

    def llm_call(system: str, user: str, attempt: int) -> str:
        from agents.integrator_base.llm import stream_llm
        # vary the seed line per attempt so a retry doesn't re-roll the same text
        seeded = user if attempt == 0 else f"{user}\n// regeneration attempt {attempt}\n"
        return stream_llm(system, seeded, cfg, max_tokens=4096)

    return llm_call
