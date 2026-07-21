"""Wave-2 backoff — space out (never widen) the N=3 llm-driven retry attempts.

The autouse ``sleep_calls`` fixture (conftest) records each inter-attempt delay
without actually sleeping, so these assert the backoff SHAPE and PLACEMENT, not
wall time.  The retry budget itself (MAX_INTEGRATOR_RETRIES) is unchanged and is
covered by test_retry.py.
"""

import json

from agents.patcher import result as R
from agents.patcher.agent import (
    BACKOFF_BASE_SEC, BACKOFF_JITTER_SEC, MAX_INTEGRATOR_RETRIES,
    _backoff_delay, make_patcher_fn)

from tests.patcher.conftest import (
    gate_returning, intent, make_shim_integrator, ok_gate)


# -- the pure delay function -------------------------------------------------

def test_backoff_delay_is_exponential_and_positive():
    d0, d1, d2 = (_backoff_delay(a) for a in range(3))
    # every delay is strictly positive (spacing, not a no-op)
    assert d0 > 0 and d1 > 0 and d2 > 0
    # exponential in the attempt index, jitter bounded — so strictly increasing
    # even in the worst-case jitter overlap (2.0–2.5 < 4.0–4.5 < 8.0–8.5)
    assert d0 < d1 < d2
    # base + jitter envelope
    assert BACKOFF_BASE_SEC <= d0 < BACKOFF_BASE_SEC + BACKOFF_JITTER_SEC
    assert 2 * BACKOFF_BASE_SEC <= d1 < 2 * BACKOFF_BASE_SEC + BACKOFF_JITTER_SEC


def test_backoff_delays_non_decreasing_across_full_budget():
    delays = [_backoff_delay(a) for a in range(MAX_INTEGRATOR_RETRIES)]
    assert delays == sorted(delays)
    assert all(d > 0 for d in delays)


# -- placement in the retry loop --------------------------------------------

def test_sleep_between_attempts_on_eventual_success(repo, make_ctx, sleep_calls):
    # fails MAX-1 times then succeeds on the final attempt → a sleep after each
    # failed attempt, i.e. attempts-1 sleeps, none after the winning attempt.
    root, start = repo
    integ = make_shim_integrator(root, fail_times=MAX_INTEGRATOR_RETRIES - 1)
    fn = make_patcher_fn(gate_fn=ok_gate, integrators={"ff": integ})
    resp = fn(intent("double-to-ff"), make_ctx(root, start))
    assert resp["status"] == R.OK
    assert len(sleep_calls) == MAX_INTEGRATOR_RETRIES - 1
    assert sleep_calls == sorted(sleep_calls)      # non-decreasing
    assert all(d > 0 for d in sleep_calls)


def test_no_sleep_on_first_attempt_success(repo, make_ctx, sleep_calls):
    root, start = repo
    fn = make_patcher_fn(gate_fn=ok_gate,
                         integrators={"ff": make_shim_integrator(root)})
    resp = fn(intent("double-to-ff"), make_ctx(root, start))
    assert resp["status"] == R.OK
    assert sleep_calls == []                        # never slept


def test_no_sleep_after_final_attempt_when_budget_exhausted(repo, make_ctx,
                                                            sleep_calls):
    # fails all N attempts → sleeps only BETWEEN attempts (attempts-1), not after
    # the last one (no next attempt to space out).
    root, start = repo
    integ = make_shim_integrator(root, fail_times=MAX_INTEGRATOR_RETRIES)
    fn = make_patcher_fn(gate_fn=ok_gate, integrators={"ff": integ})
    resp = fn(intent("double-to-ff"), make_ctx(root, start))
    assert resp["status"] == R.LLM_GEN_FAILED
    assert len(sleep_calls) == MAX_INTEGRATOR_RETRIES - 1


def test_no_sleep_on_deterministic_failure(repo, make_ctx, sleep_calls):
    # a plain-type-edit (double-to-float, non-llm-driven) build failure is a real
    # signal, not a retryable misgen — single attempt, no backoff.
    root, start = repo
    fn = make_patcher_fn(gate_fn=gate_returning(R.BUILD_FAILED, R.ERR_COMPILE))
    resp = fn(intent("double-to-float", line_start=4, line_end=6, flavor="speedup"),
              make_ctx(root, start))
    assert resp["status"] == R.BUILD_FAILED
    assert sleep_calls == []                        # deterministic path never sleeps


# -- per-attempt forensic log ------------------------------------------------

def test_attempt_log_records_each_llm_attempt(repo, make_ctx, tmp_path):
    root, start = repo
    integ = make_shim_integrator(root, fail_times=MAX_INTEGRATOR_RETRIES - 1)
    fn = make_patcher_fn(gate_fn=ok_gate, integrators={"ff": integ})
    ctx = make_ctx(root, start)
    resp = fn(intent("double-to-ff"), ctx)
    assert resp["status"] == R.OK

    log = list((_read_jsonl(ctx["run_dir"])))
    # one record per attempt: N-1 gen_failed then a final ok
    assert len(log) == MAX_INTEGRATOR_RETRIES
    assert [r["attempt"] for r in log] == list(range(MAX_INTEGRATOR_RETRIES))
    assert [r["outcome"] for r in log[:-1]] == ["gen_failed"] * (MAX_INTEGRATOR_RETRIES - 1)
    assert log[-1]["outcome"] == "ok"
    assert log[-1]["status"] == R.OK
    # winning record is a late attempt → this is the "accepted on retry 2/3" signal
    assert log[-1]["attempt"] == MAX_INTEGRATOR_RETRIES - 1
    # backoff recorded on the failed (spaced) attempts, zero on the terminal one
    assert all(r["backoff_sec"] > 0 for r in log[:-1])
    assert log[-1]["backoff_sec"] == 0.0


def test_deterministic_path_writes_no_attempt_log(repo, make_ctx):
    root, start = repo
    fn = make_patcher_fn(gate_fn=gate_returning(R.BUILD_FAILED, R.ERR_COMPILE))
    ctx = make_ctx(root, start)
    fn(intent("double-to-float", line_start=4, line_end=6, flavor="speedup"), ctx)
    from pathlib import Path
    assert not (Path(ctx["run_dir"]) / "patcher_attempts.jsonl").exists()


def _read_jsonl(run_dir):
    from pathlib import Path
    p = Path(run_dir) / "patcher_attempts.jsonl"
    for line in p.read_text().splitlines():
        if line.strip():
            yield json.loads(line)
