"""P4 bounded retry — N=3 shared budget over integrator + build."""

from agents.patcher import result as R
from agents.patcher.agent import MAX_INTEGRATOR_RETRIES, make_patcher_fn

from tests.patcher.conftest import (
    flaky_gate, gate_returning, intent, make_shim_integrator, ok_gate)


def test_integrator_recovers_within_budget(repo, make_ctx):
    root, start = repo
    integ = make_shim_integrator(root, fail_times=MAX_INTEGRATOR_RETRIES - 1)
    fn = make_patcher_fn(gate_fn=ok_gate, integrators={"ff": integ})
    resp = fn(intent("double-to-ff"), make_ctx(root, start))
    assert resp["status"] == R.OK


def test_build_recovers_within_budget(repo, make_ctx):
    root, start = repo
    fn = make_patcher_fn(gate_fn=flaky_gate(MAX_INTEGRATOR_RETRIES - 1),
                         integrators={"ff": make_shim_integrator(root)})
    resp = fn(intent("double-to-ff"), make_ctx(root, start))
    assert resp["status"] == R.OK


def test_integrator_exhausts_budget(repo, make_ctx):
    root, start = repo
    integ = make_shim_integrator(root, fail_times=MAX_INTEGRATOR_RETRIES)
    fn = make_patcher_fn(gate_fn=ok_gate, integrators={"ff": integ})
    resp = fn(intent("double-to-ff"), make_ctx(root, start))
    assert resp["status"] == R.LLM_GEN_FAILED
    assert f"{MAX_INTEGRATOR_RETRIES} attempts" in resp["error"]["detail"]


def test_build_exhausts_budget_folds_to_llm_gen_failed(repo, make_ctx):
    # P4a: with is_retryable_misgen == True, a persistent build failure on an
    # llm-driven path folds to llm_gen_failed after N (design pseudocode).
    root, start = repo
    fn = make_patcher_fn(gate_fn=flaky_gate(MAX_INTEGRATOR_RETRIES),
                         integrators={"ff": make_shim_integrator(root)})
    resp = fn(intent("double-to-ff"), make_ctx(root, start))
    assert resp["status"] == R.LLM_GEN_FAILED


def test_deterministic_path_build_failure_is_bucket_a(repo, make_ctx):
    # plain-type-edit is deterministic — a build failure is a real signal about
    # the intent (build_failed / Bucket A), not a retryable misgen.
    root, start = repo
    fn = make_patcher_fn(gate_fn=gate_returning(R.BUILD_FAILED, R.ERR_COMPILE))
    resp = fn(intent("double-to-float", line_start=4, line_end=6, flavor="speedup"),
              make_ctx(root, start))
    assert resp["status"] == R.BUILD_FAILED
    assert resp["error"]["kind"] == R.ERR_COMPILE


def test_timeout_returned_immediately(repo, make_ctx):
    # timeout is kept standalone so Strategy's P6 retry-once can act; the Patcher
    # does not fold it into its own retry loop.
    root, start = repo
    calls = {"n": 0}

    def counting_timeout(*a, **k):
        calls["n"] += 1
        return gate_returning(R.TIMEOUT, R.ERR_TIMEOUT)(*a, **k)

    fn = make_patcher_fn(gate_fn=counting_timeout,
                         integrators={"ff": make_shim_integrator(root)})
    resp = fn(intent("double-to-ff"), make_ctx(root, start))
    assert resp["status"] == R.TIMEOUT
    assert calls["n"] == 1          # not retried internally


def test_failure_resets_working_tree(repo, make_ctx):
    # after any non-ok outcome the tree is back at parent_sha (clean) so the next
    # intent starts fresh.
    root, start = repo
    fn = make_patcher_fn(gate_fn=gate_returning(R.RUNTIME_NAN, R.ERR_NAN),
                         integrators={"ff": make_shim_integrator(root)})
    resp = fn(intent("double-to-ff"), make_ctx(root, start))
    assert resp["status"] == R.LLM_GEN_FAILED  # nan on llm path folds after retries
    head = git_head(root)
    assert head == start
    assert not (root / "region_ff.h").exists()


def git_head(root):
    from tests.patcher.conftest import git
    return git(root, "rev-parse", "HEAD").stdout.strip()
