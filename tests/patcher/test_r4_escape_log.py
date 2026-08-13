"""Subtask 4 — R4 #error escape-hatch attribution in patcher_attempts.jsonl.

The retry-budget bump (3→6) absorbs LLM non-determinism on the Rule R4 ``#error``
"requires manual classification" escape hatch.  To measure the per-attempt
variance rate the report needs, a build failure whose build log carries that
#error is tagged ``r4_escape`` (with the escape-hatch ``r4_symbol``) in the
per-attempt log.  This is DIAGNOSTIC ONLY — it never changes dispatch, so a build
failure is still retried exactly as before; these tests assert the tag, not any
behavioural change.
"""

import json
from pathlib import Path

from agents.patcher import gates, result as R
from agents.patcher.agent import (
    MAX_INTEGRATOR_RETRIES, _r4_escape_symbol, make_patcher_fn)

from tests.patcher.conftest import intent, make_shim_integrator, ok_gate


# -- the pure detector -------------------------------------------------------

def test_detects_chain_integrator_escape_and_extracts_symbol():
    log = ('.../shim.hpp:12:2: error: #error "DD Chain Integrator: '
           'ql::Lnrat<DoubleDouble> requires manual classification"')
    assert _r4_escape_symbol(log) == "ql::Lnrat<DoubleDouble>"


def test_detects_any_integrator_flavor():
    for line in (
        '#error "DD Regional Integrator: foo requires manual classification"',
        '#error "FF Regional Integrator: bar requires manual classification"',
        '#error "Float Regional Integrator: baz requires manual classification"',
    ):
        assert _r4_escape_symbol(line) is not None


def test_returns_none_when_no_escape():
    assert _r4_escape_symbol("cmake configure/build failed") is None
    assert _r4_escape_symbol(None) is None
    assert _r4_escape_symbol("error: expected ';' before '}' token") is None


def test_empty_symbol_when_phrase_unparseable():
    # escape phrase present but not in the "<name> requires..." shape → fired but
    # symbol unknown (empty string), still distinct from None (no escape).
    assert _r4_escape_symbol('#error "requires manual classification"') == ""


def test_scans_multiple_texts_detail_then_log():
    # detail is the short gate summary (no #error); the log carries it.
    detail = "cmake configure/build failed"
    build_log = '#error "DD Chain Integrator: q::f<DoubleDouble> requires manual classification"'
    assert _r4_escape_symbol(detail, build_log) == "q::f<DoubleDouble>"


# -- placement in the per-attempt log ---------------------------------------

def _r4_gate(symbol):
    """A gate whose build LOG (not its short detail) carries the R4 #error."""
    def _g(*a, **k):
        logs = Path(k.get("logs_dir") or a[2])
        logs.mkdir(parents=True, exist_ok=True)
        blog = logs / "b.log"
        blog.write_text(
            f'shim.hpp:9:2: error: #error "DD Chain Integrator: {symbol} '
            f'requires manual classification"\n')
        return gates.GateResult(R.BUILD_FAILED, R.ERR_COMPILE,
                                "cmake configure/build failed", blog)
    return _g


def _read_jsonl(run_dir):
    p = Path(run_dir) / "patcher_attempts.jsonl"
    for line in p.read_text().splitlines():
        if line.strip():
            yield json.loads(line)


def test_r4_escape_tagged_in_attempt_log(repo, make_ctx):
    root, start = repo
    fn = make_patcher_fn(gate_fn=_r4_gate("ql::Lnrat<DoubleDouble>"),
                         integrators={"ff": make_shim_integrator(root)})
    ctx = make_ctx(root, start)
    resp = fn(intent("double-to-ff"), ctx)
    assert resp["status"] == R.LLM_GEN_FAILED         # unchanged dispatch behaviour

    log = list(_read_jsonl(ctx["run_dir"]))
    # every attempt hit the R4 escape → tagged, with the symbol, on all of them.
    assert len(log) == MAX_INTEGRATOR_RETRIES
    assert all(r.get("r4_escape") is True for r in log)
    assert all(r["r4_symbol"] == "ql::Lnrat<DoubleDouble>" for r in log)
    assert all(r["outcome"] == "build_failed" for r in log)


def test_non_r4_build_failure_not_tagged(repo, make_ctx):
    # a plain compile error (no escape #error) leaves the tag absent entirely.
    from tests.patcher.conftest import flaky_gate
    root, start = repo
    fn = make_patcher_fn(gate_fn=flaky_gate(MAX_INTEGRATOR_RETRIES),
                         integrators={"ff": make_shim_integrator(root)})
    ctx = make_ctx(root, start)
    fn(intent("double-to-ff"), ctx)
    log = list(_read_jsonl(ctx["run_dir"]))
    assert log                                          # attempts were logged
    assert all("r4_escape" not in r for r in log)
