"""P1 parsing + P4 pre-checks."""

import pytest

from agents.patcher import result as R
from agents.patcher.agent import make_patcher_fn
from agents.patcher.intent import IntentError, parse_intent, precheck, resolve_in_tree
from agents.strategy.models import ALL_KINDS

from tests.patcher.conftest import intent, ok_gate


def test_parse_all_11_kinds():
    for kind in ALL_KINDS:
        ident = "log1p" if kind == "reformulate-identity" else None
        p = parse_intent(intent(kind, identity=ident))
        assert p.kind == kind


def test_parse_rejects_unknown_kind():
    with pytest.raises(IntentError):
        parse_intent(intent("promote-magic"))


def test_parse_rejects_bad_line_range():
    with pytest.raises(IntentError):
        parse_intent(intent("double-to-dd", line_start=6, line_end=4))


def test_parse_identity_required():
    with pytest.raises(IntentError):
        parse_intent(intent("reformulate-identity", identity=None))


def test_resolve_by_basename(repo):
    root, _ = repo
    assert resolve_in_tree(root, "region.h") is not None
    # bare basename resolves even though the report might key it differently
    assert resolve_in_tree(root, "nope.h") is None


def test_precheck_missing_file(repo, make_ctx):
    root, start = repo
    fn = make_patcher_fn(gate_fn=ok_gate)
    resp = fn(intent("double-to-dd", file="ghost.h"), make_ctx(root, start))
    assert resp["status"] == R.PATCH_APPLY_FAILED
    assert "not found" in resp["error"]["detail"]


def test_precheck_line_out_of_range(repo, make_ctx):
    root, start = repo
    fn = make_patcher_fn(gate_fn=ok_gate)
    resp = fn(intent("double-to-dd", line_start=900, line_end=901), make_ctx(root, start))
    assert resp["status"] == R.PATCH_APPLY_FAILED
    assert "exceeds file length" in resp["error"]["detail"]


def test_precheck_missing_variable(repo, make_ctx):
    root, start = repo
    fn = make_patcher_fn(gate_fn=ok_gate)
    resp = fn(intent("double-to-dd", variables=["nonexistent_var"]), make_ctx(root, start))
    assert resp["status"] == R.PATCH_APPLY_FAILED
    assert "not found in region" in resp["error"]["detail"]


def test_malformed_intent_is_apply_failed(repo, make_ctx):
    root, start = repo
    fn = make_patcher_fn(gate_fn=ok_gate)
    resp = fn({"target": {}, "kind": "double-to-dd", "intent": "correctness"},
              make_ctx(root, start))
    assert resp["status"] == R.PATCH_APPLY_FAILED
