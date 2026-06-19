"""Unit tests for the driver compile-retry loop — no LLM, no real build.

The loop lives in ``agents.characterizer.agent.run``; it calls
``driver_gen.generate`` and ``build_run_agent.build_and_run``, both mocked here
so we exercise pure control flow.  See PLAN_retry_loop.md.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.characterizer import agent as char_agent
from agents.characterizer.agent import _classify_role
from agents.characterizer.driver_gen import (
    DriverGenOutput,
    extend_with_build_error,
)
from agents.build_run.agent import RunResult
from agents.characterizer.spec import InstrumentationSpec
from agents.config import PipelineConfig


# ---------------------------------------------------------------------------
# Helpers / fakes
# ---------------------------------------------------------------------------

def _driver(src: str = "int main(){}") -> DriverGenOutput:
    return DriverGenOutput(
        driver_source=src,
        interop_decisions=[],
        inlined_helpers={},
        notes="n",
    )


def _run_result(phase: str, returncode: int, work_dir: Path, stderr: str = "") -> RunResult:
    journal = work_dir / "journal.jsonl" if phase == "ok" else None
    if journal is not None:
        journal.write_text("{}\n", encoding="utf-8")
    return RunResult(
        returncode=returncode,
        stdout="",
        stderr=stderr,
        journal_path=journal,
        work_dir=work_dir,
        phase=phase,
    )


def _spec() -> InstrumentationSpec:
    return InstrumentationSpec(
        kernel_name="k",
        kernel_signature="void k()",
        parameter_types=[],
        input_ranges={},
        template_instantiation={},
        sample_count=4,
        framework="plain-cpp",
    )


def _fake_assistant_turn(messages, call_index):
    """Mimic driver_gen.generate appending an assistant turn ending in tool_use."""
    if messages is None:
        messages = [{"role": "user", "content": "init"}]
    return messages + [
        {
            "role": "assistant",
            "content": [
                {"type": "tool_use", "id": f"toolu_{call_index}",
                 "name": "emit_driver", "input": {}},
            ],
        }
    ]


@pytest.fixture
def patched(monkeypatch, tmp_path):
    """Patch out the spec build, LLM, log parse, overlay, and emit.

    Returns a small handle the test populates with the sequence of
    build_and_run results; records generate inputs/outputs for assertions.
    """
    state_holder = SimpleNamespace(
        build_results=[],            # filled by the test (list of RunResult)
        generate_inputs=[],          # messages arg seen by each generate call
        generate_calls=0,
        build_calls=0,
    )

    monkeypatch.setattr(char_agent, "_spec_build", lambda *a, **k: _spec())
    monkeypatch.setattr(char_agent, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(char_agent.log_parser, "parse", lambda **k: SimpleNamespace())
    monkeypatch.setattr(char_agent.symbolic_overlay, "analyze", lambda *a, **k: [])

    drivers = SimpleNamespace(seq=[_driver()])  # default one driver; test overrides

    def fake_generate(spec, cfg, messages=None):
        idx = state_holder.generate_calls
        state_holder.generate_inputs.append(messages)
        state_holder.generate_calls += 1
        out_messages = _fake_assistant_turn(messages, idx)
        dr = drivers.seq[min(idx, len(drivers.seq) - 1)]
        return dr, out_messages

    def fake_build_and_run(driver_source, framework, cfg, work_dir=None, **kw):
        i = state_holder.build_calls
        state_holder.build_calls += 1
        return state_holder.build_results[min(i, len(state_holder.build_results) - 1)]

    monkeypatch.setattr(char_agent.driver_gen, "generate", fake_generate)
    monkeypatch.setattr(char_agent.build_run_agent, "build_and_run", fake_build_and_run)

    state_holder.drivers = drivers
    state_holder.tmp_path = tmp_path
    return state_holder


def _state(cfg) -> dict:
    return {
        "config": cfg,
        "kernel_name": "k",
        "source_files": ["dummy.cpp"],
        "input_ranges": {},
    }


# ---------------------------------------------------------------------------
# Loop control flow
# ---------------------------------------------------------------------------

def test_loop_breaks_on_success_first_try(patched):
    tmp = patched.tmp_path
    patched.build_results = [_run_result("ok", 0, tmp)]
    cfg = PipelineConfig(out_dir=tmp, max_driver_attempts=5)

    result = char_agent.run(_state(cfg))

    assert patched.generate_calls == 1
    assert patched.build_calls == 1
    assert "errors" not in result or not result["errors"]
    attempts = list((tmp / "attempts").glob("*_driver.cpp"))
    assert len(attempts) == 1


def test_loop_retries_on_build_failure_then_succeeds(patched):
    tmp = patched.tmp_path
    patched.drivers.seq = [_driver("bad"), _driver("good")]
    patched.build_results = [
        _run_result("build", 1, tmp, stderr="ERROR: undefined reference to foo"),
        _run_result("ok", 0, tmp),
    ]
    cfg = PipelineConfig(out_dir=tmp, max_driver_attempts=5)

    result = char_agent.run(_state(cfg))

    assert patched.generate_calls == 2
    # Second generate call must have received the build error as a tool_result.
    second_messages = patched.generate_inputs[1]
    assert second_messages is not None
    last_turn = second_messages[-1]
    assert last_turn["role"] == "user"
    block = last_turn["content"][0]
    assert block["type"] == "tool_result"
    assert "undefined reference to foo" in block["content"]

    assert "errors" not in result or not result["errors"]

    retry_log = json.loads((tmp / "retry_log.json").read_text())
    assert retry_log["outcome"] == "ok"
    assert retry_log["attempts_used"] == 2
    assert retry_log["attempts"][0]["phase"] == "build"
    assert retry_log["attempts"][1]["phase"] == "ok"


def test_loop_exhausts_attempts(patched):
    tmp = patched.tmp_path
    patched.build_results = [_run_result("build", 1, tmp, stderr="boom")]
    cfg = PipelineConfig(out_dir=tmp, max_driver_attempts=3)

    result = char_agent.run(_state(cfg))

    assert patched.generate_calls == 3
    assert patched.build_calls == 3
    assert result["errors"]
    retry_log = json.loads((tmp / "retry_log.json").read_text())
    assert retry_log["attempts_used"] == 3
    assert retry_log["outcome"] == "build"


def test_loop_does_not_retry_run_failure(patched):
    tmp = patched.tmp_path
    patched.build_results = [_run_result("run", 1, tmp, stderr="segfault")]
    cfg = PipelineConfig(out_dir=tmp, max_driver_attempts=5)

    result = char_agent.run(_state(cfg))

    assert patched.generate_calls == 1
    assert patched.build_calls == 1
    assert result["errors"]


# ---------------------------------------------------------------------------
# extend_with_build_error helper
# ---------------------------------------------------------------------------

def test_extend_with_build_error_extracts_tool_use_id():
    messages = [
        {"role": "user", "content": "init"},
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_xyz", "name": "emit_driver", "input": {}},
        ]},
    ]
    run_result = SimpleNamespace(phase="build", stderr="link error: missing symbol")
    cfg = PipelineConfig(retry_stderr_chars=3000)

    extended = extend_with_build_error(messages, run_result, cfg, attempt=1, max_attempts=5)

    new_turn = extended[-1]
    assert new_turn["role"] == "user"
    block = new_turn["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "toolu_xyz"
    assert "link error: missing symbol" in block["content"]
    # original list not mutated
    assert len(messages) == 2


def test_extend_with_build_error_truncates_stderr():
    messages = [
        {"role": "assistant", "content": [
            {"type": "tool_use", "id": "t1", "name": "emit_driver", "input": {}},
        ]},
    ]
    long_err = "x" * 10000
    run_result = SimpleNamespace(phase="build", stderr=long_err)
    cfg = PipelineConfig(retry_stderr_chars=100)

    extended = extend_with_build_error(messages, run_result, cfg, attempt=2, max_attempts=5)
    content = extended[-1]["content"][0]["content"]
    assert "x" * 100 in content
    assert "x" * 101 not in content


# ---------------------------------------------------------------------------
# _classify_role
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("type_str,name,ranges,expected", [
    ("double", "x", {}, "input"),                 # value type
    ("const T&", "x", {}, "input"),               # const ref always input
    ("const double*", "p", {}, "input"),          # const ptr always input
    ("T&", "out", {}, "output"),                   # mutable ref, no range
    ("T&", "acc", {"acc": (0, 1)}, "inout"),       # mutable ref, has range
    ("double*", "p", {}, "output"),                # mutable ptr, no range
    ("double*", "p", {"p": (0, 1)}, "inout"),      # mutable ptr, has range
])
def test_classify_role(type_str, name, ranges, expected):
    assert _classify_role(type_str, name, ranges) == expected
