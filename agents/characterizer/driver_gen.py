"""LLM-driven micro-driver generation.

Sends the InstrumentationSpec to Claude via the Argo proxy and receives a
DriverGenOutput (structured JSON via tool use).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import anthropic

from agents.characterizer.spec import InstrumentationSpec
from agents.config import PipelineConfig


@dataclass
class InteropDecision:
    call_site: str
    strategy: Literal["interop_shim", "opaque_wrap", "inline_reimpl"]
    justification: str


@dataclass
class DriverGenOutput:
    driver_source: str
    interop_decisions: list[InteropDecision]
    inlined_helpers: dict[str, str]
    notes: str


# Tool schema mirrors DriverGenOutput so Claude is forced to return typed JSON
_TOOL_SCHEMA = {
    "name": "emit_driver",
    "description": "Emit the generated micro-driver and per-call interop decisions.",
    "input_schema": {
        "type": "object",
        "required": ["driver_source", "interop_decisions", "inlined_helpers", "notes"],
        "properties": {
            "driver_source": {"type": "string"},
            "interop_decisions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["call_site", "strategy", "justification"],
                    "properties": {
                        "call_site": {"type": "string"},
                        "strategy": {
                            "type": "string",
                            "enum": ["interop_shim", "opaque_wrap", "inline_reimpl"],
                        },
                        "justification": {"type": "string"},
                    },
                },
            },
            "inlined_helpers": {
                "type": "object",
                "additionalProperties": {"type": "string"},
            },
            "notes": {"type": "string"},
        },
    },
}


def generate(
    spec: InstrumentationSpec,
    cfg: PipelineConfig,
    messages: list[dict] | None = None,
) -> tuple[DriverGenOutput, list[dict]]:
    """Call the LLM and return the parsed output plus the full conversation.

    ``messages is None`` (first turn): a single user message is built from the
    spec and prompt template.  Otherwise ``messages`` is sent verbatim — the
    caller is expected to have appended a ``tool_result`` reporting the prior
    build error (see :func:`extend_with_build_error`).

    The returned ``messages`` is the input extended with this turn's assistant
    response, so the caller can hand it straight back on the next retry without
    reaching into the raw API response.
    """

    if messages is None:
        prompt_template = (Path(__file__).parent / "prompts" / "driver_gen.txt").read_text(
            encoding="utf-8"
        )
        user_message = _build_user_message(spec, cfg, prompt_template)
        messages = [{"role": "user", "content": user_message}]

    client = anthropic.Anthropic(
        base_url=cfg.base_url,
        api_key=cfg.auth_token,
    )

    response = client.messages.create(
        model=cfg.model,
        max_tokens=4096,
        tools=[_TOOL_SCHEMA],
        tool_choice={"type": "any"},
        messages=messages,
    )

    # Extend the conversation with this assistant turn (as plain dicts so the
    # history is JSON-serialisable and re-sendable on the next retry).
    assistant_content = [block.model_dump() for block in response.content]
    messages = messages + [{"role": "assistant", "content": assistant_content}]

    tool_input = _extract_tool_input(response)
    return _parse_output(tool_input), messages


def extend_with_build_error(
    messages: list[dict],
    run_result,
    cfg: PipelineConfig,
    attempt: int,
    max_attempts: int,
) -> list[dict]:
    """Append a ``tool_result`` user turn reporting a build/configure failure.

    The Anthropic API requires the turn following an assistant ``tool_use`` to
    open with a ``tool_result`` carrying the same ``tool_use_id``; we extract
    that id from the most recent assistant turn in ``messages``.
    """
    tool_use_id = _last_tool_use_id(messages)
    excerpt = (run_result.stderr or "")[: cfg.retry_stderr_chars]
    feedback = (
        f"Driver failed to {run_result.phase} (attempt {attempt}/{max_attempts}).\n\n"
        f"Error output:\n```\n{excerpt}\n```\n\n"
        "Revise the driver_source to fix this error. Do NOT modify kernel "
        "logic — only fix the driver scaffolding (includes, shim/opaque "
        "wrappers, type instantiations, init/finalize, sampling, journal "
        "flush). Re-emit via emit_driver."
    )
    return messages + [
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": tool_use_id,
                    "content": feedback,
                }
            ],
        }
    ]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_user_message(spec: InstrumentationSpec, cfg: PipelineConfig, prompt_template: str) -> str:
    spec_block = json.dumps(
        {
            "kernel_name": spec.kernel_name,
            "kernel_signature": spec.kernel_signature,
            "parameter_types": spec.parameter_types,
            "parameter_roles": spec.parameter_roles,
            "input_ranges": {k: list(v) for k, v in spec.input_ranges.items()},
            "framework": spec.framework,
            "detected_dispatchers": spec.detected_dispatchers,
            "template_instantiation": spec.template_instantiation,
            "sample_count": spec.sample_count,
            "source_files": spec.source_files,
        },
        indent=2,
    )

    # Embed source file contents so the LLM can see the exact kernel body
    source_sections = []
    for path in spec.source_files:
        try:
            text = Path(path).read_text(encoding="utf-8")
            source_sections.append(f"### `{path}`\n\n```cpp\n{text}\n```")
        except OSError:
            pass
    source_block = "\n\n".join(source_sections)

    strategy_note = ""
    if cfg.strategy_override:
        strategy_note = (
            f"\n\nNOTE: the user has requested --strategy-override {cfg.strategy_override!r}. "
            "Apply this strategy to ALL non-templatable calls unless it would cause a compile error."
        )

    return (
        f"{prompt_template}\n\n"
        f"## Kernel spec\n\n```json\n{spec_block}\n```{strategy_note}\n\n"
        f"## Kernel source\n\n{source_block}"
    )


def _last_tool_use_id(messages: list[dict]) -> str:
    """Return the id of the tool_use block in the most recent assistant turn."""
    for msg in reversed(messages):
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, list):
            for block in content:
                btype = block.get("type") if isinstance(block, dict) else getattr(block, "type", None)
                if btype == "tool_use":
                    bid = block.get("id") if isinstance(block, dict) else getattr(block, "id", None)
                    if bid:
                        return bid
        break  # only the latest assistant turn is relevant
    raise ValueError("no tool_use block found in the most recent assistant turn")


def _extract_tool_input(response) -> dict:
    for block in response.content:
        if block.type == "tool_use" and block.name == "emit_driver":
            return block.input
    # Fallback: try to parse JSON from the text response
    for block in response.content:
        if hasattr(block, "text"):
            text = block.text.strip()
            start = text.find("{")
            if start != -1:
                return json.loads(text[start:])
    raise RuntimeError("LLM response did not call emit_driver tool and contained no JSON")


def _parse_output(raw: dict) -> DriverGenOutput:
    decisions = [
        InteropDecision(
            call_site=d["call_site"],
            strategy=d["strategy"],
            justification=d["justification"],
        )
        for d in raw.get("interop_decisions", [])
    ]
    return DriverGenOutput(
        driver_source=raw["driver_source"],
        interop_decisions=decisions,
        inlined_helpers=raw.get("inlined_helpers", {}),
        notes=raw.get("notes", ""),
    )
