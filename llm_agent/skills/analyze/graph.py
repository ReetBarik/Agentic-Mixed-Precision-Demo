"""LangGraph subgraph for the signature analysis skill.

Flow:
    read_source
        │
    extract_signature ──(error/retry)──► extract_signature
        │ (ok)
    validate
        │
        ├──(ok)──────────────────────────► END
        ├──(invalid, retries left)────────► extract_signature
        └──(invalid, no retries)──────────► END (error set)
"""

import os
from typing import Literal

from langgraph.graph import END, StateGraph

from llm_agent import config
from llm_agent.client import make_client
from llm_agent.skills.analyze.prompts import (
    EXTRACT_SIGNATURE_TOOL,
    SYSTEM_PROMPT,
    build_extract_message,
    build_rejection_feedback,
)
from llm_agent.state import AnalyzeState, FunctionParam, FunctionSignature


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def read_source(state: AnalyzeState) -> dict:
    """Read the C++ source file into state."""
    file_path = state["file_path"]
    try:
        with open(file_path) as f:
            source = f.read()
    except OSError as exc:
        return {"source": "", "error": "cannot read {0}: {1}".format(file_path, exc)}
    return {"source": source, "error": None}


def extract_signature(state: AnalyzeState) -> dict:
    """Call the LLM to extract the function signature via tool use."""
    import sys
    print("[analyze] Extracting signature for {0} (attempt {1}/{2})...".format(
        state["function_name"], state["iteration"] + 1, state["max_iterations"]
    ), file=sys.stderr)
    if state.get("error"):
        return {}  # file read failed; propagate error

    messages = list(state["messages"])
    if not messages:
        messages = [
            build_extract_message(
                file_path=state["file_path"],
                function_name=state["function_name"],
                source=state["source"],
            )
        ]

    try:
        client = make_client(base_url=None)
        response = client.messages.create(
            model=config.DEFAULT_MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=[EXTRACT_SIGNATURE_TOOL],
            tool_choice={"type": "tool", "name": "extract_signature"},
            messages=messages,
        )
    except Exception as exc:
        return {
            "messages": messages,
            "error": "API error: {0}".format(str(exc)[:400]),
            "signature": None,
        }

    tool_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_block is None:
        return {
            "messages": messages,
            "error": "LLM did not return a tool call",
            "signature": None,
        }

    inp = tool_block.input

    def _make_param(p: dict, is_output: bool = False) -> FunctionParam:
        return FunctionParam(
            name=p["name"],
            type=p["type"],
            is_const=p.get("is_const", False),
            is_ref=p.get("is_ref", False),
            is_output=is_output,
            domain_min=p.get("domain_min"),
            domain_max=p.get("domain_max"),
        )

    sig = FunctionSignature(
        function_name=state["function_name"],
        file_path=state["file_path"],
        namespace=inp.get("namespace") or None,
        framework=inp.get("framework", "none"),
        return_type=inp.get("return_type", ""),
        is_template=inp.get("is_template", False),
        template_params=inp.get("template_params", []),
        input_params=[_make_param(p) for p in inp.get("input_params", [])],
        output_params=[_make_param(p, is_output=True) for p in inp.get("output_params", [])],
        call_expression=inp.get("call_expression", ""),
        locals_for_downcast=inp.get("locals_for_downcast", []),
        concrete_template_types=inp.get("concrete_template_types", {}),
    )

    assistant_msg = {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": tool_block.id, "name": tool_block.name, "input": dict(tool_block.input)}
        ],
    }
    return {
        "signature": sig,
        "messages": messages + [assistant_msg],
        "_last_tool_use_id": tool_block.id,
        "error": None,
    }


def validate(state: AnalyzeState) -> dict:
    """Validate the extracted signature; on failure build rejection feedback."""
    if state.get("error"):
        return {}

    sig = state.get("signature")
    tool_use_id = state.get("_last_tool_use_id")
    iteration = state["iteration"]

    errors = []
    if not sig:
        errors.append("no signature extracted")
    else:
        if not sig.get("return_type"):
            errors.append("return_type is empty")
        if not sig.get("call_expression"):
            errors.append("call_expression is empty")
        if not sig.get("input_params") and not sig.get("output_params"):
            errors.append("no input or output params found")

    if not errors:
        return {}  # ok — routing will send to END

    reason = "; ".join(errors)
    if iteration + 1 >= state["max_iterations"]:
        return {"error": "signature validation failed after {0} attempts: {1}".format(
            state["max_iterations"], reason
        )}

    # Build rejection feedback and increment iteration for retry
    feedback = build_rejection_feedback(tool_use_id or "unknown", reason)
    return {
        "messages": list(state["messages"]) + [feedback],
        "iteration": iteration + 1,
        "signature": None,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_validate(state: AnalyzeState) -> Literal["extract_signature", "__end__"]:
    if state.get("error"):
        return END
    if state.get("signature") is None:
        return "extract_signature"  # retry
    return END


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_analyze_graph():
    g = StateGraph(AnalyzeState)

    g.add_node("read_source", read_source)
    g.add_node("extract_signature", extract_signature)
    g.add_node("validate", validate)

    g.set_entry_point("read_source")
    g.add_edge("read_source", "extract_signature")
    g.add_edge("extract_signature", "validate")
    g.add_conditional_edges(
        "validate",
        route_after_validate,
        {"extract_signature": "extract_signature", END: END},
    )

    return g.compile()
