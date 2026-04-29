"""LangGraph subgraph for the downcast optimization skill.

Flow:
    init_variables
        │
    pick_variable ──(no more)──► END
        │ (has variable)
    propose
        │
    policy_check
        │
    verify
        │
    record_result
        ├──(accepted)──────────────► pick_variable
        ├──(rejected, retries left)─► propose
        └──(rejected, no retries)──► reject_and_next
                                          │
                                     pick_variable
"""

import os
import time
from typing import Literal

from langgraph.graph import END, StateGraph

from llm_agent import config
from llm_agent.client import make_client
from llm_agent.skills.downcast.prompts import (
    PROPOSE_PATCH_TOOL,
    SYSTEM_PROMPT,
    build_initial_user_message,
    build_tool_result_feedback,
)
from llm_agent.state import AttemptRecord, DowncastState, PatchProposal
from llm_agent.tools.build import apply_patches, build_and_run
from llm_agent.tools.compare import compare


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _block_to_dict(block) -> dict:
    """Convert an Anthropic SDK content block to a plain dict for state storage."""
    if block.type == "tool_use":
        return {
            "type": "tool_use",
            "id": block.id,
            "name": block.name,
            "input": dict(block.input),
        }
    if block.type == "text":
        return {"type": "text", "text": block.text}
    return {"type": block.type}


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def init_variables(state: DowncastState) -> dict:
    variables = list(state["spec"].get("locals_for_downcast", []))
    return {
        "variables": variables,
        "accepted_patches": [],
        "accepted_variables": [],
        "rejected_variables": [],
        "trace": [],
        "current_variable": None,
        "iteration": 0,
        "messages": [],
        "current_proposal": None,
        "current_tool_use_id": None,
        "policy_reject": None,
        "verify_result": None,
        "propose_error": None,
    }


def pick_variable(state: DowncastState) -> dict:
    variables = state["variables"]
    if not variables:
        return {"current_variable": None}
    var = variables[0]
    return {
        "current_variable": var,
        "iteration": 0,
        "messages": [],
        "current_proposal": None,
        "current_tool_use_id": None,
        "policy_reject": None,
        "verify_result": None,
        "propose_error": None,
    }


def propose(state: DowncastState) -> dict:
    """Call the LLM with Anthropic tool use to get a patch proposal."""
    var = state["current_variable"]
    messages = list(state["messages"])

    if not messages:
        # First attempt for this variable: build the initial user turn
        initial_msg = build_initial_user_message(
            spec=state["spec"],
            impl_source=state["impl_source"],
            current_variable=var,
            accepted_patches=state["accepted_patches"],
            accepted_variables=state["accepted_variables"],
            min_digits=state["min_digits"],
        )
        messages = [initial_msg]

    try:
        client = make_client(base_url=state["spec"].get("_base_url"))
        response = client.messages.create(
            model=config.DEFAULT_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            tools=[PROPOSE_PATCH_TOOL],
            tool_choice={"type": "tool", "name": "propose_patch"},
            messages=messages,
        )
    except Exception as exc:
        return {
            "messages": messages,
            "propose_error": "API error: {0}".format(str(exc)[:400]),
            "current_proposal": None,
            "current_tool_use_id": None,
        }

    # Extract the tool_use block
    tool_use_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_use_block is None:
        return {
            "messages": messages,
            "propose_error": "LLM did not return a tool call",
            "current_proposal": None,
            "current_tool_use_id": None,
        }

    proposal = PatchProposal(
        file_path=tool_use_block.input.get("file_path", ""),
        old_line=tool_use_block.input.get("old_line", ""),
        new_line=tool_use_block.input.get("new_line", ""),
        reasoning=tool_use_block.input.get("reasoning", ""),
    )

    # Append the assistant's response to the conversation
    assistant_msg = {
        "role": "assistant",
        "content": [_block_to_dict(b) for b in response.content],
    }
    updated_messages = messages + [assistant_msg]

    return {
        "current_proposal": proposal,
        "current_tool_use_id": tool_use_block.id,
        "messages": updated_messages,
        "propose_error": None,
    }


def policy_check(state: DowncastState) -> dict:
    """Validate the proposal before spending time on a build."""
    if state.get("propose_error"):
        return {"policy_reject": None}  # propose_error is handled by record_result

    proposal = state.get("current_proposal")
    if not proposal:
        return {"policy_reject": "no proposal produced"}

    var = state["current_variable"]
    expected_file = state["spec"].get("header_path", "src/kokkosUtils.h")

    if proposal["file_path"] != expected_file:
        return {
            "policy_reject": "must modify {0}, not {1}".format(
                expected_file, proposal["file_path"]
            )
        }
    if not proposal["old_line"].strip():
        return {"policy_reject": "old_line is empty"}
    if not proposal["new_line"].strip():
        return {"policy_reject": "new_line is empty"}
    if proposal["old_line"] == proposal["new_line"]:
        return {"policy_reject": "old_line and new_line are identical"}

    # Focus variable must appear in the changed lines
    combined = proposal["old_line"] + " " + proposal["new_line"]
    if var not in combined:
        return {
            "policy_reject": "focus variable {0!r} not found in the proposed change".format(var)
        }

    return {"policy_reject": None}


def verify(state: DowncastState) -> dict:
    """Build + run + compare with the full patch stack (accepted + current proposal)."""
    if state.get("policy_reject") or state.get("propose_error"):
        return {"verify_result": None}

    proposal = state.get("current_proposal")
    if not proposal:
        return {"verify_result": {"pass": False, "error": "no proposal to verify"}}

    root = state["root"]
    spec = state["spec"]
    impl_source = state["impl_source"]
    patch_stack = list(state["accepted_patches"]) + [proposal]

    # Apply all patches to get the candidate source
    try:
        patched_source = apply_patches(impl_source, patch_stack)
    except ValueError as exc:
        return {"verify_result": {"pass": False, "error": str(exc), "min_precise_digits": None}}

    # Build and run
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(root, "experiments", spec["id"], "generated")
    out_csv = os.path.join(
        out_dir,
        "{0}_candidate_{1}_{2}.csv".format(spec["id"], state["batch"], ts),
    )
    build_result = build_and_run(
        root=root,
        spec=spec,
        impl_source=patched_source,
        batch=state["batch"],
        seed=state["seed"],
        out_csv=out_csv,
    )
    if not build_result["ok"]:
        logs = build_result.get("logs", {})
        log_detail = next(iter(logs.values()), "") if logs else ""
        error_msg = "{0}:\n{1}".format(build_result["error"], log_detail) if log_detail else build_result["error"]
        return {
            "verify_result": {
                "pass": False,
                "error": error_msg,
                "min_precise_digits": None,
            }
        }

    # Numerical comparison
    cmp = compare(
        baseline_csv=state["baseline_csv"],
        candidate_csv=out_csv,
        min_digits=state["min_digits"],
    )
    return {
        "verify_result": {
            "pass": cmp["pass"],
            "min_precise_digits": cmp["min_precise_digits"],
            "candidate_csv": out_csv,
            "error": None if cmp["min_precise_digits"] is not None else cmp["log"],
        }
    }


def record_result(state: DowncastState) -> dict:
    """Record the attempt outcome and update accepted/rejected sets."""
    var = state["current_variable"]
    iteration = state["iteration"]
    propose_error = state.get("propose_error")
    policy_reject = state.get("policy_reject")
    verify_result = state.get("verify_result") or {}
    verify_pass = verify_result.get("pass", False)

    record = AttemptRecord(
        variable=var,
        iteration=iteration,
        proposal=state.get("current_proposal"),
        policy_reject=policy_reject,
        verify_pass=verify_pass,
        min_precise_digits=verify_result.get("min_precise_digits"),
        error=propose_error or verify_result.get("error"),
    )
    new_trace = list(state["trace"]) + [record]

    if verify_pass:
        # Accept: accumulate patch, remove variable from queue
        new_accepted_patches = list(state["accepted_patches"]) + [state["current_proposal"]]
        new_accepted_vars = list(state["accepted_variables"]) + [var]
        new_variables = state["variables"][1:]
        return {
            "trace": new_trace,
            "accepted_patches": new_accepted_patches,
            "accepted_variables": new_accepted_vars,
            "variables": new_variables,
            "current_variable": None,  # signal: accepted, move on
            "verify_result": None,
            "policy_reject": None,
            "propose_error": None,
        }

    # Rejected — build feedback message for the next LLM call
    new_iteration = iteration + 1
    messages = list(state["messages"])
    tool_use_id = state.get("current_tool_use_id")

    if propose_error:
        # API/format failure: reset the conversation so the next attempt starts fresh
        messages = []
    elif tool_use_id:
        reason = policy_reject or (
            "verify failed (min_precise_digits={0})".format(
                verify_result.get("min_precise_digits")
            )
        )
        feedback = build_tool_result_feedback(tool_use_id, reason, var)
        messages = messages + [feedback]

    return {
        "trace": new_trace,
        "iteration": new_iteration,
        "messages": messages,
        "current_proposal": None,
        "current_tool_use_id": None,
        "policy_reject": None,
        "verify_result": None,
        "propose_error": None,
    }


def reject_and_next(state: DowncastState) -> dict:
    """Give up on the current variable and remove it from the queue."""
    var = state["current_variable"]
    new_rejected = list(state["rejected_variables"]) + [var]
    new_variables = state["variables"][1:]
    return {
        "rejected_variables": new_rejected,
        "variables": new_variables,
        "current_variable": None,
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_pick(state: DowncastState) -> Literal["propose", "__end__"]:
    if state.get("current_variable") is None:
        return END
    return "propose"


def route_after_record(
    state: DowncastState,
) -> Literal["pick_variable", "propose", "reject_and_next"]:
    if state.get("current_variable") is None:
        return "pick_variable"  # accepted → move to next variable
    if state["iteration"] >= state["max_iterations"]:
        return "reject_and_next"
    return "propose"  # retry


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_downcast_graph():
    g = StateGraph(DowncastState)

    g.add_node("init_variables", init_variables)
    g.add_node("pick_variable", pick_variable)
    g.add_node("propose", propose)
    g.add_node("policy_check", policy_check)
    g.add_node("verify", verify)
    g.add_node("record_result", record_result)
    g.add_node("reject_and_next", reject_and_next)

    g.set_entry_point("init_variables")
    g.add_edge("init_variables", "pick_variable")
    g.add_conditional_edges(
        "pick_variable",
        route_after_pick,
        {"propose": "propose", END: END},
    )
    g.add_edge("propose", "policy_check")
    g.add_edge("policy_check", "verify")
    g.add_edge("verify", "record_result")
    g.add_conditional_edges(
        "record_result",
        route_after_record,
        {
            "pick_variable": "pick_variable",
            "propose": "propose",
            "reject_and_next": "reject_and_next",
        },
    )
    g.add_edge("reject_and_next", "pick_variable")

    return g.compile()
