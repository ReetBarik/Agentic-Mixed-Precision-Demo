"""LangGraph subgraph for the driver generation + compile-iterate skill.

Flow:
    generate_driver
          │
       compile ──── ok ──► run_driver ──► END
          │
    (error, retries left)
          │
       fix_driver
          │
       compile ──► ...
          │
    (error, no retries) ──► END (error)
"""

import os
import time
from typing import Literal

from langgraph.graph import END, StateGraph

from llm_agent import config
from llm_agent.client import make_client
from llm_agent.skills.driver.prompts import (
    GENERATE_DRIVER_TOOL,
    SYSTEM_PROMPT,
    build_compile_error_feedback,
    build_generate_message,
)
from llm_agent.state import DriverState
from llm_agent.tools.build import compile_driver, run_driver


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _call_generate_llm(messages: list) -> dict:
    """Call the LLM with generate_driver tool_choice. Returns raw result dict."""
    try:
        client = make_client(base_url=None)
        response = client.messages.create(
            model=config.DEFAULT_MODEL,
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=[GENERATE_DRIVER_TOOL],
            tool_choice={"type": "tool", "name": "generate_driver"},
            messages=messages,
        )
    except Exception as exc:
        return {"error": "API error: {0}".format(str(exc)[:400])}

    tool_block = next((b for b in response.content if b.type == "tool_use"), None)
    if tool_block is None:
        return {"error": "LLM did not return a tool call"}

    assistant_msg = {
        "role": "assistant",
        "content": [
            {
                "type": "tool_use",
                "id": tool_block.id,
                "name": tool_block.name,
                "input": dict(tool_block.input),
            }
        ],
    }
    return {
        "driver_source": tool_block.input.get("driver_source", ""),
        "cmake_source": tool_block.input.get("cmake_source", ""),
        "tool_use_id": tool_block.id,
        "assistant_msg": assistant_msg,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def generate_driver(state: DriverState) -> dict:
    """First LLM call — generate driver_source + cmake_source from scratch."""
    import sys
    print("[driver] Generating driver for {0} (framework={1})...".format(
        state["signature"]["function_name"],
        state["signature"].get("framework", "none"),
    ), file=sys.stderr)
    messages = [
        build_generate_message(
            sig=state["signature"],
            batch=state["batch"],
            seed=state["seed"],
        )
    ]

    result = _call_generate_llm(messages)
    if result.get("error"):
        return {
            "messages": messages,
            "error": result["error"],
            "driver_source": None,
            "cmake_source": None,
            "_last_tool_use_id": None,
        }

    updated_messages = messages + [result["assistant_msg"]]
    return {
        "driver_source": result["driver_source"],
        "cmake_source": result["cmake_source"],
        "messages": updated_messages,
        "_last_tool_use_id": result["tool_use_id"],
        "error": None,
    }


def compile(state: DriverState) -> dict:
    """Compile the current driver_source + cmake_source."""
    import sys
    print("[driver] Compiling (attempt {0}/{1})...".format(
        state["iteration"] + 1, state["max_iterations"]
    ), file=sys.stderr)
    if state.get("error"):
        return {}

    driver_src = state.get("driver_source") or ""
    cmake_src = state.get("cmake_source") or ""
    if not driver_src or not cmake_src:
        return {"compile_ok": False, "compile_error": "driver or cmake source is empty", "exe_path": None}

    # src_include_dir: directory containing the target header
    root = state["root"]
    sig = state["signature"]
    header_dir = os.path.dirname(os.path.join(root, sig["file_path"]))

    result = compile_driver(
        root=root,
        driver_src=driver_src,
        cmake_src=cmake_src,
        src_include_dir=header_dir,
    )

    import sys
    if result["ok"]:
        print("[driver] Compilation succeeded.", file=sys.stderr)
    else:
        err_msg = result.get("error", "")
        # Print full error on first attempt; subsequent attempts just show summary
        if state["iteration"] == 0:
            print("[driver] Compilation failed:\n{0}".format(err_msg), file=sys.stderr)
        else:
            first_line = err_msg.splitlines()[0] if err_msg else "(unknown)"
            print("[driver] Compilation failed: {0}".format(first_line), file=sys.stderr)
    return {
        "compile_ok": result["ok"],
        "exe_path": result.get("exe_path"),
        "compile_error": result.get("error"),
    }


def fix_driver(state: DriverState) -> dict:
    """Feed compilation error back to LLM and get a revised driver."""
    import sys
    print("[driver] Sending compile error to LLM for revision...", file=sys.stderr)
    tool_use_id = state.get("_last_tool_use_id") or "unknown"
    error_log = state.get("compile_error") or "(unknown error)"
    messages = list(state["messages"])

    feedback = build_compile_error_feedback(tool_use_id, error_log)
    messages = messages + [feedback]

    result = _call_generate_llm(messages)
    new_iteration = state["iteration"] + 1

    if result.get("error"):
        return {
            "messages": messages,
            "error": result["error"],
            "iteration": new_iteration,
            "driver_source": None,
            "cmake_source": None,
            "_last_tool_use_id": None,
        }

    updated_messages = messages + [result["assistant_msg"]]
    return {
        "driver_source": result["driver_source"],
        "cmake_source": result["cmake_source"],
        "messages": updated_messages,
        "_last_tool_use_id": result["tool_use_id"],
        "iteration": new_iteration,
        "error": None,
        "compile_error": None,
        "compile_ok": False,
    }


def run_driver_node(state: DriverState) -> dict:
    """Run the compiled driver and capture the baseline CSV."""
    exe_path = state.get("exe_path")
    if not exe_path:
        return {"run_ok": False, "error": "no exe_path in state"}

    root = state["root"]
    sig = state["signature"]
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(root, "experiments", sig["function_name"], "generated")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(
        out_dir,
        "{name}_baseline_{batch}_{seed}_{ts}.csv".format(
            name=sig["function_name"],
            batch=state["batch"],
            seed=state["seed"],
            ts=ts,
        ),
    )

    result = run_driver(
        exe_path=exe_path,
        out_csv=out_csv,
        batch=state["batch"],
        seed=state["seed"],
    )

    return {
        "run_ok": result["ok"],
        "out_csv": out_csv if result["ok"] else None,
        "error": result.get("error") if not result["ok"] else None,
        "exe_path": None,  # exe was cleaned up by run_driver
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_compile(state: DriverState) -> Literal["run_driver", "fix_driver", "__end__"]:
    if state.get("error"):
        return END
    if state.get("compile_ok"):
        return "run_driver"
    if state["iteration"] >= state["max_iterations"]:
        return END  # error will be set by the compile node's compile_error field
    return "fix_driver"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_driver_graph():
    g = StateGraph(DriverState)

    g.add_node("generate_driver", generate_driver)
    g.add_node("compile", compile)
    g.add_node("fix_driver", fix_driver)
    g.add_node("run_driver", run_driver_node)

    g.set_entry_point("generate_driver")
    g.add_edge("generate_driver", "compile")
    g.add_conditional_edges(
        "compile",
        route_after_compile,
        {"run_driver": "run_driver", "fix_driver": "fix_driver", END: END},
    )
    g.add_edge("fix_driver", "compile")
    g.add_edge("run_driver", END)

    return g.compile()
