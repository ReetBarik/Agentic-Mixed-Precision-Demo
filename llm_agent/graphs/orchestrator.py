"""Main LangGraph orchestrator graph.

Flow:
    load_target
         │
    run_analyze   ──(error)──► aggregate_results ──► END
         │
    run_driver    ──(error)──► aggregate_results
         │
    run_downcast  ──(error)──► aggregate_results
         │
    aggregate_results
         │
        END
"""

import json
import os
import re
import time
from typing import Literal

from langgraph.graph import END, StateGraph

from llm_agent import config
from llm_agent.skills.analyze.graph import build_analyze_graph
from llm_agent.skills.downcast.graph import build_downcast_graph
from llm_agent.skills.driver.graph import build_driver_graph
from llm_agent.state import (
    AnalyzeState,
    DowncastState,
    DriverState,
    OptimizationState,
)
from llm_agent.tools.build import build_and_run

# Compiled subgraphs (module-level singletons)
_analyze_graph = build_analyze_graph()
_driver_graph = build_driver_graph()
_downcast_graph = build_downcast_graph()


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def load_target(state: OptimizationState) -> dict:
    """Validate that the target file exists relative to root."""
    file_path = state["file_path"]
    root = state["root"]
    abs_path = os.path.join(root, file_path)
    if not os.path.isfile(abs_path):
        return {"error": "file not found: {0}".format(abs_path)}
    return {"error": None}


def run_analyze(state: OptimizationState) -> dict:
    """Invoke the analyze subgraph to extract the function signature."""
    if state.get("error"):
        return {}

    abs_path = os.path.join(state["root"], state["file_path"])
    analyze_state = AnalyzeState(
        file_path=abs_path,
        function_name=state["function_name"],
        source="",
        messages=[],
        signature=None,
        iteration=0,
        max_iterations=state.get("max_iterations", config.MAX_ITERATIONS_PER_VAR),
        error=None,
    )

    result = _analyze_graph.invoke(analyze_state)
    sig = result.get("signature")
    err = result.get("error")

    if err or sig is None:
        return {"error": err or "analyze skill returned no signature", "signature": None}

    # Restore the repo-relative path; abs_path was used only for file reading.
    sig["file_path"] = state["file_path"]

    import sys
    print("[orchestrator] Signature extracted:", file=sys.stderr)
    print("  framework:           {0}".format(sig.get("framework")), file=sys.stderr)
    print("  return_type:         {0}".format(sig.get("return_type")), file=sys.stderr)
    print("  input_params:        {0}".format([p["name"] for p in sig.get("input_params", [])]), file=sys.stderr)
    print("  output_params:       {0}".format([p["name"] for p in sig.get("output_params", [])]), file=sys.stderr)
    print("  locals_for_downcast:       {0}".format(sig.get("locals_for_downcast", [])), file=sys.stderr)
    print("  call_expression:           {0}".format(sig.get("call_expression")), file=sys.stderr)
    print("  concrete_template_types:   {0}".format(sig.get("concrete_template_types", {})), file=sys.stderr)

    return {"signature": sig, "error": None}


def run_driver_skill(state: OptimizationState) -> dict:
    """Invoke the driver subgraph to generate, compile, and run a baseline."""
    if state.get("error"):
        return {}

    sig = state["signature"]
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(
        state.get("output_dir") or os.path.join(state["root"], "experiments"),
        sig["function_name"],
        "generated",
    )

    driver_state = DriverState(
        signature=sig,
        root=state["root"],
        batch=state["batch"],
        seed=state["seed"],
        max_iterations=state.get("max_driver_retries", 5),
        driver_source=None,
        cmake_source=None,
        exe_path=None,
        out_csv=None,
        compile_error=None,
        compile_ok=False,
        run_ok=False,
        messages=[],
        iteration=0,
        error=None,
        _last_tool_use_id=None,
    )

    result = _driver_graph.invoke(driver_state)
    err = result.get("error")
    compile_err = result.get("compile_error")
    baseline_csv = result.get("out_csv")

    if err or not baseline_csv:
        detail = err or compile_err or "driver skill produced no baseline CSV"
        return {
            "error": "driver skill failed: {0}".format(detail),
            "baseline_csv": None,
        }

    return {"baseline_csv": baseline_csv, "error": None}


def run_downcast_skill(state: OptimizationState) -> dict:
    """Invoke the downcast subgraph if 'downcast' is in the skills list."""
    if state.get("error"):
        return {}
    if "downcast" not in state.get("skills", []):
        return {}

    sig = state["signature"]
    root = state["root"]
    impl_source = open(os.path.join(root, sig["file_path"]), encoding="utf-8").read()

    # Build a spec dict compatible with render_driver_source / build_and_run.
    return_type = sig.get("return_type", "")
    output_mode = "complex" if "complex" in return_type.lower() else "real"

    inputs = []
    for p in sig.get("input_params", []):
        # Strip qualifiers to get the bare ctype (e.g. "double const&" → "double")
        ctype = re.sub(r"\b(const|volatile)\b", "", p["type"]).replace("&", "").replace("*", "").strip()
        inputs.append({
            "name": p["name"],
            "ctype": ctype,
            "distribution": "uniform_real",
            "min": p.get("domain_min", -4.0),
            "max": p.get("domain_max", 4.0),
        })

    spec_dict = {
        "id":                       sig["function_name"],
        "header_path":              sig["file_path"],
        "function_symbol":          sig["function_name"],
        "framework":                sig.get("framework"),
        "output_mode":              output_mode,
        "return_type":              return_type,
        "inputs":                   inputs or None,
        "call":                     {"expression": sig.get("call_expression", "")},
        "locals_for_downcast":      sig.get("locals_for_downcast", []),
        "concrete_template_types":  sig.get("concrete_template_types") or {},
    }

    # Generate baseline in the same format as downcast candidates (build_and_run /
    # render_driver_source), not the LLM driver's hex CSV.
    ts = time.strftime("%Y%m%d_%H%M%S")
    baseline_dir = os.path.join(root, "experiments", sig["function_name"], "generated")
    os.makedirs(baseline_dir, exist_ok=True)
    downcast_baseline_csv = os.path.join(
        baseline_dir,
        "{0}_downcast_baseline_{1}_{2}_{3}.csv".format(
            sig["function_name"], state["batch"], state["seed"], ts
        ),
    )
    baseline_result = build_and_run(
        root=root,
        spec=spec_dict,
        impl_source=impl_source,
        batch=state["batch"],
        seed=state["seed"],
        out_csv=downcast_baseline_csv,
    )
    if not baseline_result.get("ok"):
        return {
            "error": "failed to generate downcast baseline: {0}".format(
                baseline_result.get("error", "unknown")
            )
        }

    downcast_state = DowncastState(
        spec=spec_dict,
        root=root,
        impl_source=impl_source,
        baseline_csv=downcast_baseline_csv,
        min_digits=state["min_digits"],
        batch=state["batch"],
        seed=state["seed"],
        max_iterations=state["max_iterations"],
        variables=[],
        current_variable=None,
        iteration=0,
        current_proposal=None,
        current_tool_use_id=None,
        policy_reject=None,
        verify_result=None,
        propose_error=None,
        accepted_patches=[],
        accepted_variables=[],
        rejected_variables=[],
        trace=[],
        messages=[],
    )

    result = _downcast_graph.invoke(downcast_state)

    skill_results = dict(state.get("skill_results") or {})
    skill_results["downcast"] = {
        "accepted_variables": result.get("accepted_variables", []),
        "rejected_variables": result.get("rejected_variables", []),
        "accepted_patches":   result.get("accepted_patches", []),
        "trace":              result.get("trace", []),
    }
    return {"skill_results": skill_results}


def aggregate_results(state: OptimizationState) -> dict:
    """Write a summary JSON and return."""
    output_dir = state.get("output_dir") or os.path.join(state["root"], "experiments")
    sig = state.get("signature")
    fn_name = sig["function_name"] if sig else state.get("function_name", "unknown")

    out_dir = os.path.join(output_dir, fn_name, "generated")
    os.makedirs(out_dir, exist_ok=True)

    summary = {
        "function_name": fn_name,
        "file_path":     state.get("file_path"),
        "framework":     sig.get("framework") if sig else None,
        "baseline_csv":  state.get("baseline_csv"),
        "error":         state.get("error"),
        "skill_results": state.get("skill_results") or {},
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(out_dir, "{0}_summary_{1}.json".format(fn_name, ts))
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return {}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_analyze(state: OptimizationState) -> Literal["run_driver", "aggregate_results"]:
    return "aggregate_results" if state.get("error") else "run_driver"


def route_after_driver(state: OptimizationState) -> Literal["run_downcast", "aggregate_results"]:
    return "aggregate_results" if state.get("error") else "run_downcast"


def route_after_downcast(state: OptimizationState) -> Literal["aggregate_results"]:
    return "aggregate_results"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_orchestrator():
    g = StateGraph(OptimizationState)

    g.add_node("load_target", load_target)
    g.add_node("run_analyze", run_analyze)
    g.add_node("run_driver", run_driver_skill)
    g.add_node("run_downcast", run_downcast_skill)
    g.add_node("aggregate_results", aggregate_results)

    g.set_entry_point("load_target")
    g.add_edge("load_target", "run_analyze")
    g.add_conditional_edges(
        "run_analyze",
        route_after_analyze,
        {"run_driver": "run_driver", "aggregate_results": "aggregate_results"},
    )
    g.add_conditional_edges(
        "run_driver",
        route_after_driver,
        {"run_downcast": "run_downcast", "aggregate_results": "aggregate_results"},
    )
    g.add_edge("run_downcast", "aggregate_results")
    g.add_edge("aggregate_results", END)

    return g.compile()
