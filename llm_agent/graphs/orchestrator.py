"""Main LangGraph orchestrator graph.

Flow:
    load_target
         │
    run_analyze   ──(error)──► aggregate_results ──► END
         │
    run_driver    ──(error)──► aggregate_results
         │
    init_candidate_loop ──(error / no candidates)──► aggregate_results
         │
    pick_candidate ──(no candidates left)──► maybe_requeue ──(done)──► aggregate_results
         │                                        │
         │                                   pick_candidate
         ▼
    invoke_proposer
         │
    policy_check
         │
    build_and_verify
         │
    record_outcome ──(accept or defer: current_variable=None)──► pick_candidate
                   ──(retry: current_variable still set)──► invoke_proposer
"""

import json
import os
import re
import sys
import time
from typing import Literal

from langgraph.graph import END, StateGraph

from llm_agent import config
from llm_agent.skills.downcast.graph import build_downcast_graph
from llm_agent.skills.downcast.prompts import build_tool_result_feedback
from llm_agent.skills.analyze.graph import build_analyze_graph
from llm_agent.skills.driver.graph import build_driver_graph
from llm_agent.state import (
    AnalyzeState,
    DowncastProposerState,
    DriverState,
    OptimizationState,
)
from llm_agent.tools.build import apply_patches, build_and_run
from llm_agent.tools.compare import compare
from llm_agent.tools.spec_revise import build_and_run_with_revision

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
        _last_tool_use_id=None,
    )

    result = _analyze_graph.invoke(analyze_state)
    sig = result.get("signature")
    err = result.get("error")

    if err or sig is None:
        return {"error": err or "analyze skill returned no signature", "signature": None}

    # Restore the repo-relative path; abs_path was used only for file reading.
    sig["file_path"] = state["file_path"]

    print("[orchestrator] Signature extracted:", file=sys.stderr)
    print("  framework:           {0}".format(sig.get("framework")), file=sys.stderr)
    print("  return_type:         {0}".format(sig.get("return_type")), file=sys.stderr)
    print("  input_params:        {0}".format([p["name"] for p in sig.get("input_params", [])]), file=sys.stderr)
    print("  output_params:       {0}".format([p["name"] for p in sig.get("output_params", [])]), file=sys.stderr)
    print("  locals_for_downcast: {0}".format(sig.get("locals_for_downcast", [])), file=sys.stderr)
    print("  call_expression:     {0}".format(sig.get("call_expression")), file=sys.stderr)
    print("  concrete_template_types: {0}".format(sig.get("concrete_template_types", {})), file=sys.stderr)

    return {"signature": sig, "error": None}


def run_driver_skill(state: OptimizationState) -> dict:
    """Invoke the driver subgraph to generate, compile, and run a baseline."""
    if state.get("error"):
        return {}

    sig = state["signature"]
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


def _build_spec_dict(sig: dict) -> dict:
    """Build the spec dict used by render_driver_source / build_and_run."""
    def _strip_ctype(raw: str) -> str:
        return re.sub(r"\b(const|volatile)\b", "", raw).replace("&", "").replace("*", "").strip()

    inputs = [
        {
            "name": p["name"],
            "ctype": _strip_ctype(p["type"]),
            "distribution": "uniform_real",
            "min": p.get("domain_min", -4.0),
            "max": p.get("domain_max", 4.0),
        }
        for p in sig.get("input_params", [])
    ]
    outputs = [
        {"name": p["name"], "ctype": _strip_ctype(p["type"])}
        for p in sig.get("output_params", [])
    ]
    return {
        "id":                      sig["function_name"],
        "header_path":             sig["file_path"],
        "function_symbol":         sig["function_name"],
        "framework":               sig.get("framework"),
        "return_type":             sig.get("return_type", ""),
        "inputs":                  inputs or None,
        "outputs":                 outputs,
        "call":                    {"expression": sig.get("call_expression", "")},
        "locals_for_downcast":     sig.get("locals_for_downcast", []),
        "concrete_template_types": sig.get("concrete_template_types") or {},
    }


def init_candidate_loop(state: OptimizationState) -> dict:
    """Load the target file, build the spec, run the baseline, and seed the candidate queue."""
    if state.get("error"):
        return {}
    if "downcast" not in state.get("skills", []):
        return {"candidates": []}

    sig = state["signature"]
    root = state["root"]
    impl_source = open(os.path.join(root, sig["file_path"]), encoding="utf-8").read()
    spec_dict = _build_spec_dict(sig)

    ts = time.strftime("%Y%m%d_%H%M%S")
    baseline_dir = os.path.join(
        state.get("output_dir") or os.path.join(root, "experiments"),
        sig["function_name"],
        "generated",
    )
    os.makedirs(baseline_dir, exist_ok=True)
    downcast_baseline_csv = os.path.join(
        baseline_dir,
        "{0}_downcast_baseline_{1}_{2}_{3}.csv".format(
            sig["function_name"], state["batch"], state["seed"], ts
        ),
    )

    baseline_result, spec_dict = build_and_run_with_revision(
        root=root,
        spec=spec_dict,
        impl_source=impl_source,
        batch=state["batch"],
        seed=state["seed"],
        out_csv=downcast_baseline_csv,
        max_attempts=state.get("max_driver_retries", 3),
        base_url=state.get("base_url"),
    )
    if not baseline_result.get("ok"):
        return {
            "error": "failed to generate downcast baseline: {0}".format(
                baseline_result.get("error", "unknown")
            )
        }

    candidates = list(sig.get("locals_for_downcast", []))
    print(
        "[orchestrator] Candidate variables: {0}".format(candidates),
        file=sys.stderr,
    )

    return {
        "impl_source": impl_source,
        "spec": spec_dict,
        "downcast_baseline_csv": downcast_baseline_csv,
        "candidates": candidates,
        "deferred": [],
        "patch_set": [],
        "accepted_variables": [],
        "trace": [],
        "requeue_cycles": 0,
    }


def pick_candidate(state: OptimizationState) -> dict:
    """Pop the next variable from the candidates queue and reset per-variable state."""
    candidates = list(state.get("candidates") or [])
    if not candidates:
        return {"current_variable": None}
    var = candidates[0]
    print("[orchestrator] Picking candidate: {0}".format(var), file=sys.stderr)
    return {
        "current_variable": var,
        "candidates": candidates[1:],
        "iteration": 0,
        "proposer_messages": [],
        "current_proposal": None,
        "current_tool_use_id": None,
        "propose_error": None,
        "policy_reject": None,
        "verify_result": None,
    }


def invoke_proposer(state: OptimizationState) -> dict:
    """Invoke the downcast proposer subgraph for the current variable."""
    var = state["current_variable"]
    print(
        "[orchestrator] Invoking proposer for {0!r} (iteration {1})".format(
            var, state.get("iteration", 0)
        ),
        file=sys.stderr,
    )

    proposer_state = DowncastProposerState(
        spec=state["spec"],
        impl_source=state["impl_source"],
        accepted_patches=list(state.get("patch_set") or []),
        accepted_variables=list(state.get("accepted_variables") or []),
        current_variable=var,
        iteration=state.get("iteration", 0),
        max_iterations=state["max_iterations"],
        min_digits=state["min_digits"],
        base_url=state.get("base_url"),
        messages=list(state.get("proposer_messages") or []),
        proposal=None,
        current_tool_use_id=None,
        propose_error=None,
    )

    result = _downcast_graph.invoke(proposer_state)

    return {
        "current_proposal":    result.get("proposal"),
        "current_tool_use_id": result.get("current_tool_use_id"),
        "propose_error":       result.get("propose_error"),
        "proposer_messages":   result.get("messages") or [],
    }


def policy_check(state: OptimizationState) -> dict:
    """Validate the proposal before spending time on a build."""
    if state.get("propose_error"):
        return {"policy_reject": None}

    proposal = state.get("current_proposal")
    if not proposal:
        return {"policy_reject": "no proposal produced"}

    var = state["current_variable"]
    spec = state.get("spec") or {}
    expected_file = spec.get("header_path")
    if not expected_file:
        return {"policy_reject": "spec missing header_path"}

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

    combined = proposal["old_line"] + " " + proposal["new_line"]
    if var not in combined:
        return {
            "policy_reject": "focus variable {0!r} not found in the proposed change".format(var)
        }

    return {"policy_reject": None}


def build_and_verify(state: OptimizationState) -> dict:
    """Build the cumulative patched kernel and verify numerically against the baseline."""
    if state.get("policy_reject") or state.get("propose_error"):
        return {"verify_result": None}

    proposal = state.get("current_proposal")
    if not proposal:
        return {"verify_result": {"pass": False, "error": "no proposal to verify"}}

    root = state["root"]
    spec = state["spec"]
    patch_stack = list(state.get("patch_set") or []) + [proposal]

    try:
        patched_source = apply_patches(state["impl_source"], patch_stack)
    except ValueError as exc:
        return {"verify_result": {"pass": False, "error": str(exc), "min_precise_digits": None}}

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(
        state.get("output_dir") or os.path.join(root, "experiments"),
        spec["id"],
        "generated",
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
        error_msg = (
            "{0}:\n{1}".format(build_result["error"], log_detail)
            if log_detail
            else build_result["error"]
        )
        return {"verify_result": {"pass": False, "error": error_msg, "min_precise_digits": None}}

    cmp = compare(
        baseline_csv=state["downcast_baseline_csv"],
        candidate_csv=out_csv,
        min_digits=state["min_digits"],
    )
    return {
        "verify_result": {
            "pass": cmp["pass"],
            "min_precise_digits": cmp["min_precise_digits"],
            "candidate_csv": out_csv,
            "error": None if cmp["min_precise_digits"] is not None else cmp.get("log"),
        }
    }


def record_outcome(state: OptimizationState) -> dict:
    """Record the attempt and decide: accept, defer (max retries hit), or retry."""
    var = state["current_variable"]
    iteration = state.get("iteration", 0)
    max_iterations = state["max_iterations"]
    propose_error = state.get("propose_error")
    policy_reject = state.get("policy_reject")
    verify_result = state.get("verify_result") or {}
    verify_pass = verify_result.get("pass", False)
    patchset_size = len(state.get("patch_set") or [])

    # Classify the failure reason for trace/defer metadata
    if propose_error:
        failure_reason = "propose_error"
    elif policy_reject:
        failure_reason = "policy_reject"
    elif not verify_pass and verify_result.get("min_precise_digits") is None:
        failure_reason = "build_fail"
    else:
        failure_reason = "verify_fail"

    new_iteration = iteration + 1 if not verify_pass else iteration
    at_max = new_iteration >= max_iterations

    if verify_pass:
        outcome = "accepted"
    elif at_max:
        outcome = "deferred"
    else:
        outcome = "retry"

    record = {
        "variable": var,
        "iteration": iteration,
        "proposal": state.get("current_proposal"),
        "policy_reject": policy_reject,
        "verify_pass": verify_pass,
        "min_precise_digits": verify_result.get("min_precise_digits"),
        "error": propose_error or verify_result.get("error"),
        "patchset_size_when_attempted": patchset_size,
        "outcome": outcome,
    }
    new_trace = list(state.get("trace") or []) + [record]

    print(
        "[orchestrator] {0!r} iteration {1}: {2}".format(var, iteration, outcome),
        file=sys.stderr,
    )

    if verify_pass:
        new_patch_set = list(state.get("patch_set") or []) + [state["current_proposal"]]
        new_accepted = list(state.get("accepted_variables") or []) + [var]
        return {
            "trace": new_trace,
            "patch_set": new_patch_set,
            "accepted_variables": new_accepted,
            "current_variable": None,
            "proposer_messages": [],
            "current_proposal": None,
            "current_tool_use_id": None,
            "propose_error": None,
            "policy_reject": None,
            "verify_result": None,
        }

    if at_max:
        deferred_entry = {
            "name": var,
            "failed_at_patchset_size": patchset_size,
            "requeue_count": 0,
            "last_failure_reason": failure_reason,
        }
        new_deferred = list(state.get("deferred") or []) + [deferred_entry]
        return {
            "trace": new_trace,
            "deferred": new_deferred,
            "current_variable": None,
            "proposer_messages": [],
            "current_proposal": None,
            "current_tool_use_id": None,
            "propose_error": None,
            "policy_reject": None,
            "verify_result": None,
            "iteration": new_iteration,
        }

    # Retry: build feedback message for the next proposer call
    messages = list(state.get("proposer_messages") or [])
    tool_use_id = state.get("current_tool_use_id")

    if propose_error:
        # API/format failure: reset conversation so next attempt starts fresh
        messages = []
    elif tool_use_id:
        if policy_reject:
            reason = policy_reject
        elif verify_result.get("min_precise_digits") is not None:
            reason = "verify failed (min_precise_digits={0})".format(
                verify_result["min_precise_digits"]
            )
        else:
            reason = "build or verify failed: {0}".format(
                verify_result.get("error", "unknown")
            )
        messages = messages + [build_tool_result_feedback(tool_use_id, reason, var)]

    return {
        "trace": new_trace,
        "iteration": new_iteration,
        "proposer_messages": messages,
        "current_proposal": None,
        "current_tool_use_id": None,
        "propose_error": None,
        "policy_reject": None,
        "verify_result": None,
    }


def maybe_requeue(state: OptimizationState) -> dict:
    """Re-queue deferred variables if under the cycle cap, otherwise mark as rejected."""
    deferred = list(state.get("deferred") or [])
    requeue_cycles = state.get("requeue_cycles", 0)
    max_requeue_cycles = state.get("max_requeue_cycles", config.MAX_REQUEUE_CYCLES)

    if deferred and requeue_cycles < max_requeue_cycles:
        new_candidates = [dv["name"] for dv in deferred]
        print(
            "[orchestrator] Re-queuing {0} deferred variables (cycle {1}/{2}): {3}".format(
                len(new_candidates), requeue_cycles + 1, max_requeue_cycles, new_candidates
            ),
            file=sys.stderr,
        )
        return {
            "candidates": new_candidates,
            "deferred": [],
            "requeue_cycles": requeue_cycles + 1,
        }

    # Exhausted requeue budget: mark remaining deferred as permanently rejected
    patchset_size = len(state.get("patch_set") or [])
    new_trace = list(state.get("trace") or [])
    for dv in deferred:
        new_trace.append({
            "variable": dv["name"],
            "iteration": -1,
            "proposal": None,
            "policy_reject": None,
            "verify_pass": False,
            "min_precise_digits": None,
            "error": "exhausted requeue cycles",
            "patchset_size_when_attempted": patchset_size,
            "outcome": "rejected_permanently",
        })
        print(
            "[orchestrator] {0!r} rejected permanently after {1} requeue cycle(s)".format(
                dv["name"], requeue_cycles
            ),
            file=sys.stderr,
        )
    return {"trace": new_trace, "deferred": []}


def aggregate_results(state: OptimizationState) -> dict:
    """Write a summary JSON and return."""
    output_dir = state.get("output_dir") or os.path.join(state["root"], "experiments")
    sig = state.get("signature")
    fn_name = sig["function_name"] if sig else state.get("function_name", "unknown")

    out_dir = os.path.join(output_dir, fn_name, "generated")
    os.makedirs(out_dir, exist_ok=True)

    trace = state.get("trace") or []
    accepted_vars = state.get("accepted_variables") or []
    deferred_vars = [dv["name"] for dv in (state.get("deferred") or [])]
    rejected_vars = [r["variable"] for r in trace if r.get("outcome") == "rejected_permanently"]

    summary = {
        "function_name":    fn_name,
        "file_path":        state.get("file_path"),
        "framework":        sig.get("framework") if sig else None,
        "baseline_csv":     state.get("baseline_csv"),
        "error":            state.get("error"),
        # New orchestrator-loop fields
        "final_patch_set":      state.get("patch_set") or [],
        "accepted_variables":   accepted_vars,
        "deferred_variables":   deferred_vars,
        "rejected_variables":   rejected_vars,
        "requeue_cycles_used":  state.get("requeue_cycles", 0),
        "trace":                trace,
        # Legacy field for backward compat (empty when orchestrator loop is active)
        "skill_results":    state.get("skill_results") or {},
    }

    ts = time.strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(out_dir, "{0}_summary_{1}.json".format(fn_name, ts))
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("[orchestrator] Summary written to {0}".format(summary_path), file=sys.stderr)
    return {}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_analyze(
    state: OptimizationState,
) -> Literal["run_driver", "aggregate_results"]:
    return "aggregate_results" if state.get("error") else "run_driver"


def route_after_driver(
    state: OptimizationState,
) -> Literal["init_candidate_loop", "aggregate_results"]:
    return "aggregate_results" if state.get("error") else "init_candidate_loop"


def route_after_init(
    state: OptimizationState,
) -> Literal["pick_candidate", "aggregate_results"]:
    if state.get("error"):
        return "aggregate_results"
    if not state.get("candidates"):
        return "aggregate_results"
    return "pick_candidate"


def route_after_pick(
    state: OptimizationState,
) -> Literal["invoke_proposer", "maybe_requeue"]:
    if state.get("current_variable") is None:
        return "maybe_requeue"
    return "invoke_proposer"


def route_after_record(
    state: OptimizationState,
) -> Literal["pick_candidate", "invoke_proposer"]:
    # current_variable=None signals accept or defer; otherwise retry
    if state.get("current_variable") is None:
        return "pick_candidate"
    return "invoke_proposer"


def route_after_maybe_requeue(
    state: OptimizationState,
) -> Literal["pick_candidate", "aggregate_results"]:
    return "pick_candidate" if state.get("candidates") else "aggregate_results"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_orchestrator():
    g = StateGraph(OptimizationState)

    g.add_node("load_target",           load_target)
    g.add_node("run_analyze",           run_analyze)
    g.add_node("run_driver",            run_driver_skill)
    g.add_node("init_candidate_loop",   init_candidate_loop)
    g.add_node("pick_candidate",        pick_candidate)
    g.add_node("invoke_proposer",       invoke_proposer)
    g.add_node("policy_check",          policy_check)
    g.add_node("build_and_verify",      build_and_verify)
    g.add_node("record_outcome",        record_outcome)
    g.add_node("maybe_requeue",         maybe_requeue)
    g.add_node("aggregate_results",     aggregate_results)

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
        {"init_candidate_loop": "init_candidate_loop", "aggregate_results": "aggregate_results"},
    )
    g.add_conditional_edges(
        "init_candidate_loop",
        route_after_init,
        {"pick_candidate": "pick_candidate", "aggregate_results": "aggregate_results"},
    )
    g.add_conditional_edges(
        "pick_candidate",
        route_after_pick,
        {"invoke_proposer": "invoke_proposer", "maybe_requeue": "maybe_requeue"},
    )
    g.add_edge("invoke_proposer", "policy_check")
    g.add_edge("policy_check",    "build_and_verify")
    g.add_edge("build_and_verify", "record_outcome")
    g.add_conditional_edges(
        "record_outcome",
        route_after_record,
        {"pick_candidate": "pick_candidate", "invoke_proposer": "invoke_proposer"},
    )
    g.add_conditional_edges(
        "maybe_requeue",
        route_after_maybe_requeue,
        {"pick_candidate": "pick_candidate", "aggregate_results": "aggregate_results"},
    )
    g.add_edge("aggregate_results", END)

    return g.compile()
