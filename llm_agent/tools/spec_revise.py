"""LLM-driven spec revision for the orchestrator's spec-driven baseline build.

The downcast-baseline path renders an ephemeral driver from a spec produced by the analyze
skill and compiles it via build_and_run. Unlike the driver skill (which has its own
LLM compile-fix loop on the LLM-generated driver), this path has no recovery: if analyze
picks template instantiations that make an overloaded callee ambiguous, the build fails
and the run is aborted.

build_and_run_with_revision wraps build_and_run with a small retry loop. On compile
failure, it asks the LLM to revise concrete_template_types based on the compiler's error
output, then retries. The revised spec is returned alongside the build result so that
downstream per-candidate verification builds use the same instantiation.
"""

import json
import sys
from typing import Optional, Tuple

from llm_agent import config
from llm_agent.client import make_client
from llm_agent.tools.build import build_and_run


_REVISE_SYSTEM_PROMPT = """\
You repair C++ compile errors caused by template instantiation choices in an automated
test driver. You will see the current concrete-type mapping for the target function's
template parameters and the compiler's error output. Propose a revised mapping that
compiles.

Common causes of failure:
- Two distinct template parameters mapped to the same concrete type, making overloaded
  callees ambiguous (compiler reports "call of overloaded ... is ambiguous").
- A concrete type that lacks the operators or conversions required at a call site.

Constraints on your revision:
- Map each template parameter to a concrete C++ type that compiles cleanly with the
  call expression and the function body.
- When overloaded callees take the same template signature with reordered parameters,
  the parameters MUST be mapped to different concrete types.
- Preserve the high-precision baseline: prefer double-precision concrete types unless
  the original mapping already mixes precisions for a documented reason.
- Return the FULL mapping (every template parameter present in the original), not just
  the entries you changed.

Call revise_concrete_template_types with the full updated mapping.
"""

REVISE_TOOL = {
    "name": "revise_concrete_template_types",
    "description": (
        "Propose a new mapping from template parameter names to concrete C++ types "
        "to use when instantiating the test driver."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "concrete_template_types": {
                "type": "object",
                "description": (
                    "Full updated mapping. Must include every template parameter "
                    "present in the original mapping."
                ),
                "additionalProperties": {"type": "string"},
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of why this revision should fix the build.",
            },
        },
        "required": ["concrete_template_types", "reasoning"],
    },
}


def _build_revision_message(spec: dict, build_error: str) -> dict:
    cur_ctt = spec.get("concrete_template_types") or {}
    excerpt = (build_error or "")[-3000:]
    content = (
        "The C++ driver for the target function below failed to compile.\n\n"
        "Function: {fn}\n"
        "Header:   {hdr}\n"
        "Return type: {rt}\n"
        "Call expression: {call}\n\n"
        "Current concrete_template_types:\n{ctt}\n\n"
        "Compile error (tail):\n{err}\n\n"
        "Propose an updated mapping via revise_concrete_template_types."
    ).format(
        fn=spec.get("id", "?"),
        hdr=spec.get("header_path", "?"),
        rt=spec.get("return_type", "?"),
        call=(spec.get("call") or {}).get("expression", "?"),
        ctt=json.dumps(cur_ctt, indent=2),
        err=excerpt,
    )
    return {"role": "user", "content": content}


def _request_revision(
    spec: dict, build_error: str, base_url: Optional[str] = None
) -> Optional[dict]:
    """Single LLM round-trip; returns the new ctt mapping or None on failure."""
    try:
        client = make_client(base_url=base_url)
        response = client.messages.create(
            model=config.DEFAULT_MODEL,
            max_tokens=1024,
            system=_REVISE_SYSTEM_PROMPT,
            tools=[REVISE_TOOL],
            tool_choice={"type": "tool", "name": "revise_concrete_template_types"},
            messages=[_build_revision_message(spec, build_error)],
        )
    except Exception as exc:
        print(
            "[orchestrator] Spec-revision LLM call failed: {0}".format(str(exc)[:300]),
            file=sys.stderr,
        )
        return None

    block = next((b for b in response.content if b.type == "tool_use"), None)
    if block is None:
        return None
    new_ctt = block.input.get("concrete_template_types")
    if not isinstance(new_ctt, dict) or not new_ctt:
        return None
    return new_ctt


def build_and_run_with_revision(
    root: str,
    spec: dict,
    impl_source: str,
    batch: int,
    seed: int,
    out_csv: str,
    max_attempts: int = 3,
    base_url: Optional[str] = None,
) -> Tuple[dict, dict]:
    """Build + run with LLM-driven spec revision on compile failure.

    Returns (build_result, final_spec). On success, build_result['ok'] is True and
    final_spec is the spec that compiled. On exhausted retries, returns the last failed
    build_result alongside the most recent spec attempted.
    """
    cur_spec = dict(spec)
    last_result = {"ok": False, "error": "no attempts made", "logs": {}}
    for attempt in range(1, max_attempts + 1):
        last_result = build_and_run(
            root=root,
            spec=cur_spec,
            impl_source=impl_source,
            batch=batch,
            seed=seed,
            out_csv=out_csv,
        )
        if last_result.get("ok"):
            return last_result, cur_spec
        if attempt >= max_attempts:
            break

        logs = last_result.get("logs") or {}
        log_text = next(iter(logs.values()), "") if logs else ""
        print(
            "[orchestrator] Baseline build failed (attempt {0}/{1}); "
            "asking LLM to revise concrete_template_types...".format(
                attempt, max_attempts
            ),
            file=sys.stderr,
        )
        new_ctt = _request_revision(cur_spec, log_text, base_url=base_url)
        if new_ctt is None:
            print(
                "[orchestrator] LLM did not propose a revision; giving up.",
                file=sys.stderr,
            )
            break
        prev_ctt = cur_spec.get("concrete_template_types") or {}
        if new_ctt == prev_ctt:
            print(
                "[orchestrator] LLM proposed identical mapping; giving up.",
                file=sys.stderr,
            )
            break
        print(
            "[orchestrator] Revised concrete_template_types: {0}".format(
                json.dumps(new_ctt)
            ),
            file=sys.stderr,
        )
        cur_spec = dict(cur_spec)
        cur_spec["concrete_template_types"] = new_ctt

    return last_result, cur_spec
