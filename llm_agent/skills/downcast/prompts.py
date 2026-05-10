"""Prompt construction and tool schema for the downcast proposer."""

import json
from typing import List

from llm_agent.state import PatchProposal
from llm_agent.tools.build import apply_patches

SYSTEM_PROMPT = """\
You are a mixed-precision optimization expert for high-performance C++ functions.
Your goal is to propose minimal, targeted source code changes that downcast specific
local variables from double precision to float precision while preserving numerical
accuracy above a given digit threshold.

Rules:
- Modify ONLY the file specified in the task.
- Change ONLY the focused variable's declaration or expression.
- Keep the change to a single line replacement when possible.
- Do not rename variables, change function signatures, or restructure unrelated code.
- You must call the propose_patch tool with the exact old line and new line.
- A valid downcast MUST change the variable's type to a genuinely lower-precision type
  (e.g. float). Changing a template type alias to its underlying double type, or
  rewriting initialization syntax without changing the type, is NOT a downcast and will
  be rejected. If no safe float downcast exists for the focused variable, set new_line
  identical to old_line to signal that this variable should be skipped.
- A complex variable (e.g. std::complex<double>) must only be downcast to the same
  complex template instantiated with a lower-precision scalar (e.g. std::complex<float>).
  Never replace a complex type with a scalar type.
"""

PROPOSE_PATCH_TOOL = {
    "name": "propose_patch",
    "description": (
        "Propose a single source line replacement to downcast a local variable "
        "from double to float precision in a C++ kernel."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Repository-relative path to the header/source file to modify.",
            },
            "old_line": {
                "type": "string",
                "description": "The exact existing source line to replace, as it appears verbatim in the file.",
            },
            "new_line": {
                "type": "string",
                "description": "The replacement source line with the downcast applied.",
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of why this downcast is safe numerically.",
            },
        },
        "required": ["file_path", "old_line", "new_line", "reasoning"],
    },
}


def build_initial_user_message(
    spec: dict,
    impl_source: str,
    current_variable: str,
    accepted_patches: List[PatchProposal],
    accepted_variables: List[str],
    min_digits: float,
) -> dict:
    """Build the first user message for a variable's optimization conversation."""
    header_path = spec.get("header_path")
    if not header_path:
        raise ValueError("spec missing header_path")

    # Show the current state of the file (with accepted patches already applied)
    if accepted_patches:
        try:
            current_source = apply_patches(impl_source, accepted_patches)
        except ValueError:
            current_source = impl_source
    else:
        current_source = impl_source

    all_locals = spec.get("locals_for_downcast", [])
    accepted_str = ", ".join(accepted_variables) if accepted_variables else "(none yet)"

    content = (
        "Target function: {function_symbol} in {header_path}\n"
        "Numerical threshold: minimum {min_digits} precise decimal digits vs double baseline.\n"
        "Input domain: {input_domain}\n\n"
        "All candidate locals: {all_locals}\n"
        "Already accepted (downcast in previous iterations): {accepted_str}\n\n"
        "CURRENT TASK: Propose a downcast for variable: {current_variable}\n"
        "Focus ONLY on {current_variable}. Do not modify other variables.\n\n"
        "Current file content ({header_path}):\n"
        "===== BEGIN FILE =====\n"
        "{source}\n"
        "===== END FILE =====\n\n"
        "Call propose_patch with the exact old line and your proposed new line."
    ).format(
        function_symbol=spec.get("function_symbol", "(unknown)"),
        header_path=header_path,
        min_digits=min_digits,
        input_domain=json.dumps(spec.get("inputs", spec.get("input_domain", {})), indent=2),
        all_locals=", ".join(all_locals) if all_locals else "(none)",
        accepted_str=accepted_str,
        current_variable=current_variable,
        source=current_source,
    )
    return {"role": "user", "content": content}


def build_tool_result_feedback(tool_use_id: str, reason: str, current_variable: str) -> dict:
    """Build the tool_result message that feeds rejection back to the LLM."""
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": (
                    "REJECTED: {reason}\n"
                    "Please try a different approach for variable {var}. "
                    "Keep the change minimal and focused on the declaration line."
                ).format(reason=reason, var=current_variable),
            }
        ],
    }
