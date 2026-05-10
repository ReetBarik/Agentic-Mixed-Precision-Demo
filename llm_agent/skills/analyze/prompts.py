"""Prompt construction and tool schema for the signature analysis skill."""

SYSTEM_PROMPT = """\
You are an expert C++ code analyst. Your task is to extract the complete signature of a
specific function from a C++ source file and produce a structured description suitable
for automated test driver generation.

Rules:
- Identify the target function by name. If it is overloaded, prefer the most general version.
- Detect the portability framework from annotations, macros, or includes:
    KOKKOS_INLINE_FUNCTION / KOKKOS_FUNCTION → "kokkos"
    SYCL_EXTERNAL / sycl:: namespace / cl:: namespace → "sycl"
    __device__ / __global__ / __host__ → "cuda"
    #pragma omp / omp_ functions → "openmp"
    __hip_device__ / HIP_KERNEL_NAME → "hip"
    none of the above → "none"
- For each input parameter infer a reasonable random test domain from:
    - the parameter name (e.g. "angle" → (-π, π), "mass" → (0.1, 10), "x" → (-4, 4))
    - the type (unsigned → (0, 100), floating-point → (-4, 4) by default)
    - any inline comments near the function
- Scan the entire function body and list every local variable declared as double, float,
  or a floating-point template alias in locals_for_downcast.
  These are candidates for mixed-precision downcast optimization. Be thorough.
- Fill call_expression using {param_name} placeholders for each input param.
  Example: "myns::compute<TOut, TIn>({x})"
- Fill concrete_template_types with a mapping from each template parameter name to the
  concrete C++ type to use when instantiating a test driver. For each function called
  inside the body, list its overloads and verify that the chosen concrete types produce
  distinct signatures across all overloads. Two overloads collapse to the same signature
  when two different template parameters are mapped to the same concrete type — so if
  any called function has overloads distinguished only by different template parameters
  (e.g. one overload uses TOut, another uses TIn), those parameters MUST be mapped
  to different concrete types.
- You must call extract_signature with a complete, valid answer.
"""

EXTRACT_SIGNATURE_TOOL = {
    "name": "extract_signature",
    "description": "Extract the complete signature of the target C++ function.",
    "input_schema": {
        "type": "object",
        "properties": {
            "namespace": {
                "type": "string",
                "description": "C++ namespace of the function, or empty string if none.",
            },
            "framework": {
                "type": "string",
                "enum": ["kokkos", "sycl", "hip", "openmp", "cuda", "none"],
                "description": (
                    "Portability framework detected from annotations, includes, or macros. "
                    "Use 'none' for plain C++."
                ),
            },
            "return_type": {
                "type": "string",
                "description": "Return type of the function as it appears in source, e.g. 'double' or 'void'.",
            },
            "is_template": {
                "type": "boolean",
                "description": "True if the function is a template.",
            },
            "template_params": {
                "type": "array",
                "description": "Template parameters in declaration order.",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "kind": {"type": "string", "description": "'typename' or 'class' or a non-type kind like 'int'"},
                    },
                    "required": ["name", "kind"],
                },
            },
            "input_params": {
                "type": "array",
                "description": "Parameters that are inputs (not output-by-reference).",
                "items": {
                    "type": "object",
                    "properties": {
                        "name":       {"type": "string"},
                        "type":       {"type": "string", "description": "Type as in source, e.g. 'double const&'"},
                        "is_const":   {"type": "boolean"},
                        "is_ref":     {"type": "boolean"},
                        "domain_min": {"type": "number", "description": "Lower bound for random test values."},
                        "domain_max": {"type": "number", "description": "Upper bound for random test values."},
                    },
                    "required": ["name", "type", "is_const", "is_ref", "domain_min", "domain_max"],
                },
            },
            "output_params": {
                "type": "array",
                "description": "Parameters used to return results by non-const reference or pointer (only for void-return functions).",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {"type": "string"},
                    },
                    "required": ["name", "type"],
                },
            },
            "call_expression": {
                "type": "string",
                "description": (
                    "Full call expression with {param_name} placeholders for each input. "
                    "Example: 'myns::compute<TOut, TIn>({x})'"
                ),
            },
            "locals_for_downcast": {
                "type": "array",
                "description": (
                    "Names of local variables declared inside the function body with type "
                    "double, float, or a template alias that resolves to a floating-point type. "
                    "These are candidates for mixed-precision downcasting. "
                    "Look carefully through the full function body and list every such local variable name. "
                    "Do NOT include function parameters — only locally declared variables."
                ),
                "items": {"type": "string"},
            },
            "concrete_template_types": {
                "type": "object",
                "description": (
                    "Mapping from each template parameter name to the concrete C++ type to use "
                    "when instantiating a test driver (e.g. {\"T\": \"double\"}). "
                    "Choose types such that no two overloads of any function called inside the "
                    "body collapse to the same signature under this instantiation."
                ),
                "additionalProperties": {"type": "string"},
            },
        },
        "required": [
            "framework",
            "return_type",
            "is_template",
            "template_params",
            "input_params",
            "output_params",
            "call_expression",
            "locals_for_downcast",
            "concrete_template_types",
        ],
    },
}


def build_extract_message(file_path: str, function_name: str, source: str) -> dict:
    """Build the initial user message for the signature extraction conversation."""
    content = (
        "Extract the signature of the function '{function_name}' from the file '{file_path}'.\n\n"
        "===== BEGIN FILE =====\n"
        "{source}\n"
        "===== END FILE =====\n\n"
        "Call extract_signature with the complete structured description."
    ).format(
        function_name=function_name,
        file_path=file_path,
        source=source,
    )
    return {"role": "user", "content": content}


def build_rejection_feedback(tool_use_id: str, reason: str) -> dict:
    """Build a tool_result rejection message to send back to the LLM."""
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": (
                    "REJECTED: {reason}\n"
                    "Please correct your extract_signature call and try again."
                ).format(reason=reason),
            }
        ],
    }
