"""Prompt construction and tool schema for the driver generation skill."""

import json

from llm_agent.state import FunctionSignature

SYSTEM_PROMPT = """\
You are an expert C++ test driver author for scientific computing functions.
Given a function signature, generate a complete, self-contained C++ driver that:
  1. Initializes the portability framework (if any).
  2. Generates `batch` random input samples using std::mt19937 seeded by `seed`.
  3. Calls the target function once per sample, collecting the output.
  4. Writes results to a CSV file (path = argv[2]) with the format:
       Line 1 (header):  id,<col names>
       Line 2 (meta):    # function=<name>
       Remaining lines:  <i>,<hex-encoded value(s)>
     Use IEEE 754 hex encoding: memcpy the value to uint64_t, then print with std::hex.
     This ensures bitwise reproducibility across platforms.
  5. Finalizes the portability framework (if any).

Framework rules:
  kokkos:  Use Kokkos::initialize(argc, argv), Kokkos::finalize(), Kokkos::parallel_for
           with a RangePolicy, Kokkos::View and Kokkos::create_mirror_view_and_copy.
           CMakeLists MUST use exactly:
             cmake_minimum_required(VERSION 3.16)
             project(driver LANGUAGES CXX)
             set(CMAKE_CXX_STANDARD 17)
             set(CMAKE_CXX_STANDARD_REQUIRED ON)
             find_package(Kokkos REQUIRED)
             add_executable(driver driver.cc)
             target_link_libraries(driver PRIVATE Kokkos::kokkos)
           Do NOT add target_include_directories — it will be injected automatically.
  sycl:    Use sycl::queue, USM or buffer/accessor pattern.
           CMakeLists: add_compile_options(-fsycl), or find_package(IntelSYCL).
  openmp:  Use #pragma omp parallel for.
           CMakeLists: find_package(OpenMP REQUIRED), target_link_libraries(OpenMP::OpenMP_CXX).
  cuda:    Use cudaMalloc/cudaMemcpy or thrust.
           CMakeLists: enable_language(CUDA).
  hip:     Use hipMalloc/hipMemcpy.
           CMakeLists: find_package(hip REQUIRED), target_link_libraries(hip::device).
  none:    Plain C++ loop. CMakeLists:
             cmake_minimum_required(VERSION 3.16)
             project(driver LANGUAGES CXX)
             set(CMAKE_CXX_STANDARD 17)
             add_executable(driver driver.cc)

CRITICAL for all frameworks:
  - The CMake target MUST be named exactly `driver`.
  - The driver source file is named `driver.cc` and is in the cmake source root.
  - Do NOT add target_include_directories — it is injected automatically.
  - Do NOT hardcode any absolute paths in cmake_source.
  - Read batch size from argv[1], output CSV path from argv[2], seed from argv[3].
    Provide defaults: batch=10, out="driver_out.csv", seed=123.
  - Include the target header with #include "<filename_only>" (not a full path).
  - For template functions use the concrete types provided in the signature's
    concrete_template_types mapping. These have been chosen to avoid overload ambiguity.
  - For void-return functions with output-by-reference params, declare the output
    variable before the call, then serialize it after.
  - You must call generate_driver with both driver_source and cmake_source.
"""

GENERATE_DRIVER_TOOL = {
    "name": "generate_driver",
    "description": (
        "Generate a complete C++ driver and matching CMakeLists.txt that calls the "
        "target function with random inputs and writes IEEE 754 hex CSV to stdout."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "driver_source": {
                "type": "string",
                "description": "Complete, compilable C++ source for the driver executable.",
            },
            "cmake_source": {
                "type": "string",
                "description": (
                    "Complete CMakeLists.txt for building the driver, including "
                    "find_package() and target_link_libraries() appropriate for the "
                    "detected framework. Do NOT include target_include_directories — "
                    "that will be appended automatically."
                ),
            },
            "reasoning": {
                "type": "string",
                "description": "Brief explanation of framework/output-collection choices made.",
            },
        },
        "required": ["driver_source", "cmake_source", "reasoning"],
    },
}


def build_generate_message(sig: FunctionSignature, batch: int, seed: int) -> dict:
    """Build the initial user message asking the LLM to generate a driver."""
    params_desc = []
    for p in sig.get("input_params", []):
        params_desc.append(
            "  {name} ({type}): domain [{min}, {max}]".format(
                name=p["name"],
                type=p["type"],
                min=p.get("domain_min", -4.0),
                max=p.get("domain_max", 4.0),
            )
        )
    for p in sig.get("output_params", []):
        params_desc.append("  {name} ({type}): OUTPUT by reference".format(
            name=p["name"], type=p["type"]
        ))

    template_str = ""
    if sig.get("is_template") and sig.get("template_params"):
        names = [tp["name"] for tp in sig["template_params"]]
        template_str = "\nTemplate params: {0}".format(", ".join(names))

    import os as _os
    header_filename = _os.path.basename(sig["file_path"])

    concrete_types = sig.get("concrete_template_types") or {}
    concrete_str = ""
    if concrete_types:
        concrete_str = "\nConcrete template types (use exactly these to avoid overload ambiguity):\n"
        concrete_str += "\n".join(
            "  {0} → {1}".format(k, v) for k, v in concrete_types.items()
        )

    content = (
        "Generate a test driver for the following C++ function.\n\n"
        "Function: {call_expression}\n"
        "Header filename: {header_filename}  (use #include \"{header_filename}\" — the directory is on the include path)\n"
        "Framework: {framework}\n"
        "Return type: {return_type}{template_str}{concrete_str}\n\n"
        "Parameters:\n{params}\n\n"
        "Batch size: {batch}  Seed: {seed}\n\n"
        "IMPORTANT: Name the CMake target and executable exactly `driver`. "
        "Name the source file `driver.cc`. Do not add target_include_directories.\n\n"
        "Call generate_driver with the complete driver_source and cmake_source."
    ).format(
        call_expression=sig.get("call_expression", sig["function_name"] + "(...)"),
        header_filename=header_filename,
        framework=sig.get("framework", "none"),
        return_type=sig.get("return_type", "unknown"),
        template_str=template_str,
        concrete_str=concrete_str,
        params="\n".join(params_desc) if params_desc else "  (none)",
        batch=batch,
        seed=seed,
    )
    return {"role": "user", "content": content}


def build_compile_error_feedback(tool_use_id: str, error_log: str) -> dict:
    """Build a tool_result message feeding compilation errors back to the LLM."""
    return {
        "role": "user",
        "content": [
            {
                "type": "tool_result",
                "tool_use_id": tool_use_id,
                "content": (
                    "COMPILATION FAILED:\n{error}\n\n"
                    "Fix the driver_source and cmake_source and call generate_driver again."
                ).format(error=error_log[:3000]),
            }
        ],
    }
