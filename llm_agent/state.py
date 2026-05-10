"""LangGraph TypedDict state definitions."""

from typing import Dict, List, Optional, TypedDict


# ---------------------------------------------------------------------------
# Signature analysis types
# ---------------------------------------------------------------------------

class FunctionParam(TypedDict):
    name: str
    type: str           # as it appears in source, e.g. "double const&"
    is_const: bool
    is_ref: bool
    is_output: bool     # True = output-by-reference, not an input
    domain_min: Optional[float]
    domain_max: Optional[float]


class FunctionSignature(TypedDict):
    function_name: str
    file_path: str      # repo-relative path to the header
    namespace: Optional[str]
    framework: Optional[str]  # "kokkos"|"sycl"|"hip"|"openmp"|"cuda"|"none"
    return_type: str    # "void" if no return value
    is_template: bool
    template_params: List[dict]       # [{"name": "T", "kind": "typename"}, ...]
    input_params: List[FunctionParam]
    output_params: List[FunctionParam]  # void-return output-by-reference params
    call_expression: str  # e.g. "ns::compute<T, U>({x})"
    locals_for_downcast: List[str]    # local double/float vars in function body
    concrete_template_types: dict     # e.g. {"T": "double", "U": "std::complex<double>"}


class AnalyzeState(TypedDict):
    file_path: str
    function_name: str
    source: str         # full file content
    messages: List[dict]
    signature: Optional[FunctionSignature]
    iteration: int
    max_iterations: int
    error: Optional[str]


class DriverState(TypedDict):
    signature: FunctionSignature
    root: str
    batch: int
    seed: int
    max_iterations: int  # max compile-fix attempts
    driver_source: Optional[str]
    cmake_source: Optional[str]
    exe_path: Optional[str]       # temp file written by compile_driver, consumed by run_driver
    out_csv: Optional[str]
    compile_error: Optional[str]
    compile_ok: bool
    run_ok: bool
    messages: List[dict]
    iteration: int
    error: Optional[str]
    _last_tool_use_id: Optional[str]  # transient: tool_use id for building feedback


class PatchProposal(TypedDict):
    file_path: str
    old_line: str
    new_line: str
    reasoning: str


class AttemptRecord(TypedDict):
    variable: str
    iteration: int
    proposal: Optional[PatchProposal]
    policy_reject: Optional[str]
    verify_pass: bool
    min_precise_digits: Optional[float]
    error: Optional[str]


class DowncastState(TypedDict):
    # Target context (set once at start, read-only during run)
    spec: dict
    root: str
    impl_source: str       # Original unpatched source content
    baseline_csv: str
    min_digits: float
    batch: int
    seed: int
    max_iterations: int    # Max proposal attempts per variable

    # Iteration control (mutated as the subgraph progresses)
    variables: List[str]           # Remaining variables to try
    current_variable: Optional[str]
    iteration: int                 # Attempt count for current_variable

    # Per-attempt transient state (reset each iteration)
    current_proposal: Optional[PatchProposal]
    current_tool_use_id: Optional[str]
    policy_reject: Optional[str]
    verify_result: Optional[dict]
    propose_error: Optional[str]

    # Accumulated results
    accepted_patches: List[PatchProposal]
    accepted_variables: List[str]
    rejected_variables: List[str]
    trace: List[AttemptRecord]

    # LLM conversation history for the current variable (plain list, not add_messages;
    # reset to [] by pick_variable, explicitly extended by propose and record_result)
    messages: List[dict]


class OptimizationState(TypedDict):
    # Input parameters
    file_path: str      # repo-relative path to the C++ header/source
    function_name: str  # name of the function to optimize
    root: str
    min_digits: float
    batch: int
    seed: int
    max_iterations: int
    max_driver_retries: int
    skills: List[str]
    base_url: Optional[str]
    output_dir: Optional[str]

    # Set by agents during run
    signature: Optional[FunctionSignature]
    baseline_csv: Optional[str]
    skill_results: Dict[str, dict]
    error: Optional[str]
