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
    _last_tool_use_id: Optional[str]  # set by extract_signature when LLM returns a tool call


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


class DeferredVariable(TypedDict):
    name: str
    failed_at_patchset_size: int
    requeue_count: int
    last_failure_reason: str  # "policy_reject" | "build_fail" | "verify_fail" | "propose_error"


class AttemptRecord(TypedDict):
    variable: str
    iteration: int
    proposal: Optional[PatchProposal]
    policy_reject: Optional[str]
    verify_pass: bool
    min_precise_digits: Optional[float]
    error: Optional[str]
    patchset_size_when_attempted: int
    outcome: str  # "accepted" | "deferred" | "rejected_permanently" | "retry"


# ---------------------------------------------------------------------------
# Downcast proposer subgraph state
# (used by the shrunk downcast skill — pure proposer, no loop logic)
# ---------------------------------------------------------------------------

class DowncastProposerState(TypedDict):
    spec: dict
    impl_source: str             # pristine unpatched source
    accepted_patches: List[PatchProposal]
    accepted_variables: List[str]
    current_variable: str
    iteration: int
    max_iterations: int
    min_digits: float
    base_url: Optional[str]
    messages: List[dict]         # LLM conversation history for this variable
    proposal: Optional[PatchProposal]
    current_tool_use_id: Optional[str]
    propose_error: Optional[str]


# ---------------------------------------------------------------------------
# Top-level orchestrator state
# ---------------------------------------------------------------------------

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
    max_requeue_cycles: int
    skills: List[str]
    base_url: Optional[str]
    output_dir: Optional[str]

    # Set by agents during run
    signature: Optional[FunctionSignature]
    skill_results: Dict[str, dict]
    error: Optional[str]

    # Candidate loop state — set by init_candidate_loop
    candidates: List[str]
    deferred: List[dict]            # DeferredVariable dicts
    patch_set: List[PatchProposal]  # orchestrator-owned cumulative accepted patches
    accepted_variables: List[str]   # variables with an accepted patch
    impl_source: Optional[str]      # pristine source loaded once
    spec: Optional[dict]            # build spec dict built once
    downcast_baseline_csv: Optional[str]
    trace: List[dict]               # AttemptRecord dicts
    requeue_cycles: int

    # Per-variable iteration state — reset by pick_candidate
    current_variable: Optional[str]
    iteration: int
    proposer_messages: List[dict]   # conversation history for current variable
    current_tool_use_id: Optional[str]
    current_proposal: Optional[PatchProposal]
    propose_error: Optional[str]
    policy_reject: Optional[str]
    verify_result: Optional[dict]
