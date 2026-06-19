"""InstrumentationSpec — what the characterizer knows about a kernel before generating a driver."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class InstrumentationSpec:
    kernel_name: str
    kernel_signature: str                          # raw text from source
    parameter_types: list[tuple[str, str]]         # [(arg_name, type_str), ...]
    input_ranges: dict[str, tuple[float, float]]   # arg_name → (min, max)
    template_instantiation: dict[str, str]         # e.g. {"TOutput": "tracked::Complex<double>"}
    sample_count: int
    framework: str                                 # "plain-cpp" | "kokkos-serial"
    source_files: list[str] = field(default_factory=list)  # absolute paths to kernel source files
    detected_dispatchers: list[str] = field(default_factory=list)  # ["kAbs", "kLog", ...]
    # Per-parameter role, parallel to parameter_types: input | output | inout.
    parameter_roles: list[Literal["input", "output", "inout"]] = field(default_factory=list)
