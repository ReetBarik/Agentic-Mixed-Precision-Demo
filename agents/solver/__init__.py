"""Phase 2e greedy mixed-precision solver.

The first pipeline stage that actually *writes* an optimized source tree: it ranks
the fan-out's measured DISCRIM ``(region, rung)`` cells cheapest-first
(float < ff < dd) and greedily layers each onto the accumulated tree, keeping it if
the whole-app p100 precise-digits gate holds and reverting it otherwise.
"""

from agents.solver.queue import (
    Candidate, QueueBuild, RUNG_RANK, build_queue, load_manifest_rows,
)
from agents.solver.solver import (
    ACCEPTED, APPLY_FAILED, DEFAULT_GATE, REJECTED, SKIPPED_RESOLVED,
    STOPPED_GATE_UNIMPLEMENTABLE, ApplyResult, CandidateOutcome, SolveResult,
    ValidateResult, solve,
)

__all__ = [
    "Candidate", "QueueBuild", "RUNG_RANK", "build_queue", "load_manifest_rows",
    "ACCEPTED", "APPLY_FAILED", "DEFAULT_GATE", "REJECTED", "SKIPPED_RESOLVED",
    "STOPPED_GATE_UNIMPLEMENTABLE", "ApplyResult", "CandidateOutcome",
    "SolveResult", "ValidateResult", "solve",
]
