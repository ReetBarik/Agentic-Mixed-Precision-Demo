"""Phase-1 template-argument promotion — per-integral dispatch layer (deliverable 3).

Once each promoted integral's group has its own dd TU binary (deliverable 2) and every
other integral stays in the vanilla binary, *dispatch* is the step that routes each
integral to its precision-specific symbol.  The key architectural fact makes this a pure
data merge, not a source edit:

  ``run_app<TOutput,TMass,TScale,Printer>`` compiles ALL 21 integrals into one binary at
  ONE precision, and the integrals are **independent** — each seeds its own
  ``mt19937(12345)`` and shares no state — so a given integral's coefficient stream is
  identical whether it is dispatched from the whole-app binary or from a per-group binary
  restricted to its mass group.  Dispatch therefore = *choose which binary's RES stream
  supplies each integral*, then union the per-integral coeff arrays.

That is exactly what :func:`dispatch_and_aggregate` does: it runs the vanilla binary for
the non-promoted integrals and each promoted group's dd binary for its integrals, then
merges the per-integral :data:`CoeffArrays` — promoted integrals taken from their dd
binary, all others from vanilla.  **No driver source is mutated** (STOP #UU): the vanilla
driver is untouched and the dd drivers are pipeline-generated build artifacts
(deliverable 2), not edits to a user driver.  The merge is keyed by integral name, the
same key ``runner.run_and_aggregate`` already uses, so a promoted integral's dd stream
drops straight into the Validator's scorer with no format change.

Extensible for Phase 2/3: a :class:`DispatchPlan` maps each integral to a *source*
(a binary + the precision it was built at).  Phase-1 uses two precisions (vanilla double
+ dd); Phase 2 can add ff/float sources by adding more binaries to the plan — the merge
is precision-agnostic (it copies whatever ``(hi, lo)`` the source emitted).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from agents.patcher.precision_flip import TargetPrecision
from agents.validator.runner import CoeffArrays, run_and_aggregate


@dataclass(frozen=True)
class BinarySource:
    """One built binary + the precision it was compiled at, and the integrals it owns.

    ``integrals`` are the integral names whose RES stream should be taken from this
    binary in the merge.  A binary emits every integral it was compiled with, but the
    plan only *keeps* the ones it owns — so a per-group dd binary owns only its group's
    promoted integrals even though its pruned BO also emits zero-filled RES for the rest.
    """

    binary: Path
    precision: TargetPrecision | str      # "double" for the vanilla baseline
    integrals: tuple[str, ...]
    label: str = ""


@dataclass(frozen=True)
class DispatchPlan:
    """The per-integral binary→precision routing for one measurement pass.

    ``baseline`` is the vanilla (double) binary that owns every integral NOT claimed by a
    flip source; ``flips`` are the per-group dd (future ff/float) binaries that own their
    promoted integrals.  Construction validates that no integral is claimed twice.
    """

    baseline: BinarySource
    flips: tuple[BinarySource, ...] = ()

    def __post_init__(self) -> None:
        seen: dict[str, str] = {}
        for src in self.flips:
            for name in src.integrals:
                if name in seen:
                    raise ValueError(
                        f"integral {name!r} claimed by two flip sources "
                        f"({seen[name]!r} and {src.label!r})")
                seen[name] = src.label

    @property
    def promoted(self) -> set[str]:
        """Integrals routed to a flip source (everything else falls to the baseline)."""
        return {n for src in self.flips for n in src.integrals}

    def source_for(self, integral: str) -> BinarySource:
        """The binary source that owns ``integral`` (a flip source, else the baseline)."""
        for src in self.flips:
            if integral in src.integrals:
                return src
        return self.baseline


@dataclass
class DispatchResult:
    """Merged per-integral coeff arrays + provenance (which precision each came from)."""

    coeffs: CoeffArrays
    provenance: dict[str, str] = field(default_factory=dict)   # integral -> precision str


def _precision_str(p: TargetPrecision | str) -> str:
    return p.value if isinstance(p, TargetPrecision) else str(p)


def dispatch_and_aggregate(plan: DispatchPlan, total: int, *,
                           chunk: int = 0, workers: int = 1) -> DispatchResult:
    """Run every binary in ``plan`` and merge per-integral RES streams by ownership.

    Each source binary is run over ``[0, total)`` via the shared
    :func:`agents.validator.runner.run_and_aggregate` (byte-identical sampling to the
    single-precision path), then the promoted integrals are taken from their flip source
    and everything else from the baseline.  Returns a :class:`DispatchResult` whose
    ``coeffs`` is a drop-in for the scorer and whose ``provenance`` records the precision
    each integral was dispatched at (for the report / acceptance gate).

    A flip source that does not actually emit an integral it claims is a wiring bug
    (the generated group driver did not include that integral's group) — fail loud rather
    than silently fall back to the baseline stream, which would mask a non-promotion as a
    clean measurement.
    """
    # Baseline supplies every non-promoted integral.
    base_coeffs = run_and_aggregate(plan.baseline.binary, total,
                                    chunk=chunk, workers=workers)
    promoted = plan.promoted
    merged: CoeffArrays = {name: arr for name, arr in base_coeffs.items()
                           if name not in promoted}
    provenance: dict[str, str] = {name: _precision_str(plan.baseline.precision)
                                  for name in merged}

    for src in plan.flips:
        src_coeffs = run_and_aggregate(src.binary, total, chunk=chunk, workers=workers)
        for name in src.integrals:
            if name not in src_coeffs:
                raise ValueError(
                    f"flip source {src.label!r} ({src.binary}) was expected to emit "
                    f"integral {name!r} but its RES stream has none — the generated "
                    f"group driver did not cover it (wiring bug); refusing to fall back")
            merged[name] = src_coeffs[name]
            provenance[name] = _precision_str(src.precision)

    return DispatchResult(coeffs=merged, provenance=provenance)
