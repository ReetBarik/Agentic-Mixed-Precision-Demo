"""Shared result type for the *regional* integrators (ff / dd).

Both ``ff_integrator.integrate_region`` and ``dd_integrator.integrate_region``
return this: a small shim (written under ``out_dir``) plus a boundary patch (a
unified diff, ``git apply -p1`` from the repo root) that promotes the region's
inputs to the extended scalar type on entry and demotes results on exit.  The
Patcher applies the boundary patch to the working tree and commits the shim +
patch together (P2).

``status`` is deliberately coarse — ``ok`` or ``llm_failed`` — mirroring the
bounded-retry contract in the design (P4): the Patcher retries ``llm_failed`` up
to N times, and any other integrator problem is surfaced as ``llm_failed`` with a
message so the caller never has to branch on a wider enum.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RegionIntegrationResult:
    status: str                                   # "ok" | "llm_failed"
    shim_paths: list[str] = field(default_factory=list)
    boundary_patch: str | None = None             # unified diff (git apply -p1)
    error: str | None = None
    llm_tokens: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @classmethod
    def failed(cls, error: str, *, llm_tokens: int = 0) -> "RegionIntegrationResult":
        return cls(status="llm_failed", error=error, llm_tokens=llm_tokens)
