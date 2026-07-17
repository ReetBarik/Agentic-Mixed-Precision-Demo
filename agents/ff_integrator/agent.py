"""FF-integrator agent — STUB (regional float-float promotion).

Sibling of :mod:`agents.dd_integrator` and :mod:`agents.tracked_integrator`, but
with a deliberately **different scope and API**.  Where the tracked and dd
integrators are *whole-app* (instrument / DD-promote an entire library so it is
callable with an instrumented / extended type), the ff integrator is
*regional*: the Patcher hands it a specific code region — a file + line range and
the set of variables in play — and it promotes **just that region** to a
float-float (``ffloat`` / ``ffcomplex``) representation, producing a small shim
plus a boundary patch that converts at the region's edges and leaves the rest of
the app untouched.

Float-float ("single-single") emulates ~double precision from two ``float``s; the
strategy is the catalog's "float-float emulation (DD recovery)" entry — recover
lost precision in a hot region cheaply, without paying whole-app double-double.

=============================  STUB STATUS  ================================

This is a STUB with **no implementation**.  It is deferred until the Patcher
agent lands, because the Patcher is what defines a "region" (file/line range +
variable set) and drives the promote→build→validate loop that would call this.
Until then :func:`integrate` raises ``NotImplementedError`` rather than guessing
at an API the Patcher hasn't pinned down.

===========================  DEFERRED REAL WORK  ==========================

When built out, the ff integrator will:

* accept a **region descriptor** from the Patcher — ``(file, line_start,
  line_end, variables)`` — not a whole-header directory;
* emit an ``ffloat`` / ``ffcomplex`` shim for the types/ops that region uses,
  and a **boundary patch** that promotes region inputs to float-float on entry
  and demotes results on exit (the region-scoped analogue of the whole-app C8
  boundary patch);
* reuse the shared plumbing in :mod:`agents.integrator_base` where it fits
  (SOURCE_HASH cache keyed on region+ruleset, streaming shim, bounded retry
  loop, and a C8-style boundary patcher parameterized on the ``ffloat`` scalar
  type name) — the same reuse pattern the real dd_integrator will follow.

The API therefore differs from the whole-app integrators (region-scoped inputs,
a boundary patch as a first-class output) and is intentionally left unspecified
here until the Patcher's region contract exists.  See PLAN_overview.md's strategy
catalog (float-float emulation / DD recovery) and the Patcher section.
"""

from __future__ import annotations

_NOT_IMPLEMENTED_MSG = (
    "ff_integrator is a STUB: regional float-float promotion is deferred until "
    "the Patcher agent lands (it defines the code-region contract this integrator "
    "consumes). See this module's docstring for the intended regional API "
    "(file/line range + variables -> ffloat/ffcomplex shim + boundary patch)."
)


def integrate(*args, **kwargs):
    """Not implemented — see module docstring.

    Placeholder entrypoint kept for structural symmetry with the tracked / dd
    integrators.  The real signature will be region-scoped (a Patcher-supplied
    ``(file, line_start, line_end, variables)`` descriptor), not the whole-app
    ``(target_library_headers, driver_source_path, …)`` shape, so it is left
    unspecified until the Patcher's region contract exists.
    """
    raise NotImplementedError(_NOT_IMPLEMENTED_MSG)
