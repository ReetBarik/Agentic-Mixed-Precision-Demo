# Stage 2 boundary finding: 3-/4-internal-mass boxes (B16, BIN3, BIN4)

**Status:** not validated — out of scope for shim-only Tracked integration.
**Date:** 2026-07-14. **Branch:** `langgraph-agents`.

## Summary

The Stage-2 leaf sweep instruments `ql::BO<TOutput, TMass, TScale>` with
`T = Tracked<double>` via a generated interop shim, one box configuration per
target. Targets B1–B15 route to the **B0m / B1m / B2m** families (0, 1, 2
nonzero internal masses) and validate cleanly. B16 (and BIN3, BIN4) route to
**B3m / B4m** (3 and 4 internal masses), which the shared B13 header tree
deliberately prunes.

Un-pruning the box dispatch in an isolated copy of the header tree
(`box/B3m.h`, `box/B4m.h` + the `massive == 3 / == 4` branches in `boxGPU.h`)
and regenerating the shim against it confirms — mechanically — that these
boxes **cannot be instrumented by a shim of free functions/overloads** under
the project's standing constraints (do not modify Tracked upstream; the shim
may not edit library source). The B13 pruning was correct; this document
records *why*, with the exact failing sites.

## The three failing sites

Building B16's driver against the un-pruned tree with a freshly generated shim
fails to compile with three distinct errors, all in the newly-reached box code:

| # | Site | Code (paraphrased) | Nature |
|---|------|--------------------|--------|
| 1 | `box/B3m.h:64, 68–70` | `int ir12 = ql::Constants<TScale>::_ten() * ql::Sign(...)` — RHS is `Tracked<double>`, assigned to an `int` flag | `Tracked → int` in an assignment |
| 2 | `box/B3m.h:110–117` | `ql::xspence<...>(x4, ix4, r14, ir14)` — `ir14` is `int`; the parameter is `TScale const&` (`const Tracked<double>&`) | `int → Tracked` reference binding |
| 3 | `box/B4m.h:172` | `if (ql::Imag(r13) == 0)` — `Tracked<double> == int` | missing `operator==(Tracked, int)` |

These are representative, not exhaustive: the same `ir12 / ir14 / ir24`
"iε-region" pattern recurs throughout B3m/B4m, so clearing the first errors
only surfaces more of the same class.

## Root cause

`Tracked<double>` deliberately omits the implicit int↔scalar bridges that
pure-`double` code relies on:

- its scalar constructor is `explicit Tracked(T v)` — see
  `third_party/tracked/include/tracked/tracked.hpp:146`;
- it defines **no** `operator int` / `operator bool` conversion, and its
  `operator==` (tracked.hpp:176) is a member taking `const Tracked&`.

qcdloop's B3m/B4m compute the branch-cut / iε-region indicators `ir12, ir14,
ir24` as **`int` flags derived from a tracked-typed expression**
(`_ten() * Sign(Real(...))`) and then **consume them as the working scalar
type** (`xspence(..., TScale const&)`). In pure `double` this round-trips
silently through implicit `int↔double` conversions. With `TScale = Tracked<double>`
those conversions are intentionally absent, so the pattern fails to compile.

## Why the shim cannot bridge it

- **#3** is the one case within the existing ruleset (**C3**: supply a missing
  operator as a free function in the Tracked namespace, found by ADL). A free
  `operator==(const Tracked<T>&, int)` returning `bool` via `.value()` would
  clear it.
- **#1** needs a `Tracked → int` conversion, which must be a **member**
  conversion operator on `Tracked` — a shim of free functions cannot supply it,
  and the assignment lives in library code the shim may not edit.
- **#2** needs an `int → Tracked` implicit conversion for reference binding,
  which is blocked by the **`explicit`** scalar constructor — again unreachable
  from a free-function shim.

So #1 and #2 are **structural**: fixing them requires changing either the
Tracked API or the library source. No shim variant (or retry) resolves them.
Because #1/#2 gate the same enclosing pattern that #3 sits inside, **#3 does not
earn a new rule** — a `==` overload is pointless when the surrounding block is
out of scope regardless. Ruleset stays at Rules 1–9 + C1–C7; no C8.

## Bounds of shim-only integration

- **In scope (validated):** B0m / B1m / B2m families — B1–B15 across the massless,
  1-mass, and 2-mass box branches.
- **Out of scope (this finding):** B3m / B4m families — B16, BIN3, BIN4
  (3- and 4-internal-mass boxes).

## Deferred path for 3-/4-mass support (no commitment)

If 3-/4-mass instrumentation is later required, the preferred fix is
**option (ii): surgical library edits in a qcdloop fork** — wrap the `ir*`
boundary sites with `.value()` / `TScale(...)` so the int↔tracked crossings are
explicit at the ~half-dozen locations. This is preferred over changing the
Tracked API: **Tracked is the reusable component shared across applications;
qcdloop is one specific consumer. Never distort the reusable piece to
accommodate one caller.** Making the scalar ctor non-explicit or adding a
`Tracked → int` conversion would erase exactly the type-safety Tracked exists
to provide, for every other application, to satisfy one library's internal
type-punning.
