# Stage 2: 3-/4-internal-mass boxes (B16, BIN3, BIN4) — C8 coverage

**Status:** validated — coverage extended via C8 library annotations, **9 sites
patched**. B16, BIN3, BIN4 compile and validate cleanly.
**Original finding:** 2026-07-14 (out of scope). **Extended:** 2026-07-15.
**Branch:** `langgraph-agents`.

## Summary

The Stage-2 leaf sweep instruments `ql::BO<TOutput, TMass, TScale>` with
`T = Tracked<double>` via a generated interop shim, one box configuration per
target. Targets B1–B15 route to the **B0m / B1m / B2m** families (0, 1, 2
nonzero internal masses) and validate with the shim alone. B16 (and BIN3, BIN4)
route to **B3m / B4m** (3 and 4 internal masses), whose bodies contain
`int`↔`Tracked` crossings that a free-function shim cannot bridge.

Originally these were declared out of scope. They are now **in scope**: the
integrator emits a small **library patch** (`<app>.patch`) alongside the shim
that makes each crossing explicit with an annotation that is a no-op on the pure
`double` build and journal-transparent under Tracked. The crossings are found by
the **compiler**, not the LLM, and mapped to source edits deterministically — see
"How C8 bridges it" below. The 3-/4-mass boxes use an **un-pruned copy** of the
header tree, `runs/qcdloop_headers_full/` (B3m/B4m restored in `boxGPU.h`); the
locked B13 spike tree stays pruned and untouched.

## The nine crossing sites (all in the statically-instantiated B3m/B4m graph)

Every target that includes the un-pruned `boxGPU.h` statically instantiates both
`B3m.h` and `B4m.h`, so all three targets share the **same 9-site patch** (the
derived `ql_tracked.patch` is byte-identical across B16, BIN3, BIN4).

| # | Site | Pattern | Crossing | Annotation |
|---|------|---------|----------|------------|
| 1–3 | `box/B3m.h:68,69,70` | (a) | `ir12/ir14/ir24 = _ten()*Sign(...)` — `Tracked` RHS assigned to `int` | wrap RHS: `static_cast<int>((RHS).value())` |
| 4–8 | `box/B3m.h:110,111,113,114,117` | (b) | `int`/`-int` `ir*` passed to `xspence/xetatilde(TScale const&)` | wrap arg: `tracked::Tracked<double>(ir*)` |
| 9 | `box/B4m.h:172` | (c) | `ql::Imag(r13) == 0` — `Tracked == int` literal | `.value()` on the tracked side: `(ql::Imag(r13)).value() == 0` |

`B4m.h`'s `ir13/ir24/ir1324` are already `TScale` and its line 102 compares
against `_zero()` (a `Tracked`), so only the bare-literal `== 0` at line 172
crosses.

## Root cause (unchanged)

`Tracked<double>` deliberately omits the implicit `int`↔scalar bridges that
pure-`double` code relies on:

- its scalar constructor is `explicit Tracked(T v)` — see
  `third_party/tracked/include/tracked/tracked.hpp:146`;
- it defines **no** `operator int` / `operator bool`, and its `operator==`
  (`tracked.hpp:176`) is a member taking `const Tracked&`.

qcdloop's B3m/B4m compute the branch-cut / iε-region indicators `ir12, ir14,
ir24` as **`int` flags derived from a tracked-typed expression** and then
**consume them as the working scalar type** (`xspence(..., TScale const&)`). In
pure `double` this round-trips silently through implicit `int↔double`
conversions; with `TScale = Tracked<double>` those conversions are intentionally
absent, so the pattern fails to compile at exactly the nine sites above (and
nowhere else — the shim handles the entire rest of the B3m/B4m call graph).

## How C8 bridges it (deterministic, compiler-error-driven)

C8 is **not** an LLM task and does **not** change shim generation (the system
prompt stays at Rules 1–9 + C1–C7). Instead:

1. The shimmed target is built once against the un-pruned tree with **no patch**.
   The compiler is a perfect, reproducible detector: it names each un-shimmable
   crossing by `file:line` and by kind (`cannot convert Tracked to int in
   assignment` / `invalid initialization of reference … from int` / `no match for
   operator== (Tracked, int)`).
2. `agents/tracked_integrator.derive_c8_patch(compile_stderr, headers_dir,
   repo_root)` maps each recognized diagnostic to a mechanical rewrite — using
   gcc's caret ruler for the (a)/(c) spans and the `in passing argument N` note +
   a balanced argument splitter for (b) — and synthesizes a git-apply-able
   unified diff via `difflib` with an **exactly-once** `original` guard.
3. `build_and_run` applies the patch (`git apply`) and rebuilds; the library tree
   is reset to pristine in a `finally`, so the committed `qcdloop_headers_full`
   copy is never left modified. The whole step is idempotent.

An `int`↔`Tracked` diagnostic that fits none of the three patterns is a hard
failure (`C8_UNCLASSIFIED_ERROR`) surfaced for human review — the same
ambiguity-surfacing discipline as the ruleset's UNCLASSIFIED escape hatch. Any
missed crossing simply fails the rebuild with a clear diagnostic, so the compiler
is also the exhaustiveness backstop.

The annotations preserve exact semantics: the crossed values are discrete branch
tags with no rounding to track, so `.value()` / `Tracked(...)` are transparent
boundary markers. **No Tracked API change and no qcdloop upstream change** — the
patch lives in integrator output and is applied only at build time. This upholds
the standing rule that Tracked is the reusable component shared across
applications and must not be distorted to accommodate one consumer.

## Validation (256 samples each)

All three pass the refined cond gate: `cond>1e15` only under (a) the documented
`atan2` saturation cap at `2^53` and (b) genuine `sub`/`add` catastrophic
cancellation; **0 "other"**. 0 empty operand slots, 0 UNCLASSIFIED. Vocab delta
vs the B13 reference is `+atan2` (the 3-/4-mass complex-arg path reaches
`arg(z)=atan2`).

| Target | dispatch | records | cond>1e15 (a atan2 / b cancel / other) | top rel_err hotspot |
|--------|----------|---------|----------------------------------------|---------------------|
| B16  | B3m→B16()  | 333,737 | 6912 / 0 / 0    | `add`  2.35e8 @ cond 8.9e3 |
| BIN3 | B3m→BIN3() | 579,737 | 9146 / 29 / 0   | `sub`  1.98e22 @ cond 5.6e4 |
| BIN4 | B4m→BIN4() | 988,752 | 15483 / 50 / 0  | `sub`  5.37e25 @ cond 87.7 |

The `sub`/`add` hotspots are the intended catastrophic-cancellation signal —
amplification driven by `|a−b|→0`, sometimes at low apparent `cond` (BIN4:
rel_err 5.37e25 at cond 87.7). Its remediation is Kahan / algebraic reformulation
upstream, not extended precision at the op.

## Bounds of shim-only + C8 integration

- **In scope (validated):** B0m / B1m / B2m (shim only) **and** B3m / B4m
  (shim + C8 library patch) — B1–B16 and BIN0–BIN4.
- The B0m/B1m/B2m targets are unaffected by C8 (they compile clean, no patch);
  regression spot-checks B7/B12/BIN0 confirm byte-identical journal shapes.
