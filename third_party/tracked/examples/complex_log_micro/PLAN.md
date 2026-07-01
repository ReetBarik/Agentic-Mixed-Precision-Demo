# Plan: Manual `complex_log` micro-driver against Tracked

## Goal

Stand up a one-off micro-driver that instruments a representative `complex_log`
(complex log with branch-cut handling) kernel against the
`tracked::Complex<double>` / `tracked::Tracked<double>` types. Run a handful of
sample inputs, flush the JSONL journal, eyeball the records to confirm Tracked
is wired correctly into a complex-valued kernel that calls framework math.

Scope is tiny on purpose: this is a manual sanity check of the instrumentation
pattern, not an automated characterization pipeline.

## Strategy

The kernel calls `kLog(z)` and `kAbs(z)`, which in turn call `Kokkos::log` /
`Kokkos::abs`. Those framework free functions have no overload for
`tracked::Complex<T>` (would be a hard compile error). For this manual check we
**deliberately treat the framework math calls as opaque**:

- `kLog(Complex)` → call `Kokkos::log` on the raw underlying value, wrap the two
  result components as `tracked::opaque("Kokkos::log", ...)`, forwarding the
  tracked inputs so error + provenance propagate through the boundary.
- `kAbs(Complex)` → same pattern with `Kokkos::abs`.

We lose visibility into the numerics inside `Kokkos::log`. That's the deliberate
limitation of the opaque-barrier approach — the alternative (an interop shim
adding overloads in the framework namespace) preserves visibility but is out of
scope for this manual check.

We also **do not include the original framework helper headers directly.** They
pull in `Kokkos::complex` and `KOKKOS_INLINE_FUNCTION` everywhere, and their
templated function bodies trigger the `Kokkos::log(tracked::Complex)` error at
instantiation time. Instead, we **copy the kernel body** into the micro-driver
and stub the few helpers it needs (a minimal `extmath::` surface).

## Files

```
examples/cln_micro/
  complex_log_micro.cpp   # the driver
  CMakeLists.txt          # standalone build (needs Kokkos)
  README.md               # what this is, how to run, what to expect
  build_kokkos_serial.sh  # helper to build a Serial-only Kokkos
```

Lives under `examples/cln_micro/`. **Not** built by the top-level
`CMakeLists.txt` — the core library is dependency-free, while this driver needs
Kokkos. Build it from inside this directory.

The full driver source is `complex_log_micro.cpp`; see it for the exact
`extmath::` stubs and the three sample inputs.

## Pitfalls hit (and the generic lesson each teaches)

1. **Header pollution.** Including the framework's math/util headers forced
   `Kokkos::log` / `Kokkos::abs` instantiations on `tracked::Complex<T>` — a
   hard compile error. Fix: reimplement the minimal helper surface inline and
   skip the includes. Lesson: prefer reimplementing tiny helper surfaces over
   wrestling header chains when the helpers are < ~50 LoC.

2. **`constexpr` + non-literal type.** `Tracked<T>` is not a literal type (it
   holds a `std::set<std::string>`). Any `constexpr T foo()` in copied helpers
   becomes ill-formed when `T = Tracked<...>`. Strip `constexpr` to plain
   `inline`.

3. **Comparison against raw value.** Helpers that return raw `T` make the
   kernel's comparison become `T == Tracked<T>`, which doesn't exist. Fix:
   helpers must return `Tracked<T>` to keep both sides consistent.

4. **Provenance through opaque calls.** Pass the tracked inputs to
   `tracked::opaque(fn_name, raw_result, in1, in2, ...)` so error + provenance
   propagate across the boundary, rather than severing attribution at the
   black box.

## What to validate

After running, check `complex_log.jsonl`:

1. **Sample 1** (`z=1.5+0.3i`): two `op="opaque"` records with
   `fn_name="Kokkos::log"` (one per output component). Confirms the opaque path.
2. **Sample 2** (`z=-1+1e-15·i`): `Imag(z)==0` is false (`1e-15 != 0`), so the
   `else` branch runs. Same two opaque records, nothing else — **the case where
   opaque hides the interesting behavior.** Confirms the limitation.
3. **Sample 3** (`z=-2+0i`): the branch-cut `if` triggers. Journal shows `neg`
   on `z`, the two `opaque` records for `Kokkos::log(-z)`, a `mul` for
   `π · Sign(branch_sign)`, and `add` records for the complex addition.

## Out of scope

- Replacing the opaque wrappers with a visible interop shim.
- Driving many samples in a loop (single-shot per input is enough here).
- Building this as part of the core test suite (it depends on Kokkos, which the
  core library deliberately does not).
