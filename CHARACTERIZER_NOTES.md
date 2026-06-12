# Notes from manual micro-driver: input for the characterizer prompt

What we did by hand to build `cln_micro` that the characterizer agent will
need to do automatically.  Keep this generalizable — do **not** bake QCDLoop
specifics into the prompt.

## Generalizable steps the characterizer must perform

For each target kernel, given (a) source files, (b) kernel name, (c) per-arg
input ranges, (d) build instructions:

1. **Determine which math operations in the kernel are templatable on
   `Tracked<T>` and which are not.**
   - Templatable: anything that resolves through ADL or your own dispatcher
     once `Tracked<T>` overloads are visible.
   - Non-templatable: hardcoded calls into framework / standard / vendor
     namespaces — `std::sin`, `Kokkos::log`, `cuda::sqrt`, vendor BLAS, etc.
     These need either an interop shim (preferred, see below) or opaque
     wrapping (fallback).

2. **For each non-templatable call, choose a strategy:**
   - **Interop shim** (preferred): emit overloads in the target framework's
     namespace mapping `std::log(tracked::T)` → `tracked::log(tracked::T)`,
     etc.  Preserves visibility into the math.  Document that this is
     technically namespace-injection UB but works on all major compilers.
   - **Opaque wrap** (fallback): when the framework call cannot be cleanly
     overloaded, or when the agent decides the interior numerics of the
     library function don't matter for this characterization run.  Calls
     `tracked::opaque(fn_name, raw_result, in1, in2, ...)` passing the
     tracked inputs so error + provenance propagate through the boundary.

3. **Decide which headers to `#include` vs which to reimplement minimally
   in the micro-driver.**
   - Including a header that internally calls non-templatable framework math
     on `Tracked<T>` will cause hard compile errors at template instantiation
     time.  Two ways out:
     a. Include the header and apply the interop shim **before** any
        instantiation site is reached.  Shim must be in scope at the
        point of call, not just at the point of declaration.
     b. Skip the include and reimplement the minimal surface the kernel
        needs inside the micro-driver.  Faster to get working but doesn't
        scale to kernels that depend on a large helper surface.
   - First-pass policy: try (a).  If the shim doesn't suffice (e.g., the
     framework header has internal calls the shim doesn't cover), fall
     back to (b) for the offending helpers only.

4. **Generate the micro-driver `.cpp`.**
   - Includes: framework headers, `<tracked/*.hpp>`, interop shim header.
   - For each input argument of the kernel, sample a value from the user's
     declared range and wrap with `tracked::track("<arg_name>", value)`.
   - Use `tracked::Complex<T>` for any complex-valued kernel argument
     (instead of `std::complex<T>` or `Kokkos::complex<T>`).
   - Call the kernel.
   - Print the result (helps the agent confirm the run produced sensible
     numbers even before parsing the JSONL).
   - Call `tracked::journal::flush("<run_id>.jsonl")`.

5. **Generate a `CMakeLists.txt`** (or invoke the build/run agent with a
   build spec) that links Tracked headers, the framework, and the user's
   build environment.  Build/run agent owns this; characterizer hands off
   the driver source plus build spec.

6. **Run, then parse the JSONL.**

## Specific pitfalls hit during the manual run

Worth surfacing in the characterizer prompt as known failure modes:

1. **Header pollution.** Including `kokkosMaths.h` and `kokkosUtils.h`
   pulled `KOKKOS_INLINE_FUNCTION` annotations and forced `Kokkos::log` /
   `Kokkos::abs` instantiations on `tracked::Complex<T>`.  Hard compile
   error.  Fix: reimplement the minimal `ql::` surface (`Imag`, `Real`,
   `Sign`, `Constants<T>`) in the micro-driver and skip the includes.
   Generalizable lesson: prefer reimplementing tiny helper surfaces over
   wrestling with header chains, when the helpers are <50 LoC.

2. **`constexpr` + non-literal type.** `tracked::Tracked<T>` is not a
   literal type (holds a `std::set<std::string>`).  Any `constexpr T foo()`
   in user-side helpers becomes ill-formed when `T = Tracked<...>`.  The
   characterizer must either strip `constexpr` from copied helpers or
   reimplement them as plain `inline`.

3. **Comparison-against-raw-value type mismatch.** User helpers often
   return raw `T` (e.g., `T Imag(complex<T>)`) and the kernel compares
   against another raw `T`.  When the kernel is instantiated on Tracked
   types, the comparison becomes `T == Tracked<T>` which doesn't exist.
   Fix: helpers must return `Tracked<T>` to keep both sides consistent.

4. **Provenance attribution through opaque calls.**  Initial implementation
   of `tracked::opaque` lost the input provenance chain (result carried only
   the fn_name, severing attribution).  Fixed by passing tracked inputs:
   `tracked::opaque(fn_name, raw_result, tracked_in1, tracked_in2)`.  The
   characterizer must always forward the tracked inputs to opaque wrappers
   so the agent can later trace "this output came from variables {a, b}
   passed through Kokkos::log" rather than just "this came from Kokkos::log."

5. **Opaque cond = 1, not 0.**  Cond = 0 would falsely tell the downstream
   strategy agent that opaque calls are perfectly stable.  The conservative
   default is cond = 1 (pass-through).  Characterizer doesn't choose this
   directly — Tracked enforces it — but the agent should understand that
   opaque records carry no information about the interior numerics of the
   wrapped function, only error pass-through.  When characterization output
   is dominated by opaque records, the agent should flag the kernel as
   under-characterized and recommend either rewriting the offending calls
   through interop or expanding Tracked's op coverage.

## Anti-patterns to avoid in the prompt

- **Do not** name specific user libraries (QCDLoop, Kokkos, SYCL).  Reason
  about "the user's framework" generically.  Examples in the prompt should
  vary the framework name on each invocation or use placeholders.
- **Do not** assume the kernel is complex-valued, real-valued, scalar,
  array-typed, or any particular shape.  Detect from the kernel signature.
- **Do not** assume `std::` or any specific math namespace.  The interop
  shim list is per-framework; detect which namespaces the kernel calls into
  and shim those.
- **Do not** hardcode sample counts, ranges, or strategies in the prompt.
  Those come from the user-supplied config.

## Suggested prompt structure (rough)

```
You are the characterizer agent.  You receive:
  - kernel_signature: function name and parameter types
  - source_files: paths to user source
  - input_ranges: per-argument (min, max)
  - framework: detected build framework (kokkos|sycl|openmp|plain-cpp|...)
  - existing_dispatchers: list of user-side math dispatcher functions
                          (e.g., {"kAbs", "kLog", "kSqrt"} for QCDLoop)

Produce:
  - micro_driver.cpp source
  - cmake_target_spec (handed to build/run agent)

Strategy:
  1. Identify all math operations the kernel performs.
  2. For each, decide: directly templatable / shim via interop /
     opaque wrap.  Default to interop; fall back to opaque only when the
     interior numerics don't materially affect characterization (justify).
  3. For helper headers the kernel includes, decide:
     - include directly if the shim covers all internal calls,
     - else reimplement minimal helper surface inline.
  4. Generate driver with sampled inputs, tracked::track per argument,
     tracked::Complex<T> for complex args, journal flush at end.
  5. Return the artifacts plus a one-paragraph rationale of which strategy
     you picked for each non-templatable call.
```

Refine as the agent gets real reps; this is the v0 sketch.
