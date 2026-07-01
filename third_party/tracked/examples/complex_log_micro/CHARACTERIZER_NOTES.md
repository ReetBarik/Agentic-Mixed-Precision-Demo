# Notes from the manual micro-driver: generalizing the pattern

What we did by hand to build `complex_log_micro` that an automated
characterizer would need to do for an arbitrary kernel. Keep this
generalizable — do **not** bake in any one library's specifics.

## Generalizable steps

For each target kernel, given (a) source files, (b) kernel name, (c) per-arg
input ranges, (d) build instructions:

1. **Determine which math operations in the kernel are templatable on
   `Tracked<T>` and which are not.**
   - Templatable: anything that resolves through ADL or your own dispatcher
     once `Tracked<T>` overloads are visible.
   - Non-templatable: hardcoded calls into framework / standard / vendor
     namespaces (e.g. `std::sin`, a framework's `log`, a vendor BLAS call).
     These need either an interop shim (preferred) or opaque wrapping
     (fallback).

2. **For each non-templatable call, choose a strategy:**
   - **Interop shim** (preferred): emit overloads in the target framework's
     namespace mapping `framework::log(tracked::T)` → `tracked::log(tracked::T)`,
     etc. Preserves visibility into the math. (Technically namespace-injection
     UB, but works on all major compilers.)
   - **Opaque wrap** (fallback): when the framework call cannot be cleanly
     overloaded, or when the interior numerics don't matter for this run. Call
     `tracked::opaque(fn_name, raw_result, in1, in2, ...)`, forwarding the
     tracked inputs so error + provenance propagate through the boundary.

3. **Decide which headers to `#include` vs reimplement minimally.**
   Including a header that internally calls non-templatable framework math on
   `Tracked<T>` causes hard compile errors at instantiation time. Either apply
   the interop shim *before* any instantiation site is reached, or skip the
   include and reimplement the minimal surface the kernel needs.

4. **Generate the micro-driver `.cpp`.** Include framework headers,
   `<tracked/*.hpp>`, and the interop shim; sample each input from its declared
   range and wrap with `tracked::track("<arg_name>", value)`; use
   `tracked::Complex<T>` for complex args; call the kernel; print the result;
   `tracked::journal::flush("<run_id>.jsonl")`.

5. **Generate a build spec** that links the Tracked headers, the framework, and
   the user's build environment.

6. **Run, then parse the JSONL.**

## Specific pitfalls (known failure modes)

1. **Header pollution.** Including framework math/util headers pulls in
   framework annotations and forces instantiations on `tracked::Complex<T>`.
   Hard compile error. Fix: reimplement the minimal helper surface inline and
   skip the includes. Lesson: prefer reimplementing tiny helper surfaces over
   wrestling header chains when the helpers are < ~50 LoC.

2. **`constexpr` + non-literal type.** `tracked::Tracked<T>` is not a literal
   type (holds a `std::set<std::string>`). Any `constexpr T foo()` in copied
   helpers becomes ill-formed when `T = Tracked<...>`. Strip `constexpr` or
   reimplement as plain `inline`.

3. **Comparison-against-raw-value mismatch.** Helpers that return raw `T` make
   the kernel's comparison become `T == Tracked<T>`, which doesn't exist. Fix:
   helpers must return `Tracked<T>` to keep both sides consistent.

4. **Provenance attribution through opaque calls.** An early `opaque`
   implementation lost the input provenance chain (the result carried only the
   fn_name). Fixed by passing the tracked inputs:
   `tracked::opaque(fn_name, raw_result, tracked_in1, tracked_in2)`. Always
   forward the tracked inputs so you can later trace "this output came from
   variables {a, b} passed through <fn>" rather than just "this came from <fn>."

5. **Opaque cond = 1, not 0.** Cond = 0 would falsely imply opaque calls are
   perfectly stable. The conservative default is cond = 1 (pass-through).
   Opaque records carry no information about the interior numerics of the
   wrapped function — only error pass-through. When output is dominated by
   opaque records, flag the kernel as under-characterized and recommend either
   rewriting the offending calls through interop or expanding op coverage.

## Anti-patterns to avoid when generalizing

- **Do not** name a specific user library. Reason about "the user's framework"
  generically; use placeholders.
- **Do not** assume the kernel is complex-valued, real-valued, scalar, or
  array-typed. Detect from the kernel signature.
- **Do not** assume `std::` or any specific math namespace. Detect which
  namespaces the kernel calls into and shim those.
- **Do not** hardcode sample counts, ranges, or strategies. Those come from the
  user-supplied config.
