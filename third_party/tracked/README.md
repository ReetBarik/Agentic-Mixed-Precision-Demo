# Tracked — numerical sensitivity instrumentation for C++

`tracked::Tracked<T>` is a header-only, host-only C++17 wrapper around a
floating-point type (primarily `double`). As you do arithmetic and call math
functions on it, it propagates — alongside the value — a first-order
**relative-error bound** and the **condition number** of every operation, and
records each op to a structured JSONL journal.

Replace `double` with `tracked::Tracked<double>` in a computation and the
journal shows you *where* accuracy is lost: catastrophic cancellation, swamping,
ill-conditioned branches, and the like. There is no GPU, CUDA, quadmath, or
extended-precision machinery here — this is a diagnostics tool, not a
higher-precision number type.

## Quick start

```cpp
#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>
#include <tracked/journal.hpp>

using tracked::track;

int main() {
    auto a = track("a", 1.0 + 1e-12);
    auto b = track("b", 1.0);
    auto d = a - b;                       // catastrophic cancellation
    // d.value() ≈ 1e-12, but d.max_cond_seen() is huge → accuracy warning

    tracked::journal::flush("run.jsonl"); // one JSONL record per op
}
```

Each journal record looks like:

```json
{"op":"sub","at":"main.cpp:main:8","in":["a","b"],
 "val":1e-12,"cond":2e12,"rel_err":2.2e-4,"prov":["a","b"]}
```

`cond` is the local condition number of the op; bits lost ≈ `log2(cond)`,
decimal digits lost ≈ `log10(cond)`.

## Build & test

No external dependencies beyond a C++17 compiler and CMake (Catch2 is fetched
automatically):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

## What's included

- **`include/tracked/`** — the library:
  - `tracked.hpp` — `Tracked<T>`, `track()`, `opaque()`, arithmetic.
  - `ops.hpp` — `sqrt`, `exp`, `log`, `abs`, `sin`, `cos`, `atan2` with
    per-op condition numbers.
  - `complex.hpp` — `tracked::Complex<T>`, full complex math decomposed into
    real ops so each is visible in the journal.
  - `journal.hpp` — the JSONL journal (`tracked::journal` namespace).
- **`tests/tracked/`** — Catch2 calibration suite: each test drives a known
  numerical pathology and asserts `Tracked<T>` surfaces it.
- **`examples/complex_log_micro/`** — optional Kokkos-based micro-driver
  demonstrating the **opaque-barrier** pattern for kernels that call framework
  math you can't see into. Not built by the top-level CMake; see its README.
- **`docs/`** — design records (`PLAN.md`, `PLAN-v1.1.md`) and
  `CONDITION_NUMBERS.md` (full derivations of every per-op condition number,
  the error-propagation model, and the numerical-analysis references).

## Key ideas

- **Condition number is the only signal.** The library doesn't hardcode a
  "too unstable" threshold — consumers apply their own based on their precision
  needs.
- **Error model** (first-order):
  `new_rel_err = local_cond · (max(input_rel_errs) + unit_roundoff<T>())`,
  with `unit_roundoff<double>() = 2⁻⁵³ ≈ 1.11e-16`.
- **Opaque barriers** let you wrap a call that crosses a non-`Tracked`
  boundary (a vendor/framework math function): the interior is hidden as a
  single record, but error and provenance still propagate through.
- **Complex is two tracked reals.** `Complex<T>` emits no records of its own;
  every complex op decomposes into real ops, so you see each underlying
  `add`/`mul`/`sqrt`/etc. A complex-op view is recovered by grouping the
  journal by source location.

## License / provenance

The condition-number and error-propagation formulas are standard first-order
numerical-analysis results. See `docs/` for the full design rationale.
