# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with
code in this repository.

## Project Overview

This repository provides **`tracked::Tracked<T>`** — a header-only, host-only
C++17 library for **numerical sensitivity instrumentation**. It wraps a
floating-point type (primarily `double`) and, as arithmetic and math functions
are applied, propagates a first-order **relative-error bound** and the
**condition number** of each operation alongside the value, recording every op
to a structured JSONL journal.

The goal is to make it visible *where* a real computation loses accuracy —
catastrophic cancellation, swamping, ill-conditioned branches — by replacing
`double` with `Tracked<double>` in a kernel and inspecting the journal.

This is **not** a GPU or extended-precision project. There is no Kokkos, CUDA,
or quadmath dependency in the core library. (One optional example under
`examples/` uses Kokkos to demonstrate instrumenting a kernel that calls
framework math; the library itself does not.)

## Build

The core library and tests have no external dependencies beyond a C++17
compiler and CMake (Catch2 is fetched automatically via `FetchContent`):

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc)
cd build && ctest --output-on-failure
```

## Architecture

### Header-only library (`include/tracked/`)

| Header | Contents |
|---|---|
| `tracked.hpp` | `unit_roundoff<T>()`, `Tracked<T>` type, free functions `add/sub/mul/div/neg`, `track()` factory, `opaque()` barrier |
| `journal.hpp` | `SourceLocation`, `TRACKED_HERE`, `LogRecord`, thread-local record buffer, `emit()` / `records()` / `clear()` / `flush()` |
| `ops.hpp` | math overloads with condition-number propagation: `sqrt`, `exp`, `log`, `abs`, `sin`, `cos`, `atan2` |
| `complex.hpp` | `tracked::Complex<T>` — two `Tracked<T>` reals; full complex arithmetic and math decomposed into real ops so each shows up in the journal |

Key design points:

- **No GPU / device support** — host-only by design; the journal is a
  thread-local `std::vector<LogRecord>`.
- **No flag field** — the condition number is the only signal; consumers apply
  their own threshold. Bits lost ≈ `log2(cond)`; decimal digits lost ≈
  `log10(cond)`.
- **Error-bound formula** (first-order):
  `new_rel_err = local_cond * (max(input_rel_errs) + unit_roundoff<T>())`.
- **Logging namespace is `tracked::journal`**, deliberately *not* `tracked::log`
  (which would collide with the `tracked::log(Tracked<T>)` math function).
- **`Complex<T>` emits no records of its own** — every complex op decomposes
  into real `Tracked<T>` ops that emit normal records; a complex-op-level view
  is recovered by rolling up the journal by source location. The one exception
  is `opaque(Complex<T>)`, which emits two `op="opaque"` records sharing one
  `fn_name`.

### Tests (`tests/tracked/`)

Catch2 calibration suite. Each file runs a known-problematic computation and
asserts `Tracked<T>` surfaces the problem (cancellation, naive variance,
log-sum-exp stability, Kahan vs naive, trig conditioning, complex
basic/sqrt/log/div). Registered in `CMakeLists.txt` via a `foreach` loop.

### Example (`examples/complex_log_micro/`)

An **optional, manually-written** micro-driver (not built by the top-level
CMake) showing how to instrument a real kernel that calls framework math. It
treats `Kokkos::log` / `Kokkos::abs` as **opaque barriers** — computing on raw
values, then wrapping the result with `tracked::opaque(...)` while forwarding
the tracked inputs so error and provenance still propagate through the black
box. Requires Kokkos; see its README.

### Design docs (`docs/`)

`docs/PLAN.md` (v1) and `docs/PLAN-v1.1.md` (complex support) are historical
design records describing the rationale behind the type and its API.
`docs/CONDITION_NUMBERS.md` derives every per-op condition number from first
principles, documents the error-propagation model and where it departs from the
strict textbook formula (e.g. `mul`/`div` use `cond=1`), and lists the
numerical-analysis literature (Higham, Trefethen & Bau, Goldberg, Smith, Kahan).

## Conventions

- Everything lives in `namespace tracked` (logging in `tracked::journal`).
- Math functions take an optional trailing `SourceLocation loc = {}`; use the
  `TRACKED_HERE` macro to capture `file:function:line`.
- When decomposing a complex op into real ops, use **named locals** (not nested
  temporaries) so each journal record gets a stable source location.
