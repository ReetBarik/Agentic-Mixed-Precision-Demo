# Tracked<T> — numerical sensitivity instrumentation library

**Status:** v1 — the initial design of the library (historical design record).

## Goal

Header-only C++ library providing `tracked::Tracked<T>` — a wrapper around a
floating-point type (primarily `double`) that overloads arithmetic and common
math functions to propagate per-op **condition number** and **accumulated
relative error bound** alongside the value, and emits a structured JSONL log of
every op performed.

Intended to be consumed by a downstream characterizer tool that produces
per-variable / per-line sensitivity profiles used to drive mixed-precision
optimization decisions.  Any such consumer lives in its own repository; this
repository provides only the instrumentation type and its tests.

## Repository layout

```
include/tracked/
  tracked.hpp        # Tracked<T> type, factory functions, named free functions
  journal.hpp        # logging infrastructure (tracked::journal namespace)
  ops.hpp            # math function overloads (sqrt, exp, log, abs)
tests/tracked/
  test_basic_ops.cpp       # operators, error propagation, provenance, source location, opaque
  test_cancellation.cpp    # catastrophic cancellation via cond >> 1
  test_naive_variance.cpp  # naive one-pass variance loses all digits when mean >> std dev
  test_log_sum_exp.cpp     # exp cond = |x|; stable shift keeps cond low
  test_kahan_vs_naive.cpp  # swamping invisible to first-order model; Kahan compensation intentional
CMakeLists.txt             # standalone — no Kokkos, no quadmath, no GPU demo targets
```

## Build

No module loading or external dependencies needed:

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build -j$(nproc) && cd build && ctest --output-on-failure
```

## Type design

```cpp
namespace tracked {

template <class T>
class Tracked {
  T                     value_;
  T                     rel_err_bound_;   // accumulated relative error bound
  T                     max_cond_seen_;   // max local cond on any op producing this
  std::set<std::string> provenance_;      // contributing source-variable ids
  // ... operator overloads, comparisons return plain bool ...
};

// Tag a fresh input with a source-variable id.
template <class T> Tracked<T> track(std::string id, T value);

// Opaque-call barrier: value crosses a non-Tracked boundary.
template <class T> Tracked<T> opaque(std::string_view fn_name, T value, SourceLocation loc = {});

} // namespace tracked
```

### Per-op condition numbers (closed form)

| Op | Local condition |
|---|---|
| `a + b` | `(\|a\| + \|b\|) / \|a + b\|` |
| `a - b` | same as add |
| `a * b` | 1 |
| `a / b` | 1 |
| `sqrt(x)` | 0.5 |
| `exp(x)` | `\|x\|` |
| `log(x)` | `1 / \|log(x)\|` (capped at `1/u` when `\|log(x)\| < u`) |
| `abs(x)` | 1 |
| `-x` | 1 |

### Error bound propagation

Standard first-order model. For each op:

```
new_rel_err = local_cond * (max(input_rel_errs) + unit_roundoff<T>())
```

where `unit_roundoff<double>() = epsilon/2 = 2^-53 ≈ 1.11e-16`.

### Design decision: no flag field

The condition number is the only signal.  A downstream consumer applies its own
threshold based on its precision requirements — the library does not hardcode
one.  Removed: `flag` field from `LogRecord`, `cancellation_threshold<T>()`.
How many bits are lost: `log2(cond)`.  How many decimal digits: `log10(cond)`.

## Logging: `tracked::journal`

Namespace is `tracked::journal` (not `tracked::log`, which conflicts with the
`tracked::log(Tracked<T>)` math function in `ops.hpp`).

- **Buffer**: thread-local `std::vector<LogRecord>`, unbounded.
- **API**: `journal::emit(...)` (called internally by every op), `journal::records()`, `journal::clear()`, `journal::flush(path)`.
- `flush` writes one JSONL record per op to `path`.

## Log schema (JSONL)

One record per op:

```json
{"op":"sub","at":"quadratic.hpp:quadratic_root:5","in":["a","b"],
 "val":5e-9,"cond":4e16,"rel_err":4.4e0,"prov":["a","b","c"]}
```

Fields:

- `op` — op name (`add`, `sub`, `mul`, `div`, `neg`, `sqrt`, `exp`, `log`, `abs`, `opaque`).
- `at` — `file:function:line` from `TRACKED_HERE` macro; empty string when not captured.
- `in` — primary provenance id per input operand (`"_"` for unnamed intermediates).
- `val` — output value (double precision; `null` for NaN, clamped for Inf).
- `cond` — local condition number for this op.
- `rel_err` — accumulated relative error bound on the output.
- `prov` — sorted union of input provenance sets.

## Source location

`SourceLocation{file, function, line}` struct + `TRACKED_HERE` macro.
Operators (`+`, `-`, etc.) call named free functions with `loc={}` (no location
captured).  For explicit location: `sub(a, b, TRACKED_HERE)`.

## Opaque barrier

`tracked::opaque("fn_name", value)` emits an `op="opaque"` record and returns a
fresh `Tracked<T>` carrying `fn_name` as its sole provenance.  Use it to wrap
any call that crosses a non-Tracked boundary so attribution stops cleanly there.

## Calibration suite — what was actually tested

Each test file runs a known-problematic computation and asserts `Tracked<T>`
surfaces the problem via condition number and/or rel\_err\_bound:

| File | Pathology | Key assertion |
|---|---|---|
| `test_cancellation` | `(1+eps)-1` for small eps | `max_cond_seen_ > 1e8` |
| `test_naive_variance` | `E[X²]-(E[X])²` with mean=1e8 | `max_cond_seen_ > 1e10`, FP result = 0 (true = 2/3) |
| `test_log_sum_exp` | `exp(100)` naive vs shifted | naive `max_cond ≥ 100`; stable `max_cond < 10` |
| `test_kahan_vs_naive` | Kahan compensation | swamping: `max_cond < 2` (missed); compensation: `max_cond > 1e8` (intentional) |

## Scope

In scope (complete):
- `Tracked<T>` type, header-only, C++17, host-only.
- Operator overloads: `+ - * /`, unary `-`, compound assignment, comparisons.
- Math overloads: `sqrt`, `exp`, `log`, `abs`.
- Condition number propagation (closed-form per op).
- Accumulated relative error bound.
- Flat provenance sets.
- Source location capture.
- Opaque-call barrier.
- Thread-local JSONL log buffer with flush.
- Calibration test suite (40 tests, all passing).

Not in scope for v1 (deferred):
- **GPU / device execution** — log buffers hostile to GPU; characterizer runs are host-side and small (hundreds of samples).
- **Ring-buffer / dedup** — unbounded `std::vector` is sufficient for v1 sample sizes; add capacity cap if needed.
- **Micro-driver template** — belongs in a downstream consumer repo, not here.
- **Weighted DAG provenance** — per-input weights for attributing final error to a specific variable.
- **Expanded op coverage** (`sin`, `cos`, `pow`, `atan2`, etc.) — add on demand as kernels need them.
- **Higher-order error models** — v1 uses first-order bounds.
