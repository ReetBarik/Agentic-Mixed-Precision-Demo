# runs/B13/ — Tracked instrumentation feasibility spike

**Status:** feasibility spike (not Phase 0 proper).

**Goal:** answer three questions:

1. Does `ql::BO<Complex<Tracked<double>>, Tracked<double>, Tracked<double>>`
   compile against qcdloop@master given a small tracked interop shim, and
   correctly dispatch to `ql::B13` based on B13's mass configuration?
2. Does it link and run under a Serial Kokkos build?
3. Does one small batch produce a non-degenerate op journal (finite
   condition numbers, provenance flowing through the dilog family)?

If yes → proceed to Phase 0 per `todos/iteration1_phase0_b13.md`.
If no → the failure surface names the missing tracked overloads (or
qcdloop template constraints, or tracked host/device gaps) that need
work first.

## Design decisions

**Route through `ql::BO()`, not directly to `ql::B13()`.**
The reference for a working non-double precision swap in qcdloop is
[`boxGPU_test_dd_B13.cc` on the `ddfun_enabled` branch][dd_b13], which
fills `Views<TMass* [4]> m` and `<TMass* [6]> p` with B13's kinematics
(m1=m2=0, m3²=10, m4²=2500) and lets `BO()`'s dispatcher select B13
based on the mass count. We mirror that exactly so we don't have to
fabricate B13's internal `Y` matrix by hand.

[dd_b13]: https://github.com/ReetBarik/qcdloop/blob/ddfun_enabled/examples/box/boxGPU_test_dd_B13.cc

**Host loop, not `Kokkos::parallel_for`.**
The tracked library's ops (`third_party/tracked/include/tracked/`) are
host-only — no `KOKKOS_INLINE_FUNCTION` annotations, and the journal
uses `thread_local std::vector<LogRecord>`. A `KOKKOS_LAMBDA` closure
containing them would fail the device-function check even on the Serial
backend. Kokkos is still initialized because `BO()`'s body uses
`Kokkos::Array` and `Kokkos::View`. This matches `runs/cln/`,
`runs/lnrat/` prior art, which also loop on the host despite calling
`Kokkos::initialize`.

**Minimal `ql_tracked_interop.hpp`, not a full `kokkosMaths_tracked.h`.**
The proven pattern on `ddfun_enabled` is to swap out `kokkosMaths.h`
entirely via `kokkosMaths_wrapper.h`, shipping a parallel
`kokkosMaths_dd.h` that re-declares `Constants<T>`, `Real`, `Imag`,
`Sign`, `iszero`, `kPow`, `kAbs`, `kLog`, `kSqrt` for the DD types
(~390 lines). For the *spike*, we ship a much smaller
`ql_tracked_interop.hpp` that adds just the overloads B13 touches. If
the spike passes, Phase 0 will invest in the full
`kokkosMaths_tracked.h` per the DD/quad pattern.

## Layout

```
runs/B13/
├── CMakeLists.txt
├── qcdloop_headers/           # vendored from ReetBarik/qcdloop@master (see below)
│   ├── boxGPU.h
│   ├── kokkosMaths.h
│   ├── kokkosMaths_wrapper.h
│   ├── kokkosUtils.h
│   ├── timer.h
│   └── box/
│       ├── B0m.h              # BIN0, B1-B5   (0 internal masses)
│       ├── B1m.h              # BIN1, B6-B10  (1 internal mass)
│       ├── B2m.h              # BIN2, B11-B15 (2 internal masses — contains B13)
│       ├── B3m.h              # BIN3, B16     (3 internal masses)
│       ├── B4m.h              # BIN4          (4 internal masses)
│       └── box_common.h
├── src/
│   ├── micro_driver.cpp       # spike driver — mirrors boxGPU_test_dd_B13.cc pattern
│   └── ql_tracked_interop.hpp # ql:: overloads for tracked types (audit surface)
└── README.md
```

## Vendored qcdloop sha

`ReetBarik/qcdloop@master` at commit **`8de2089`** (2026-06-08).

The full `boxGPU.h` + all five `box/Bnm.h` groups are vendored so
`ql::BO()`'s dispatcher compiles cleanly. Only the B13 branch will be
exercised on tracked types by our kinematics, but the other branches
still get instantiated by the template — if any of them fail on
tracked, that's useful signal too.

**Divergence from `ddfun_enabled`:** the DD branch modified `B2m.h`
to hoist `TOutput inv_r14 = 1/r14` and `TOutput inv_xs = 1/xs` as
named locals (aliasing workaround for DD temporaries). The changes
are in BIN2 and BIN5, not B13 itself. B13's body is byte-identical
between `master` and `ddfun_enabled`, so master's `B2m.h` is safe
for the spike. If Phase 0 hits similar aliasing issues on tracked,
apply the same hoists.

**Why vendor rather than reuse `Agentic-Mixed-Precision-Demo/src/*`?**
The agentic repo's `kokkosMaths.h` has drifted from qcdloop@master —
specifically `kPow` no longer handles negative exponents. The spike must
faithfully represent production qcdloop behavior. Keeping a self-contained
vendored copy makes the divergence audit trivial. See
`qcdloop_headers/README.md` for the refresh procedure.

## ql_tracked_interop.hpp

Documents the exact surface of `ql::*` overloads that had to be added to
make B13 compile against tracked types. Two categories:

- **Ported from `runs/cln/` and `runs/lnrat/` prior art:**
  `kAbs`, `kLog` (both scalar and complex).
- **New for B13:**
  `Real`, `Imag`, `Sign`, `iszero`, `kSqrt`, `kPow` (scalar and complex
  where applicable).

Both categories follow the two idioms established by cln/lnrat:
`interop_shim` (delegate to instrumented tracked op) and `opaque_wrap`
(re-enter tracked domain via `tracked::opaque_at`). See the header for
per-overload rationale.

**Reference against `kokkosMaths_dd.h`** (from `ddfun_enabled`): the DD
implementation ships full explicit overloads for the same set. If any
of our tracked overloads misbehave, the DD implementation is the source
of truth for correct semantics.

## Build

Requires a Serial-only Kokkos install at `$HOME/kokkos-install` (the
tracked repo ships `examples/cln_micro/build_kokkos_serial.sh` if you
don't have one yet).

```sh
cd runs/B13
mkdir -p build && cd build
cmake -DCMAKE_PREFIX_PATH=$HOME/kokkos-install ..
make -j
```

## Run

```sh
./build/B13_spike             # 256 samples (default), writes journal.jsonl
./build/B13_spike 1000        # override sample count
```

## Expected first-run failure modes

The spike is designed so that failures name the gap:

- **"no matching overload for `ql::Foo(tracked::Bar<double>)`"** — add
  `Foo` to `ql_tracked_interop.hpp`. Cross-check against
  `kokkosMaths_dd.h` for the correct semantics.
- **`ql::Constants<Tracked<double>>::_something()` not found or wrong
  value** — Constants relies on the generic `Constants<T>` template in
  `kokkosMaths.h`, which returns `T(literal_double)`. If Tracked's
  constructor from double doesn't do the right thing (e.g., zero
  provenance vs. named provenance), we may need an explicit
  `Constants<Tracked<T>>` specialization.
- **`KOKKOS_LAMBDA` compile error involving tracked types** — should
  not happen since we run on the host, but if `BO()` internally uses a
  `KOKKOS_LAMBDA` for something we're not aware of, tracked's
  host-only ops will break the device check.
- **`Kokkos::View<Tracked<double>*>` allocation error** — Kokkos may
  require a trivially-constructible element type for some View ops.
  Fallback: use `std::vector<Tracked<double>>` for host storage.
- **NaN/inf storm in the journal** — a tracked overload silently
  returning wrong values. Diff the journal against a plain-double run
  of the same B13 subsection from `boxGPU_test.cc`.

## Not in scope for the spike

- Phase 0 schemas (`per_dependency.parquet`, range aggregation)
- Ranges YAML wiring
- Symbolic hints comparison
- Recall verification
- Full `kokkosMaths_tracked.h` replacement (Phase 0 investment)

All of the above land in Iteration 1 proper once the spike passes.
