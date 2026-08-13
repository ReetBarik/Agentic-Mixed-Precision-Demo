# Vendored extended-precision headers

Header-only, Kokkos-native extended-precision arithmetic, portable across all Kokkos
backends. Three backends, six files:

| backend | words | digits | range | files |
|---|---|---|---|---|
| DD (double-double) | 2 × FP64 | ~30–31 | FP64 (~1.8e308) | `dd_math.hpp`, `dd_complex.hpp` |
| FF (float-float) | 2 × FP32 | ~14 | **FP32 (~3.4e38)** | `ff_math.hpp`, `ff_complex.hpp` |
| QF (quad-float) | 4 × FP32 | ~28.9 | **FP32 (~3.4e38)** | `qf_math.hpp`, `qf_complex.hpp` |

- `*_math.hpp` — the **real** type (`Kokkos::Experimental::DoubleDouble` / `FloatFloat` /
  `QuadFloat`) plus arithmetic, `sqrt`, `log`, `exp`, trig and special functions.
- `*_complex.hpp` — the **complex** type (`…Complex`, with `re`/`im` members); each
  `#include`s its own `*_math.hpp`.

All are `#pragma once`, every function `KOKKOS_INLINE_FUNCTION`. They hard-depend on
`<Kokkos_Core.hpp>` (Kokkos must be on the include path); they do **not** include any
qcdloop headers.

> **QF widens precision but not range.** Stacking FP32 words grows the significand and
> leaves the exponent alone, so `qf` resolves nearly twice as many digits as `double`
> while overflowing ~270 orders of magnitude sooner. `float`, `ff` and `qf` share one
> range ceiling; Strategy keys its range guard on that family
> (`agents/strategy/models.FP32_FAMILY`), not on `float`.

Also here: `kokkosMaths_ff.h` and `kokkosMaths_qf.h` — the `ql::`-surface **enrichment**
headers that layer qcdloop's `ql::ffun` / `ql::qfun` namespaces, `Constants<T>` coefficient
tables and `kAbs`/`kLog`/`kSqrt` dispatch on top of the vendored primitives. They exist
because `Kokkos::complex` static_asserts on a non-built-in element type, so the FP32
backends need their own complex container (`FloatFloatComplex` / `QuadFloatComplex`)
rather than `Kokkos::complex<FloatFloat>` — the original STOP #EEE. The DD equivalent
lives in the snapshot as `runs/qcdloop_headers_full/kokkosMaths_dd.h`.

## Provenance

**`UPSTREAM.sha` is authoritative** — source repo, per-file shas, the full local-patch
inventory (the vendored files are *not* byte-identical to upstream), and the measured
per-file delta. Read it before any refresh; the local patches must be re-applied on top of
a fresh copy or the tree will not compile, and dropping the hypot `abs` would silently move
the DD oracle.

### Refresh

```sh
cd $(git rev-parse --show-toplevel)
SRC=/path/to/kokkos-extended-precision-demo   # ReetBarik/kokkos-extended-precision-demo
for f in dd_math.hpp dd_complex.hpp ff_math.hpp ff_complex.hpp qf_math.hpp qf_complex.hpp; do
  cp "$SRC/third_party/include/$f" third_party/include/$f
done
grep -n 'LOCAL PATCH' third_party/include/*.hpp   # re-apply every block UPSTREAM.sha lists
```

Then update the shas and the `local_delta_lines` counts in `UPSTREAM.sha`.

## Who uses these

- The **precision-flip TU emitter** (`agents/patcher/tu_emit.py`) builds per-group flip
  translation units against these, selecting a backend via `USE_DD_COMPLEX` /
  `USE_FF_COMPLEX` / `USE_QF_COMPLEX`.
- The **Validator's DD ground-truth oracle** builds against `qcdloop@ddfun_enabled`, whose
  own copies of the dd/ff primitives are stripped by
  `agents/validator/runner.stage_dd_headers()` so the quoted includes fall through to
  *this* directory — the oracle tracks these files, not the fork's frozen ones.
- The regional `dd_integrator` / `ff_integrator` LLM shims reference these type names.

## Licensing

DD and FF descend from DDFUN (D. H. Bailey) and carry `LicenseRef-DHB-License`. QF is a
port of QD 2.3.24 (Hida–Li–Bailey, LBNL) and carries `LicenseRef-LBNL-BSD-License` — a
**different** lineage and a different license. The `LICENSES/` texts those headers
reference are not vendored here; that bookkeeping is deferred (see `UPSTREAM.sha`).
