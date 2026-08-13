# Vendored double-double (DD) headers

Header-only, Kokkos-native double-double arithmetic (~30–31 decimal digits, two
`double` per number), portable across all Kokkos backends.

- `dd_math.hpp`  — DD **real** type `Kokkos::Experimental::DoubleDouble` + arithmetic / `sqrt`,
  `log`, `exp`, trig, special functions. Ported from DDFUN (D.H. Bailey).
- `dd_complex.hpp` — DD **complex** type `Kokkos::Experimental::DoubleDoubleComplex` (`re`,`im` as
  `DoubleDouble`); `#include "dd_math.hpp"`.

Both are `#pragma once`, every function `KOKKOS_INLINE_FUNCTION`. They hard-depend
on `<Kokkos_Core.hpp>` (Kokkos must be on the include path at compile time); they
do **not** include any qcdloop headers.

## Provenance

Vendored verbatim from **`ReetBarik/kokkos-extended-precision-demo@ddfunKokkos`**,
commit `8e34425f8d9ca220d6187ea4e5fd1c9cd508c878` (2026-07), source path
`third_party/include/{dd_math.hpp, dd_complex.hpp}`.

A plain vendored copy (not a git subtree): the upstream repo is a whole Kokkos demo
project, but only these two leaf headers are wanted here, and `git subtree`
operates on whole-tree prefixes — it cannot target two files at this destination.
This mirrors the *intent* of `third_party/tracked/` (vendored in-tree, not a
submodule) and the refresh pattern of `runs/qcdloop_headers_full/README.md`.

### Refresh

```sh
cd $(git rev-parse --show-toplevel)
SRC=/path/to/kokkos-extended-precision-demo   # ReetBarik/kokkos-extended-precision-demo
for f in dd_math.hpp dd_complex.hpp; do
  git -C "$SRC" show ddfunKokkos:third_party/include/$f > third_party/include/$f
done
```

Then update the commit sha above.

## Who uses these

The **future** real `agents/dd_integrator` (LLM-driven DD shim generation) will
target these canonical headers. The **current** `dd_integrator` *stub* and the
Validator's DD ground-truth oracle instead build against qcdloop's own adapted
fork (`ql::ddfun`, in `qcdloop@ddfun_enabled:src/qcdloop/`), which differs from
these (namespace `Kokkos::Experimental`→`ql::ddfun`, `Kokkos::bit_cast` portability, extra
operators). See `agents/dd_integrator/agent.py`.
