# complex_log_micro — manual sanity-check micro-driver

One-off driver that instruments a representative `complex_log` (complex log with
branch-cut handling) kernel against `tracked::Complex<double>`. Calls into
`Kokkos::log` / `Kokkos::abs` are treated as opaque — the interior of those
functions is deliberately not characterized here. This is a wiring test, not a
real characterization.

This example is **not** built by the top-level `CMakeLists.txt`: the core
library is dependency-free, while this driver pulls in Kokkos. Build it on its
own from this directory.

## Build

```bash
cmake -B build -DKokkos_DIR=/path/to/kokkos/lib/cmake/Kokkos
cmake --build build -j
./build/complex_log_micro
```

If you don't already have a Kokkos install, `build_kokkos_serial.sh` builds a
minimal Serial-backend Kokkos to `$HOME/kokkos-install`.

## What you should see

Three sample inputs printed with their computed `complex_log` values, followed
by a `complex_log.jsonl` file containing one record per real op performed.
Eyeball with:

```bash
cat complex_log.jsonl | jq .
```

You should see `op="opaque"` records carrying `fn_name="Kokkos::log"` (the
black-box log calls) interleaved with normal `add` / `mul` / `neg` records
from the branch-cut handling around them.

## Notes for the reader

Two small adaptations were needed to compile a copied kernel against the
Tracked API — both are generic lessons, not specific to this kernel:

1. **`constexpr` removed from helper constants** — `Tracked<T>` holds a
   `std::set<std::string>` provenance field and is not a literal type, so
   `constexpr T _zero()` is ill-formed when `T = Tracked<double>`.

2. **Accessors return `Tracked<T>`, not raw `T`** — stripping to `.value()`
   yields a plain `double`, which then can't be compared against another
   `Tracked<double>` (`Tracked<T>` only defines comparisons against another
   `Tracked<T>`). Returning `z.imag()` / `z.real()` directly keeps both sides
   `Tracked<double>`.

## Result

**Works.** All three samples produce correct numerical output and the expected
journal structure: two `opaque` records per `Kokkos::log` call, plus visible
`neg` / `mul` / `add` records for the branch-cut arithmetic in Sample 3. The
opaque wrappers correctly hide the interior of `Kokkos::log` while still
capturing the branch-cut logic — confirming the micro-driver pattern works
end-to-end for complex-valued kernels.
