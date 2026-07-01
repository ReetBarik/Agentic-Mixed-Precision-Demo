# Plan: Tracked v1.1 — complex number support

**Status:** v1.1 — adds complex-number support on top of v1 (historical design record).

## Goal

Add `tracked::Complex<T>` so that complex-valued numerical kernels (where most
of the interesting cancellation lives — complex `log`, complex `sqrt` near the
origin, etc.) can be instrumented with full per-real-op visibility in the
JSONL journal.

The wrapper stores **two `Tracked<T>` reals** so every underlying real op shows
up in the journal individually. We deliberately do **not** wrap `std::complex<T>`
or `Kokkos::complex<T>` — both would either be UB (`std::complex` outside
`float/double/long double`) or hide all the interesting real ops behind opaque
complex math (`Kokkos::complex` calls `std::sqrt`/`std::atan2`/`std::hypot` on
the underlying scalar, which buries condition information).

## Out of scope for v1.1

- GPU/device support (still v2 — same blockers as v1).
- `std::complex<Tracked<T>>` compatibility shim — explicitly not supported.
- Interop with `Kokkos::complex<T>` storage layout — `tracked::Complex<T>` is
  its own type with its own ABI.
- Trig functions beyond `sin`/`cos` (no `tan`, no hyperbolics) — added on demand.

## Prerequisites: extend `Tracked<T>` with three more real ops

Before the complex wrapper makes sense, `Tracked<T>` needs the underlying real
ops that complex math decomposes into. Add to `include/tracked/ops.hpp`:

### `sin(Tracked<T>)`

- Value: `std::sin(x.value())`.
- Local condition: `|x · cos(x) / sin(x)|` = `|x · cot(x)|`.
- Blows up near integer multiples of π (where `sin(x) → 0`).
- Cap cond at `1/u` (matching the existing `log` cap) when `|sin(x)| < u·|x|`
  to avoid overflow.

### `cos(Tracked<T>)`

- Value: `std::cos(x.value())`.
- Local condition: `|x · sin(x) / cos(x)|` = `|x · tan(x)|`.
- Blows up near `π/2 + kπ`.
- Same cap policy.

### `atan2(Tracked<T> y, Tracked<T> x)`

Two-input. The cond formula needs both inputs:

- Value: `std::atan2(y.value(), x.value())`.
- Partial derivatives: `∂atan2/∂y = x/(x²+y²)`, `∂atan2/∂x = -y/(x²+y²)`.
- Local condition (use sum-of-relative-contributions form):
  `cond = (|y · ∂/∂y| + |x · ∂/∂x|) / |atan2(y,x)|`
  `     = (|x·y| + |y·x|) / ((x²+y²) · |atan2(y,x)|)`
  `     = 2·|x·y| / ((x²+y²) · |atan2(y,x)|)`
- Pathological near `atan2 → 0` (i.e., on the positive real axis) and at the
  origin. Cap as above when `|atan2(y,x)| < u`.
- Provenance: union of `y.provenance_` and `x.provenance_`.
- Error propagation: `cond · (max(y.rel_err, x.rel_err) + u)`.

### Tests for the new real ops

Add to `tests/tracked/test_basic_ops.cpp` (or a new `test_trig.cpp`):

- `sin(0.1)` cond ≈ `|0.1 · cos(0.1)/sin(0.1)|` ≈ 1.0 (verify within 1%).
- `sin(π)` cond is huge (capped at `1/u`).
- `cos(0)` cond = 0 exactly (since `sin(0)=0`).
- `cos(π/2)` cond is huge (capped).
- `atan2(1, 1)` cond ≈ small (well-conditioned away from axes/origin).
- `atan2(1e-20, 1)` cond ≈ huge (near positive real axis where atan2 → 0).
- Provenance union: `atan2(track("y", 1), track("x", 1))` → prov = {x, y}.

## The complex wrapper

New file: `include/tracked/complex.hpp`.

### Type

```cpp
namespace tracked {

template <class T>
class Complex {
    Tracked<T> re_;
    Tracked<T> im_;

public:
    // Constructors
    Complex() = default;                                              // (0, 0)
    Complex(T re, T im = T(0));                                       // from raw
    Complex(Tracked<T> re, Tracked<T> im);                            // from tracked
    explicit Complex(Tracked<T> re);                                  // (re, 0)

    // Accessors
    const Tracked<T>& real() const { return re_; }
    const Tracked<T>& imag() const { return im_; }
    Tracked<T>&       real()       { return re_; }
    Tracked<T>&       imag()       { return im_; }

    // Compound assignment
    Complex& operator+=(const Complex& rhs);
    Complex& operator-=(const Complex& rhs);
    Complex& operator*=(const Complex& rhs);
    Complex& operator/=(const Complex& rhs);
    Complex& operator+=(const Tracked<T>& rhs);   // scalar overloads
    Complex& operator-=(const Tracked<T>& rhs);
    Complex& operator*=(const Tracked<T>& rhs);
    Complex& operator/=(const Tracked<T>& rhs);
};

// Factory: tag a fresh complex input with a source-variable id.
// Both re and im inherit the same provenance id; if you need separate ids
// for the real and imag parts, construct manually from two track() calls.
template <class T>
Complex<T> track(std::string id, T re, T im = T(0));

} // namespace tracked
```

### Arithmetic (free functions or hidden friends, your call)

Standard formulas, decomposed into Tracked real ops so the journal sees each:

- `(a, b) + (c, d) = (a+c, b+d)` — 2 adds.
- `(a, b) - (c, d) = (a-c, b-d)` — 2 subs.
- `(a, b) * (c, d) = (ac-bd, ad+bc)` — 4 muls, 1 sub, 1 add.
- `(a, b) / (c, d)`: use the **stable Smith algorithm** (Smith 1962) **not**
  the naive `((ac+bd) + i(bc-ad)) / (c²+d²)` because the naive denominator
  overflows for moderately large `|c|, |d|` (around `1e150` for `double`).
  Smith branches on `|c| vs |d|` to keep all intermediates bounded:

  ```
  if |c| >= |d|:
      r = d / c                       // |r| <= 1
      den = c + r*d                   // = c + d²/c, bounded
      re_out = (a + b*r) / den
      im_out = (b - a*r) / den
  else:
      r = c / d                       // |r| < 1
      den = d + r*c                   // = d + c²/d, bounded
      re_out = (a*r + b) / den
      im_out = (b*r - a) / den
  ```

  Both forms decompose into tracked real ops; the agent should be free to
  choose either via a future catalog entry. For v1.1, use Smith by default
  and add a `naive_div` test fixture (the naive formula written inline in the
  test, not exposed in the API) to demonstrate Tracked correctly flags the
  naive form as more unstable for large `|c|, |d|`.
- Unary `-`: negate both components.
- Scalar overloads: `Complex op Tracked<T>`, `Tracked<T> op Complex`,
  `Complex op T`, `T op Complex` — generated via the obvious lifts.

### Comparisons

- `operator==(Complex, Complex)` → plain `bool`, returns `re_==rhs.re_ && im_==rhs.im_` (comparing the underlying `Tracked<T>` values).
- `operator!=` → negation of above.
- **No `<` `>` `<=` `>=`** — complex is not ordered.

### Math functions (free functions in `tracked::`)

All decompose into real `Tracked<T>` ops so per-op cond is preserved:

```cpp
template <class T> Tracked<T> abs (const Complex<T>& z);   // sqrt(re² + im²) via hypot? No — use sqrt(norm)
template <class T> Tracked<T> norm(const Complex<T>& z);   // re² + im² (squared magnitude)
template <class T> Tracked<T> arg (const Complex<T>& z);   // atan2(im, re)
template <class T> Complex<T> conj(const Complex<T>& z);   // (re, -im)
template <class T> Complex<T> sqrt(const Complex<T>& z);   // see formula below
template <class T> Complex<T> log (const Complex<T>& z);   // log|z| + i·arg(z)
template <class T> Complex<T> exp (const Complex<T>& z);   // e^re · (cos(im) + i·sin(im))
```

**`abs(z)`** — `sqrt(norm(z))`. Do **not** use `hypot` (would be one opaque op,
defeats the purpose). The naive `sqrt(re² + im²)` can overflow for huge
components; this is a known trade-off and a future catalog candidate for a
Smith-style scaled magnitude. For v1.1, naive is fine — it lets Tracked report
the cond profile honestly.

**`sqrt(z)`** — use the cancellation-aware form:
```
let r = abs(z)
if re ≥ 0:
    w = sqrt((r + re) / 2)
    return Complex(w, im / (2·w))           // when w ≠ 0
else:
    w = sqrt((r - re) / 2)
    return Complex(|im| / (2·w), sign(im)·w)
```
This avoids the catastrophic-cancellation case that `sqrt(z) = sqrt(r)·(cos(arg/2) + i·sin(arg/2))` exhibits near the negative real axis. Document this in
a comment — it's instructive.

**`log(z)`** — `(log(abs(z)), arg(z))`. Branch cut is on the negative real
axis; cond blows up near `|z| = 1` (from the real `log` factor) and near the
negative real axis (from `atan2`'s branch behavior).

**`exp(z)`** — `(exp(re)·cos(im), exp(re)·sin(im))`. Decomposes into 1 `exp`,
1 `cos`, 1 `sin`, 2 muls. Cond profile inherits from `exp(re)` (which grows
with `|re|`) and the trig calls (which spike near their pathological inputs).

### Source location

Operators take no `SourceLocation`. Named free functions take an optional
`SourceLocation loc = {}` last parameter, same convention as `Tracked<T>`.
`TRACKED_HERE` macro already works since it's defined in `tracked.hpp`.

### Opaque barrier

`tracked::opaque("fn_name", Complex<T>{...})` overloads the existing `opaque`
for `Complex<T>`. Emits **two** real `op="opaque"` records (one for the real
component, one for the imaginary), both carrying `fn_name` as provenance and
as the source-variable id for the returned components. Consumers identify a
complex barrier by seeing two consecutive opaque records with the same
`fn_name`. Keeps the v1 schema unchanged.

## Log schema

**No changes to the v1 `LogRecord` schema.**

The complex wrapper is a structural convenience around two `Tracked<T>` reals
and emits **no records of its own**. Every complex op decomposes into real
ops on the underlying components, and those real ops emit normal records
(`add`, `sub`, `mul`, etc.) via the existing `Tracked<T>` machinery.

If the agent wants a complex-op-level view (e.g., "this complex multiply had
overall cond X"), it rolls up the journal by source location — all real
sub-ops produced by a single user-level complex op share the same `at:` field.
Location-based roll-up is something the characterizer needs to do anyway for
line-level reporting, so this comes for free.

The one exception is `opaque(Complex<T>)`: emit two `op="opaque"` records,
one for the real component and one for the imaginary, sharing the same
`fn_name`. Document that consumers should treat consecutive opaque records
with the same fn_name as a complex-valued barrier.

## Calibration suite (new tests)

New file: `tests/tracked/test_complex_basic.cpp`:
- Construction, accessors, arithmetic produce correct values.
- Provenance union across complex binary ops.
- Compound assignment.

New file: `tests/tracked/test_complex_sqrt.cpp`:
- `sqrt(z)` near the negative real axis: confirm the stable formula keeps
  `max_cond_seen_` low for, e.g., `z = -1 + 1e-10·i`.
- A naive formulation written inline in the test (using `arg/2` and `cos/sin`)
  should produce visibly higher `max_cond_seen_` for the same input. Assert
  the gap.

New file: `tests/tracked/test_complex_log.cpp`:
- `log(z)` for `|z| ≈ 1`: cond from the real `log` factor dominates.
- `log(z)` for `z` near negative real axis: `arg` branch-cut cond dominates.
- Assert both pathologies surface in `max_cond_seen_`.

New file: `tests/tracked/test_complex_div.cpp`:
- Smith division on `z1/z2` for `|z2|` moderate: well-conditioned.
- Naive `(ac+bd, bc-ad)/(c²+d²)` (written inline) on `z2` with `|z2| ≈ 1e150`:
  components overflow, journal shows the bad ops. Confirm Smith stays safe.

Plus the existing real-op tests get the three new functions covered in
`test_basic_ops.cpp` extensions or a new `test_trig.cpp` (your call).

Expect ~15–20 new tests. Total should land around 55–60 passing.

## Repository layout after v1.1

```
include/tracked/
  tracked.hpp          # unchanged interface; gains sin/cos/atan2 overloads
  journal.hpp          # unchanged
  ops.hpp              # gains sin, cos, atan2
  complex.hpp          # NEW
tests/tracked/
  test_basic_ops.cpp           # extended with sin/cos/atan2
  test_cancellation.cpp        # unchanged
  test_naive_variance.cpp      # unchanged
  test_log_sum_exp.cpp         # unchanged
  test_kahan_vs_naive.cpp      # unchanged
  test_complex_basic.cpp       # NEW
  test_complex_sqrt.cpp        # NEW
  test_complex_log.cpp         # NEW
  test_complex_div.cpp         # NEW
CMakeLists.txt                 # add new tests
```

## Milestones

1. Tag current main as `tracked-v1`.
2. Extend `Tracked<T>` with `sin`, `cos`, `atan2` + tests. All existing tests
   still pass.
3. Skeleton `include/tracked/complex.hpp`: type, constructors, accessors,
   arithmetic, comparisons. `test_complex_basic` passes.
4. Complex math functions: `abs`, `norm`, `arg`, `conj`, `exp`, `log`. Their
   tests pass.
5. Stable `sqrt` and stable (Smith) `div`. `test_complex_sqrt` and
   `test_complex_div` pass, including the demonstrations that the naive forms
   produce higher cond.
6. `opaque` overload for `Complex<T>` + a small test.
7. Update README on the branch noting v1.1 additions and tag as `tracked-v1.1`.

## Notes for the implementer

- The `arg` cond formula matters most for `log`'s imaginary part — get it right
  before building `log`.
- When decomposing a complex op into real ops, do **not** call `Tracked<T>`'s
  operator overloads via temporaries the compiler might elide; use named locals
  so journal records have stable source locations. Example:
  ```cpp
  // Bad: source location of multiplied result is ambiguous.
  return Complex<T>(a.real()*b.real() - a.imag()*b.imag(), ...);

  // Good:
  auto ac = a.real() * b.real();
  auto bd = a.imag() * b.imag();
  auto ad = a.real() * b.imag();
  auto bc = a.imag() * b.real();
  auto re = ac - bd;
  auto im = ad + bc;
  return Complex<T>(re, im);
  ```
- Don't be clever with hidden friends if it complicates ADL for a future
  interop header — free functions in `tracked::` are easier to
  forward-declare and overload.
- All complex ops should accept the optional `SourceLocation loc = {}` final
  parameter for `TRACKED_HERE` use, even if the operator versions don't.
