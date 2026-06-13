# Plan: Characterizer slice — remaining work

**Branch:** `langgraph-agents` (continuing).
**Sibling dep:** `kokkos-extended-precision-demo` branch `tracked` at
commit `8cae2c0` or later (Tracked v1.1 with the opaque fix).
**Last commit on this branch:** `6d357bb` (TRACKED_HERE in fixtures + README walkthrough).

This is the followup to `agents/characterizer/PLAN.md` after the v1 vertical
slice landed.  Four calibration fixtures (cancellation, naive_variance,
log_sum_exp, kahan) already run end-to-end and produce sensitivity profiles
that flag the expected hotspots.  Items below are what's left to get the
slice to "done enough to start strategy agent work."

## Priority order

1. **`cLn` end-to-end** — first complex/Kokkos test.  Required to prove the
   characterizer handles real-world kernels.
2. **A few more QCDLoop kernels** — `Lnrat`, `fndd`, maybe `ddilog`.  Cheap
   wins once `cLn` works.
3. **Test suite expansion** — `test_build_run_stub.py`,
   `test_characterizer_e2e.py`.  Deferred from v1 because they need cmake
   + LLM infra.
4. **Snapshot tests for prompts** — `test_driver_gen.py`,
   `test_symbolic_overlay.py`.  Lowest priority; nice to have for prompt
   regression but slow to write and maintain.
Items 1 and 2 are the real work; 3 and 4 are polish.

**Note on `runs/`:** committed intentionally so the user (developing on a
remote cluster) can share run artifacts with assistants who don't have
SSH access.  Do not gitignore.  Add new runs as they're produced.

---

## 1. `cLn` end-to-end

### 1a. Kokkos availability check

Confirm `$HOME/kokkos-install` exists and has
`lib64/cmake/Kokkos/KokkosConfig.cmake` (or `lib/...`).  If not, build it
by running this script from inside the Tracked submodule:

```bash
bash third_party/tracked/examples/cln_micro/build_kokkos_serial.sh
```

Captures Kokkos 5.1.0, Serial backend only, C++20.  ~5 min build.

### 1b. CLI flag pass-through (already exists, just verify)

`agents/cli.py` already accepts `--kokkos-root`.  `PipelineConfig` already
has `kokkos_root: Path | None`.  `agents/build_run/agent.py` already wires
it into `-DKokkos_DIR=<path>` when `framework == "kokkos-serial"`.  Should
be no code changes here — just verify the path actually flows through by
inspecting the rendered `CMakeLists.txt` after a run.

### 1c. New fixture: `cln_kernel.hpp`

Path: `tests/agents/fixtures/kernels/cln_kernel.hpp`.

Content: the `cLn` function body lifted from
`src/kokkosUtils.h`, **plus** a minimal self-contained `ql::` surface so
the kernel header has no external dependencies beyond `<Kokkos_Core.hpp>`.

Use the manual `cln_micro.cpp` driver from the Tracked repo
(`third_party/tracked/examples/cln_micro/cln_micro.cpp`) as the reference
for which `ql::` helpers are needed and how they're shaped:

- `ql::Constants<T>` — `_zero()`, `_pi()`.  NOT `constexpr` (Tracked is
  not a literal type).
- `ql::Imag(z)`, `ql::Real(z)` — must return `Tracked<T>` not `T` so
  comparisons with `Constants<T>::_zero()` typecheck.
- `ql::Sign(scalar)` — returns int.
- Stub declarations for `ql::kAbs` and `ql::kLog` — left as declarations
  only.  The characterizer generates the wrappers (opaque or shim).

The fixture must compile **standalone** when included from a driver, so
test it by hand-pasting into the manual `cln_micro.cpp` first to confirm.

Key design decision: do **not** pre-bake opaque wrappers into the
fixture.  The characterizer's whole job is to make the interop-vs-opaque
decision.  Pre-baking would short-circuit the test.

### 1d. New ranges file: `cln_kernel.yaml`

Path: `tests/agents/fixtures/input_ranges/cln_kernel.yaml`.

`cLn(z, isig)` takes a complex `z` and a real `isig`.  Complex ranges
need two scalar entries — `z_re` and `z_im` — since the current YAML
schema (verify in `agents/cli.py`'s parser) only supports flat
`{name: [min, max]}` pairs.

Suggested ranges to hit the interesting cases:

```yaml
ranges:
  z_re:  [-2.0,  2.0]
  z_im:  [-1.0,  1.0]
  isig:  [-1.0,  1.0]
```

This covers ordinary complex inputs, near-branch-cut (small `z_im`),
and negative-real-axis (when sampling happens to land near `z_im=0,
z_re<0`).

If the existing YAML parser doesn't support complex ranges natively,
the simplest fix is to add a `complex_ranges:` section in the YAML and
update `agents/cli.py` to merge them into `input_ranges` as a flat dict
with `_re`/`_im` suffixes.  ~10 LoC change.

### 1e. Spec builder updates (likely needed)

In `agents/characterizer/agent.py::_spec_build`:

- `is_complex` detection already triggers on `\bImag\b|\bReal\b|\bcomplex\b`.
  Should fire for `cln_kernel.hpp`.  ✓
- Template instantiation maps `T*` params to `tracked::Complex<double>`
  when complex.  Need to verify this picks the right type per parameter
  (e.g., `TScale` should stay `tracked::Tracked<double>` even when
  `TOutput` is complex).  May need to special-case: scan how each
  template parameter is used — if used as the type of a `Imag()`/`Real()`
  argument or stored in a complex-shaped variable, complex; otherwise
  scalar.
- Framework detection: `\bKokkos\b` → `"kokkos-serial"`.  ✓ if the
  kernel header includes `Kokkos_Core.hpp`.

### 1f. Prompt revision

Open `agents/characterizer/prompts/driver_gen.txt`.  Add these clarifications:

1. **Complex input construction.**  Add an example near the existing
   `tracked::track("x", x_sample)` line:

   ```
   For complex parameters, use the two-argument track factory:
       auto z = tracked::track("z", z_re_sample, z_im_sample);
   The two component arguments share the provenance id "z".
   ```

2. **Kokkos initialization.**  Add a line in the strategy rules:

   ```
   When framework is "kokkos-serial", wrap the main() body in
   Kokkos::initialize() ... Kokkos::finalize().  Place tracked::track
   calls and the kernel invocation between them.
   ```

3. **`KOKKOS_INLINE_FUNCTION` macro.**  Note that source headers often
   carry this annotation.  The driver should `#define KOKKOS_INLINE_FUNCTION
   inline` before including the kernel if needed, to avoid CUDA/HIP
   attributes leaking into host-only Tracked code.  (Alternative:
   ensure the host-only Kokkos install defines it correctly.  Test which
   applies before deciding.)

4. **Optional: inline the manual `cln_micro.cpp` as a worked example.**
   The prompt is currently abstract.  A concrete few-shot showing
   "here's a Kokkos+complex kernel and here's the right driver for it"
   would dramatically reduce first-try failures.  But adds ~150 lines
   to the prompt.  Try without first; add if the LLM struggles.

### 1g. Run, debug, iterate

```bash
python -m agents.cli characterize \
  --kernel tests/agents/fixtures/kernels/cln_kernel.hpp \
  --kernel-name cLn \
  --ranges-yaml tests/agents/fixtures/input_ranges/cln_kernel.yaml \
  --samples 256 \
  --kokkos-root $HOME/kokkos-install \
  --out runs/cln
```

Expected first-failure modes (in rough order of likelihood):

- **Include path for Kokkos.**  `cmake` finds `Kokkos::kokkos` but the
  compiler can't find `<Kokkos_Core.hpp>` — usually means the
  `Kokkos::kokkos` target's interface includes aren't being picked up.
  Fix: ensure `target_link_libraries(... PUBLIC Kokkos::kokkos)` rather
  than `PRIVATE`, or add explicit `target_include_directories`.
- **Complex track factory signature.**  LLM writes
  `tracked::track("z", {z_re, z_im})` (uniform init) instead of
  `tracked::track("z", z_re, z_im)`.  Compile error.  Refine the prompt
  example.
- **Opaque wrappers missing.**  LLM thinks `Kokkos::log` will template
  on `tracked::Complex` and skips the wrapper.  Hard compile error at
  the call site.  Prompt nudge: "Kokkos::log on complex types is NOT
  overloaded for tracked::Complex — always wrap with either an interop
  shim that delegates to tracked::log(Complex) OR an opaque wrap."
- **`ql::Imag/Real` return type confusion.**  Already documented in the
  fixture header; should be fine.
- **C++ standard mismatch.**  Tracked builds C++17.  Kokkos 5.1 prefers
  C++20.  Driver template uses C++17 by default in `PipelineConfig`.
  Bump to C++20 for the cln run if Kokkos errors with "requires C++20".

Each iteration: read the build log in `runs/cln/build.log`, identify the
issue, adjust either the prompt or the fixture, re-run.  Budget 3 cycles
max before stepping back to reconsider.

### 1h. Validation

After a successful run, verify:

- `runs/cln/journal.jsonl` exists, contains records.
- At least one record has `op == "opaque"` and `prov` containing both
  `"Kokkos::log"` and one of `"z_re"`/`"z_im"`.  This proves the v1.1
  opaque provenance fix works end-to-end through the characterizer.
- `runs/cln/sensitivity_profile.json` lists `Kokkos::log` in
  `per_variable` (the opaque op promotes its fn_name to a "variable"
  in the rollup, which is fine — agents downstream will see it as a
  named attribution target).
- `runs/cln/interop_decisions.json` has entries for `Kokkos::log` and
  `Kokkos::abs` with non-empty `justification` strings.
- `opaque_coverage` in the profile is moderate (not 100%); if it's
  100%, the LLM picked opaque for everything and we'd want to revisit.

---

## 2. More QCDLoop kernels

Once `cLn` works, the path to additional kernels in
`src/kokkosUtils.h` should be repetitive.  Suggested order (easiest →
hardest):

1. **`Lnrat`** — wraps `kLog(x/y)` plus a branch-cut term.  Same shape
   as `cLn`, slightly more complex.
2. **`fndd`** — Denner-Dittmaier formula evaluation.  Uses `kAbs`,
   `kPow`, `iszero`, `cLn`.  Multiple branches.  Tests whether the
   characterizer handles helper-function calls.
3. **`Sign`, `Imag`, `Real`, `iszero`** — trivial; mostly exists already
   in the cln fixture's inlined surface.  Add as standalone fixtures
   only if useful for unit-level characterizer testing.
4. **`ddilog`** — Chebyshev series evaluation of the dilogarithm.  ~50
   ops in a loop.  The first kernel with non-trivial control flow.
   May expose new prompt or spec-builder issues.

Each one gets a `.hpp` fixture and a `.yaml` ranges file in the same
pattern as `cln_kernel`.  Skip the `ql::` surface re-export for these —
have them include `cln_kernel.hpp` or extract the shared `ql::` surface
into `tests/agents/fixtures/kernels/_ql_surface.hpp` and include that
from each fixture.

Functions to **defer past today's session:**

- Any taking `Kokkos::Array<TOutput, 2>` (xeta, xspence, solveabc, kfn,
  etc.) — array-typed parameters need `_spec_build` extension.
- Anything that calls other QCDLoop functions in a deep chain
  (`R3int`, `Rint`, `R`) — characterizing one of these implicitly
  characterizes all transitive callees.  Manageable but burns
  iteration budget.
- `eta`/`etatilde` family — integer-valued return type, less interesting
  for numerical sensitivity.

---

## 3. Test suite expansion

### `tests/agents/test_build_run_stub.py`

Pure subprocess test.  No LLM.  Requires cmake + Tracked submodule
present.

```python
def test_build_run_compiles_minimal_driver(tmp_path):
    driver = """
        #include <tracked/tracked.hpp>
        #include <tracked/journal.hpp>
        int main() {
            auto a = tracked::track("a", 1.0);
            auto b = tracked::track("b", 2.0);
            auto c = a + b;
            tracked::journal::flush("journal.jsonl");
            return 0;
        }
    """
    cfg = PipelineConfig(tracked_root=TRACKED_ROOT, cxx_standard=17, ...)
    result = build_and_run(driver, framework="plain-cpp", cfg=cfg,
                            work_dir=tmp_path)
    assert result.returncode == 0
    assert result.journal_path is not None
    assert result.journal_path.exists()
    journal_text = result.journal_path.read_text()
    assert '"op":"add"' in journal_text
```

Add a `kokkos-serial` variant marked `@pytest.mark.kokkos` and skipped
when `KOKKOS_ROOT` env var is unset.

### `tests/agents/test_characterizer_e2e.py`

LLM-required, marked `@pytest.mark.llm`.  One test per calibration
fixture asserting the predicted hotspot appears:

```python
def test_cancellation_flags_sub(tmp_path):
    profile = run_characterizer(
        kernel="tests/agents/fixtures/kernels/cancellation.cpp",
        kernel_name="cancellation_check",
        ranges_yaml="tests/agents/fixtures/input_ranges/cancellation.yaml",
        out=tmp_path,
    )
    flagged_subs = [r for r in profile.per_op if r.op == "sub" and r.flagged]
    assert len(flagged_subs) >= 1
    assert flagged_subs[0].max_cond > 1e8
```

Plus a `test_cln_kernel` marked `@pytest.mark.llm @pytest.mark.kokkos`
that asserts the opaque-with-tracked-provenance condition from §1h.

Skip CI integration for these — they're slow (LLM round-trips) and
require external services.  Local-only with explicit marker selection.

---

## 4. Snapshot tests for prompts (lowest priority)

`tests/agents/test_driver_gen.py` using `syrupy` (or hand-rolled
fixture diffing).  For each calibration kernel, run `driver_gen`,
snapshot the `driver_source`, fail on diff unless `--snapshot-update`
is passed.

Catches accidental prompt regressions but requires manual re-blessing
after every intentional prompt change.  Defer until the prompt has
stabilized.

Similar pattern for `test_symbolic_overlay.py`.

---

## Completion criteria

Done with this round when:

- `runs/cln/sensitivity_profile.json` exists and passes the §1h checks.
- 2–3 additional QCDLoop kernels (Lnrat, fndd at minimum) have
  successful runs in `runs/`.
- `test_build_run_stub.py` and `test_characterizer_e2e.py` exist and
  pass locally with `pytest -m "not slow"` and `pytest -m llm`
  respectively.
- README's "Characterizer slice (v2)" section mentions the new cln/QCDLoop
  examples.

After this round, the next step is the **strategy agent** — turning
sensitivity profiles into a queue of catalog optimizations.  That's a
separate plan.
