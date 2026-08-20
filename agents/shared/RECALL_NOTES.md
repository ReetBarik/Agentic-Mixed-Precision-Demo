# Recall verifier — notes & findings

Companion to `agents/shared/recall_verifier.py` — the end-to-end signal
usefulness check: for each validation fixture, recall = fraction of
`symbolic_hints[*].location` covered by
`sensitivity_profile.top_hotspots[*].location`, grouped by severity.

Pass criteria: **≥80% recall on `high`, ≥50% on `medium`, precision
unbounded** (false positives acceptable). Surfaced as a non-blocking `pass`
status; the script never exits non-zero on a threshold miss (only on
structural errors — missing files, malformed JSON).

## How attribution works (resolved via the hybrid — Track 1)

`location` originates from the `TRACKED_HERE` macro
(`::tracked::SourceLocation{__FILE__, __func__, __LINE__}`) attached to a
tracked op. The key question is *where `TRACKED_HERE` is lexically expanded*,
because that fixes `__func__` and `__FILE__`. As of Track 1, **every fixture
attributes ops to the kernel function**, via one of two mechanisms:

| Mechanism | `TRACKED_HERE` lives in | `__func__` is | Example profile location |
|---|---|---|---|
| **kernel-direct** (`cancellation`, `kahan`, `naive_variance`) | the kernel `.hpp`, in named `tracked::add/sub/...` calls | the kernel function | `…/cancellation.cpp:cancellation_check:10` |
| **call-site forwarding** (`log_sum_exp`, `lnrat`, `cln`) | the kernel `.hpp`/`.cpp`, passed *into* a shim call (`ql::kLog(z, TRACKED_HERE)`, `std::exp(a, TRACKED_HERE)`); the driver shim forwards it to the instrumented op | the kernel function | `…/lnrat_kernel.hpp:Lnrat:69` |

The earlier (pre-Track-1) state attributed the second group to the *shim*
function (`src/micro_driver.cpp:exp:28`, `:kLog:64`), which is what broke
recall — see "Before/after" below.

**Mechanics of call-site forwarding.** The shim takes a trailing
`tracked::SourceLocation loc = {}` and threads it into the `_at` op
(`tracked::opaque_at(fn, v, loc, …)`, `tracked::log(x, loc)`). The kernel
declares the dispatcher with the `= {}` default (so the default appears exactly
once) and passes `TRACKED_HERE` at the call site. The driver definitions omit
the default. See `agents/characterizer/prompts/driver_gen.txt` rule 9, and the
worked shims in `runs/{lnrat,cln,log_sum_exp}/src/micro_driver.cpp`.

Path relativization (commit `918738e`) makes the file part relative to the run
dir **when the file is inside the run dir**. Kernel files live under
`tests/agents/fixtures/kernels/*.{hpp,cpp}`, *outside* the run dir, so kernel
locations stay absolute (the parser's documented fallback). Matching is on the
function-name token, not the path, so this is fine.

## The matcher (v1)

Hint and hotspot locations use different formats:

```
hint    cancellation_check:2-4        (func:line-range)
hint    file:log_sum_exp_naive:11-13  (literal-"file":func:line-range)
hint    Lnrat:complex_overload        (kernel:logical-overload, no line)
hotspot /abs/.../lnrat_kernel.hpp:Lnrat:69         (abs-path:func:line)
hotspot /abs/.../cancellation.cpp:cancellation_check:10
```

(Hotspots now always carry the *kernel* function name — see "How attribution
works" — so rule (b) below matches the hint's function token even though the
formats otherwise differ.)

A hint matches a hotspot location if **either**:

- **(a) substring** — one string contains the other (after stripping a leading
  literal `file:` from the hint), or
- **(b) shared identifier token** — any C-identifier token of length ≥ 3 from
  the hint location (excluding the literal `file`) appears as a substring of the
  hotspot location.

Rule (b) is what lets kernel-direct hints match (shared function-name token,
e.g. `cancellation_check`). Precision is unbounded by spec, so this leniency is
acceptable. The matcher is intentionally simple and lives in `match_hint()`.

## Resolved (Track 1): call-site location forwarding — the hybrid option

The earlier writeup recorded a real attribution mismatch: hints named the
*kernel* (`log_sum_exp_naive`, `Lnrat:complex_overload`) but ops were attributed
to *driver-shim* functions (`exp`, `log`, `kLog`, `kAbs`), so they couldn't
match. Three fixes were on the table — (A) tighten the `symbolic_hints` schema
to `file:func:line`, (B) add a logical-name → location alias table the verifier
consults, or (C) the **hybrid**: emit `TRACKED_HERE` at the kernel call site so
`__func__` resolves to the kernel function.

**Track 1 implemented (C), the hybrid.** Option A could not fix it alone (the
hint emitter only sees kernel source and can't know ops land on shim functions),
and Option B pushes app-specific aliases into the verifier; the hybrid removes
the mismatch at the source instead of papering over it. Concretely:

- shims take a trailing `tracked::SourceLocation loc = {}` and forward it via the
  `_at` ops (`opaque_at`, `log(x, loc)`, …);
- kernels pass `TRACKED_HERE` at each shim call site
  (`ql::kLog(z, TRACKED_HERE)`, `std::exp(a, TRACKED_HERE)`);
- `driver_gen.txt` rule 9 instructs the generator to emit location-forwarding
  shims for future kernels.

### Before / after recall

Measured by `agents.shared.recall_verifier` on a full rebuild+rerun+reparse
(`scripts/regen_recall.sh`):

| fixture | before (shim-attributed) | after (kernel-attributed) |
|---|---|---|
| `cancellation` | high 1/1 (100%) PASS | high 1/1 (100%) PASS *(unchanged)* |
| `naive_variance` | high 1/1 (100%) PASS | high 1/1 (100%) PASS *(unchanged)* |
| `log_sum_exp` | high 0/1 (0%) **FAIL** | **high 1/1 (100%) PASS** |
| `lnrat` | medium 0/2, low 0/1 **FAIL** | **medium 2/2 (100%), low 1/1 (100%) PASS** |
| `cln` | no annotations | no annotations *(unchanged)* |
| `kahan` | no annotations | no annotations *(unchanged)* |
| **overall** | **FAIL** | **PASS** |

Post-fix locations: `…/log_sum_exp.cpp:log_sum_exp_naive:14`,
`…/lnrat_kernel.hpp:Lnrat:69`, `…/cln_kernel.hpp:cLn:55`. The two `lnrat` hints
(`Lnrat:complex_overload`, `Lnrat:real_overload`) both match on the shared
`Lnrat` token — only the complex overload runs in the micro-driver, but both
overloads share `__func__ == "Lnrat"`, so the token covers both (a benign
false-positive; precision is unbounded by spec).

### Known limitation — `log_sum_exp` `std::` interop

`lnrat`/`cln` retrofit cleanly: `ql::kLog`/`kAbs` are project dispatchers that
naturally accept a `SourceLocation`. `log_sum_exp` is uglier: the kernel calls
`std::exp(a, TRACKED_HERE)` / `std::log(…, TRACKED_HERE)`, which resolve to a
**2-argument overload injected into `namespace std`** by the driver. This works
(and is consistent with the already-UB std-injection the fixture relies on), but
no real-world kernel would write `std::exp(x, TRACKED_HERE)`. It is acceptable
for a fixture whose purpose is to exercise the `std::` interop path, but it is
**not** a pattern to recommend to users. For genuine user kernels that call
`std::` math directly and cannot be edited, call-site forwarding is not
available and ops remain shim-attributed — recall against kernel-named hints
would miss, and Option A or B would be the fallback. (Open for the human:
whether to keep `log_sum_exp` as-is or revert it to shim-attribution and
document it as an inherent `std::`-interop limitation.)

## Reproducing

```
# 1. rebuild + rerun each micro-driver so journals carry TRACKED_HERE locations
#    (needs cmake + Kokkos for lnrat/cln) — see scripts/regen_recall.sh
# 2. re-parse journals -> sensitivity_profile.json (relativizes paths)
python -m agents.shared.regen_profile runs/*
# 3. recall verifier -> stdout + runs/recall_summary.json
python -m agents.shared.recall_verifier runs
```
