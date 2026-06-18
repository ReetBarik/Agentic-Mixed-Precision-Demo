# Plan: Driver compile-retry loop

**Branch:** `langgraph-agents`
**Status:** ready to implement
**Source:** §0 of `agents/characterizer/NEXT.md`

## Goal

`driver_gen.generate()` currently makes one LLM call. If the generated driver
fails to compile, the pipeline dies and the user fixes things by hand. Build a
multi-turn retry loop so the LLM sees its own previous attempt plus the build
error and revises.

The README diagram has always shown this:

```
H[Compile]
H -- Failed --> I[Feed error back to LLM]
I --> G
```

This is the implementation.

## Scope

- Retry only on **configure** or **build** phase failures.
- Do NOT retry on **run** phase failures (driver compiled but binary exited
  non-zero). Most runtime failures are kernel-side or fixture-side; LLM has
  no actionable lever since it's told not to touch the kernel. Surface as
  today. Revisit later if a clear class of LLM-fixable runtime errors emerges.
- Conversation history feeds back the assistant's prior `tool_use` block
  followed by a `tool_result` with the build error excerpt.
- Driver scaffolding only; the LLM is reminded each retry not to change the
  kernel.

## Config changes (`agents/config.py`)

```python
@dataclass
class PipelineConfig:
    ...
    max_driver_attempts: int = 5      # total LLM attempts incl. first
    retry_stderr_chars: int = 3000    # truncation budget for fed-back stderr
```

Naming: "attempts" not "retries" — `max_driver_attempts=5` means 1 initial
+ up to 4 retries. Honest about what the budget covers.

## CLI flag (`agents/cli.py`)

Add `--max-driver-attempts N` (default 5) wired into `PipelineConfig`.

## `RunResult.phase` (`agents/build_run/agent.py`)

Add an explicit phase label so the retry discriminator is unambiguous:

```python
@dataclass
class RunResult:
    returncode: int
    stdout: str
    stderr: str
    journal_path: Path | None
    work_dir: Path
    phase: Literal["configure", "build", "run", "ok"]   # NEW
```

`build_and_run` already returns at three distinct points — tag each:

- Configure failure → `phase="configure"`
- Build failure → `phase="build"`
- Run completed (regardless of returncode) → `phase="run"` on non-zero exit,
  `phase="ok"` on zero exit + journal present.

Retry discriminator: `result.phase in {"configure", "build"}`.

## `driver_gen.generate()` signature change

```python
def generate(
    spec: InstrumentationSpec,
    cfg: PipelineConfig,
    messages: list[dict] | None = None,   # full conversation; None = first turn
) -> tuple[DriverGenOutput, list[dict]]:
    """Return parsed output AND the conversation including the assistant turn.

    The caller doesn't have to reach into the raw response to extend the
    history on retry.
    """
```

First-call path (`messages is None`): build the single user message exactly
as today. Returned `messages` is `[user_msg, assistant_response]`.

Retry path (`messages` provided): pass straight to
`client.messages.create(messages=messages, ...)`. Returned `messages` is
the input extended with the new assistant response.

Only caller is `characterizer/agent.py` — no external API churn.

## Conversation extension on failure

The assistant turn ends with a `tool_use` block (the `emit_driver` call).
Anthropic API requires the next user turn to start with a `tool_result` for
that tool_use id. So the extension is:

```python
[
    *prev_messages,    # includes the assistant turn that ended in tool_use
    {
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": <id of prior emit_driver call>,
            "content": (
                f"Driver failed to {phase} (attempt {n}/{N}).\n\n"
                f"Error output:\n```\n{stderr[:cfg.retry_stderr_chars]}\n```\n\n"
                "Revise the driver_source to fix this error. Do NOT modify "
                "kernel logic — only fix the driver scaffolding (includes, "
                "shim/opaque wrappers, type instantiations, init/finalize, "
                "sampling, journal flush). Re-emit via emit_driver."
            ),
        }],
    },
]
```

Helper lives in `driver_gen.py`:

```python
def extend_with_build_error(
    messages: list[dict],
    run_result: RunResult,
    cfg: PipelineConfig,
    attempt: int,
    max_attempts: int,
) -> list[dict]:
    ...
```

Extracts the `tool_use_id` from the last assistant turn in `messages`.

## Loop in `agents/characterizer/agent.py`

```python
attempts: list[tuple[DriverGenOutput, RunResult]] = []
messages = None
driver_result = None
run_result = None

for attempt in range(1, cfg.max_driver_attempts + 1):
    driver_result, messages = driver_gen.generate(spec, cfg, messages=messages)

    run_result = build_run_agent.build_and_run(
        driver_source=driver_result.driver_source,
        framework=spec.framework,
        cfg=cfg,
        work_dir=cfg.out_dir,
    )

    _archive_attempt(cfg.out_dir, attempt, driver_result, run_result)
    attempts.append((driver_result, run_result))

    if run_result.phase == "ok":
        break

    if run_result.phase in {"configure", "build"} and attempt < cfg.max_driver_attempts:
        messages = driver_gen.extend_with_build_error(
            messages, run_result, cfg, attempt, cfg.max_driver_attempts,
        )
        continue

    # Runtime failure, or out of attempts — fall through to existing error path
    break

_write_retry_log(cfg.out_dir, attempts)

# Existing returncode-check branch handles final failure the same as today.
```

## Build directory hygiene

Before each retry, wipe `work_dir/build/` to avoid stale `CMakeCache.txt`
poisoning the next configure. Cheap (cmake is incremental on a fresh dir
for small projects) and removes a real footgun.

```python
# Inside build_and_run, when called on an existing work_dir
import shutil
if build_dir.exists():
    shutil.rmtree(build_dir)
build_dir.mkdir()
```

Add a `clean_build: bool = True` parameter to `build_and_run` so the retry
loop can ask for it explicitly. Default True is fine for v1.

## Per-attempt artifacts

Each attempt currently overwrites `work_dir/src/micro_driver.cpp` and
`work_dir/build/`. Archive each attempt for debugging:

```
runs/<kernel>/
  src/micro_driver.cpp           # final (winning or last-tried) attempt
  build/                         # final attempt's build
  journal.jsonl                  # winning attempt's output
  attempts/
    01_driver.cpp
    01_phase                     # "build" | "configure" | "run" | "ok"
    01_stderr.log                # exactly what was fed back to LLM
    01_returncode                # numeric
    02_driver.cpp
    02_phase
    02_stderr.log
    02_returncode
    ...
  retry_log.json                 # at-a-glance summary
```

`retry_log.json` shape:

```json
{
  "max_attempts": 5,
  "attempts_used": 3,
  "outcome": "ok",
  "attempts": [
    {"attempt": 1, "phase": "build", "returncode": 1,
     "stderr_excerpt": "first 500 chars...", "notes": "..."},
    {"attempt": 2, "phase": "build", "returncode": 1,
     "stderr_excerpt": "...", "notes": "..."},
    {"attempt": 3, "phase": "ok", "returncode": 0,
     "stderr_excerpt": "", "notes": "..."}
  ]
}
```

Headline artifact for the user: "how many tries, what was wrong each time."

## Prompt additions (`agents/characterizer/prompts/driver_gen.txt`)

### 1. Multi-turn retry contract (one line in the header)

Add to the top of the prompt, near "## What you receive":

```
If you receive a follow-up message reporting a build error from a previous
attempt, revise the driver_source to fix that specific error.  Do NOT modify
kernel logic — only fix the driver scaffolding.  Re-emit via emit_driver.
```

### 2. Output-by-reference / void-returning kernels (new section)

Many kernels return `void` and write results into parameters passed by
reference or pointer. The current prompt assumes value-returning kernels and
gives no guidance for the void/by-ref case. Add a new section under
"Strategy rules":

```
## Output-by-reference kernels

Some kernels return `void` and write results into parameters passed by
reference or pointer.  Detect this by checking the return type in
`kernel_signature` and by scanning `parameter_types` and `parameter_roles`
for parameters marked `"output"` or `"inout"`.

For these kernels:

1. The spec lists each parameter's role (input | output | inout) in
   `parameter_roles`, parallel to `parameter_types`.  Use that, not your own
   inference.
2. For each output parameter, declare a default-constructed Tracked instance
   in the driver and pass it by reference.  Do NOT wrap output parameters
   with `tracked::track()` — they have no input range and no initial value
   to track:

   ```cpp
   tracked::Tracked<double> result;          // output: no track() call
   auto x = tracked::track("x", x_sample);   // input
   kernel(x, result);                        // void return, writes to result
   ```

3. Print `result.value()` on the first iteration just like a returned value,
   so the per-iteration log line behaves identically to value-returning
   kernels.

4. Provenance flows through the output parameter automatically — the journal
   records emitted inside the kernel carry the input provenance.

5. For **complex output parameters**, default-construct
   `tracked::Complex<double>` the same way:

   ```cpp
   tracked::Complex<double> z_out;
   kernel(z_in, z_out);
   ```

6. For output **arrays** (e.g. `T result[2]`, `Kokkos::Array<T,2>&`),
   default-construct the array of Tracked elements; do not call `track()`
   on the elements.

7. For **inout parameters** (role == "inout"): wrap with
   `tracked::track(name, sample)` like an input — the kernel will update
   in place.

8. **const-qualified reference/pointer parameters** (`const T&`, `const T*`)
   are always inputs regardless of any other heuristic.
```

## `_spec_build` change in `agents/characterizer/agent.py`

Classify each parameter's role and surface it to the LLM.

### Rule

```python
def _classify_role(type_str: str, name: str, input_ranges: dict) -> str:
    has_ref_or_ptr = "&" in type_str or "*" in type_str
    is_const = "const " in type_str or type_str.startswith("const")

    if not has_ref_or_ptr:
        return "input"                      # value type
    if is_const:
        return "input"                      # const ref/ptr always input
    if name in input_ranges:
        return "inout"                      # mutable + has range
    return "output"                         # mutable, no range
```

Const-correctness is the only reliable signal short of parsing the kernel
body. A real output cannot be `const` in valid C++, so collapsing all const
ref/ptr params to `"input"` is safe.

### Spec changes

Add to `InstrumentationSpec`:

```python
parameter_roles: list[Literal["input", "output", "inout"]]
```

Parallel array to `parameter_types`. Populate in `_spec_build`. Include in
the JSON spec block surfaced to the LLM (`_build_user_message` in
`driver_gen.py`).

## Testing

`tests/agents/test_driver_retry_loop.py` (new):

- **`test_loop_breaks_on_success_first_try`**: mock `driver_gen.generate` to
  return a good driver; mock `build_and_run` to return `phase="ok"`. Assert
  exactly one generate call, one build call, one attempt archived.
- **`test_loop_retries_on_build_failure_then_succeeds`**: mock generate to
  return bad-then-good drivers; mock build_and_run to return
  `phase="build"` then `phase="ok"`. Assert 2 attempts, second message has
  `tool_result` with stderr excerpt, retry_log.json shows outcome="ok".
- **`test_loop_exhausts_attempts`**: mock build_and_run to always return
  `phase="build"`. Assert exactly `max_driver_attempts` calls, final error
  surfaces as before.
- **`test_loop_does_not_retry_run_failure`**: mock generate good, mock
  build_and_run to return `phase="run"`, `returncode=1`. Assert exactly one
  generate call, error surfaces.
- **`test_extend_with_build_error_extracts_tool_use_id`**: unit test on the
  helper — given a messages list ending in an assistant turn containing a
  `tool_use` block, the extended messages start the next user turn with a
  `tool_result` referencing the correct id.

Optional: `test_void_return_classification` — `_classify_role` returns
expected roles for `const T&`, `T&` in ranges, `T&` not in ranges, `T*`,
`T` by value.

## Implementation order

1. `RunResult.phase` field + `build_and_run` tagging (mechanical, no logic
   change).
2. `clean_build` flag in `build_and_run` (so the loop can wipe between
   attempts).
3. `driver_gen.generate()` signature change + `extend_with_build_error`
   helper.
4. Loop in `characterizer/agent.py`.
5. Per-attempt archiving + `retry_log.json`.
6. `PipelineConfig.max_driver_attempts` / `retry_stderr_chars` + CLI flag.
7. Prompt additions (multi-turn contract + output-by-reference).
8. `_classify_role` + `parameter_roles` in spec.
9. Tests.

Steps 1–6 land the retry loop. Steps 7–8 cover the prompt/spec gaps the
loop exposes. Step 9 throughout.

## Completion criteria

- A characterization run on a fixture with an intentionally broken initial
  driver template succeeds within ≤3 attempts and emits a `retry_log.json`
  showing the recovery.
- All new tests pass.
- A void-returning fixture (e.g., a kernel that writes to `T& out`) runs
  end-to-end without manual driver hacks.
- The existing 6 calibration fixtures continue to pass on the first
  attempt (no regression).

## Out of scope

- Retrying runtime failures.
- Re-running cmake configure with different flags as a retry strategy
  (the LLM only sees and edits the driver source).
- Cross-kernel learning (each kernel's retry history is isolated).
- Token budget accounting in addition to attempt counting.
