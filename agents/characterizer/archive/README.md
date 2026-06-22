# Archived characterizer plans

These plans drove the v1 characterizer slice and are kept here for paper trail.
Their content is reflected in code that has already landed on `langgraph-agents`.

| File | Status |
|---|---|
| `PLAN.md` | Original v1 vertical-slice plan. Slice is implemented; see top-level `README.md` "Characterizer Agent" section. |
| `PLAN_retry_loop.md` | Driver compile-retry loop design. Implemented and landed. |
| `NEXT.md` | Follow-up "remaining work" notes (cLn, Lnrat, retry loop, relative paths, test expansion). Most items done; the framing ("done enough to start strategy agent work") is superseded by the whole-app plan below. |

**Current authoritative plan:** [`PLAN_whole_app_characterization.md`](../../../PLAN_whole_app_characterization.md) at the repo root.

Live, open work items that survived from `NEXT.md`:

- More leaf kernels (fndd, ddilog) — deprioritized; whole-app pipeline will surface which leaves actually fire and which need characterization
- Test suite expansion (`test_build_run_stub.py`, `test_characterizer_e2e.py`) — still nice to have; not blocking
- Snapshot tests for prompts — explicitly deferred
