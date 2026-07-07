# Plan: Tracked v0.3 — provenance redesign

**Status:** v0.3 — schema-breaking hard cutover of the provenance model
(design record). Supersedes the flat-`provenance` model of the earlier releases;
`PLAN.md` and `PLAN-v1.1.md` remain the historical records of the value/error and
complex designs.

## Goal

Turn provenance from a heuristic into a data model. Give every `Tracked` value a
stable identity, record operand identities directly in the journal, and split
the flat provenance set into two honest categories — so that "which inputs fed
this value?" is answered by a graph walk instead of a guess.

## Motivation — the aliasing bug at the root

v0.2's journal `in` field was `primary_id(operand.provenance_)` =
`*provenance_.begin()` = the alphabetically-first name in a `std::set`. Library
constants used a `_`-prefix convention (`_four`, `_half`) which sorts before real
variable names, so `in` on any derived value systematically reported a *constant*
rather than the actual operand — e.g. `log(2.15e-8)` logged `in=[_four]`.

The root cause is structural: a derived value had no per-value id, so the journal
had nothing to point at when naming an operand. Any single-name pick from the
provenance set is a guess. Fix the data model and the aliasing dissolves — there
is no "pick one" step left to get wrong.

Full rationale, id scheme, scope semantics, and migration notes live in
[PROVENANCE.md](PROVENANCE.md); this file is the decision log.

## Decisions

### 1. Two-category provenance, two factories

- `track(name, value)` → source variable, seeds `prov_vars`, is an attribution
  root.
- `constant(name, value)` → named constant, seeds `prov_consts`, recorded for
  audit but never reported as an origin.

The `_`-prefix convention is **retired as a dependency**: no library code
inspects the prefix. It survives only as an optional cosmetic cue.

### 2. Stable value ids

Every value carries a `TrackedId` (`std::string`):

- source/constant → the given name;
- derived → `<op>@<file>:<line>#<counter>[@<scope>]`.

The callsite counter is thread-local, keyed on `(file, line, op)`, and reset by
`journal::clear()`. This gives the **ID-stability property**: same call sequence
+ same scopes ⇒ same ids, run to run (guarded by a reproducibility test).

### 3. RAII scopes

`tracked::scope(ctx)` pushes a thread-local context string; the joined stack
(nested scopes separated by `/`) is appended to generated ids as `@<scope>`.
Scopes are lexical (owned by the RAII object) and are **not** reset by `clear()`.

### 4. Journal schema (breaking)

- add `id`;
- `in` now holds direct-operand ids verbatim (no `primary_id`);
- replace `prov` with `prov_vars` + `prov_consts`, both `std::set`-backed for
  dedup and fast membership.

Sets, not lists: attribution is a bag question (membership/dedup matter, order
does not). Ordered causal ancestry is recovered by `trace_ancestors`, which is
the honest place for order.

### 5. Graph-walk helpers owned by the library

`tracked::journal::{index, parents, trace_sources, trace_ancestors}`. The library
owns the DAG traversal so consumers don't reinvent it. `trace_sources` returns
source-variable roots only (constants and opaque markers excluded);
`trace_ancestors` returns all ancestors in topological order. O(depth) after a
lazily-built, `emit`-invalidated `id → record-index` cache.

### 6. Opaque markers — deferred (option 3)

The `opaque` `fn_name` marker is neither a source variable nor a named constant.
Rather than add a third category now (scope creep) or fold it into `prov_consts`
(semantically wrong — a black box is not a constant), v0.3 defers:

- `in = [fn_name, input_ids...]` — marker leads, operand ids follow so the graph
  stays traversable through the barrier;
- `fn_name` is kept out of both prov sets (retiring v0.2's fold-into-provenance);
- `trace_sources` ignores the marker and resolves to the forwarded sources.

Follow-up issue tracks promoting the marker to a first-class `prov_opaque`
category if opaque-provenance querying becomes important.

## Scope of the change

Touched: `include/tracked/{journal,tracked,ops,complex}.hpp`, all existing tests
(field renames only — no value/cond/rel_err assertion shifts), a new
`tests/tracked/test_provenance_v03.cpp`, `README.md`, and this doc + `PROVENANCE.md`.

## Out of scope for v0.3

- **Downstream consumers.** No consumer repo is touched. The characterizer
  (Agentic-Mixed-Precision-Demo) gets its own PR after v0.3 lands and is tagged.
- **A first-class opaque category** (`prov_opaque`) — deferred, see Decision 6.
- GPU/device support, extended precision — unchanged non-goals from v1.

## Verification

- `ctest` green: all pre-existing tests pass with updated field names; the 11 new
  provenance tests pass.
- A hand-written driver flushes ~10 ops and is eyeballed for schema readability,
  correct `in`-graph resolution, and correct `trace_sources`.

## Open questions called out at review

- **Opaque marker handling** — deferred (Decision 6); is that the right call for
  the first release, or is `prov_opaque` worth doing now?
- **ID stability** — same location + counter + scope ⇒ same id, run to run;
  covered by a reproducibility test. Confirm this is a property we want to commit
  to publicly.
- **`std::map` vs `std::unordered_map` for callsite counters** — chose `map` for
  simplicity (small maps, ordered key via `std::tie`); a performance choice, not
  a correctness one.
