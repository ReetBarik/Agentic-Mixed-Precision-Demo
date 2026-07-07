# Plan: Tracked v0.4 — anonymous literals

**Status:** v0.4 — non-breaking addition (design record). Extends the v0.3
provenance model with a third leaf factory; the journal schema is unchanged.
[PLAN-v0.3.md](PLAN-v0.3.md) remains the record of the provenance redesign this
builds on, and [PROVENANCE.md](PROVENANCE.md) is the reference for the model
itself (including the [Literals](PROVENANCE.md#literals-anonymous-scalars)
section).

## Goal

Give anonymous scalars a way to enter the tracked graph with a valid, walkable
id — closing the last hole through which the journal emitted records with an
empty `in` reference.

## Motivation — the empty-`in` bug at the root

Under v0.3, values built with the bare `Tracked<T>(T v)` constructor carry an
**empty id** and empty provenance. Any op consuming one emits a record with `""`
in its `in` field, severing graph traceability at that node.

A post-v0.3 B13 journal showed **6,286 records** with `in: ["", ...]` — `sub`
(4,351), `neg` (1,165), `mul` (770). Every one traced back to `Complex<T>`
internals:

- **Scratch constants in op bodies** — `Tracked<T>(T(0.5))`, `Tracked<T>(T(2))`
  in `sqrt`. These have textbook names and deserve them.
- **Anonymous scalar-to-complex promotions** — the `Complex(T re)` component
  constructors and every mixed-scalar operator wrap a bare `T` via
  `Tracked<T>(T(s))`. Triggered by legitimate user code such as qcdloop's
  `kPow` (`TOutput temp = TOutput(1.0);`). These scalars are anonymous by design.

The `Tracked<T>(T v)` constructor itself is a legitimate ergonomic escape hatch;
removing it would break real user code. What was missing was a path for
*anonymous* scalars to enter the graph with an auto-generated id.

## Decisions

### 1. A third factory: `literal()`, completing the role taxonomy

| Factory              | Role             | Attribution      | Provenance    |
| -------------------- | ---------------- | ---------------- | ------------- |
| `track(name, v)`     | source variable  | attribution root | `prov_vars`   |
| `constant(name, v)`  | named constant   | audit-only       | `prov_consts` |
| `literal(v [, loc])` | anonymous scalar | auto-id, neither | *(neither)*   |

`literal()` builds a `Tracked<T>` and assigns it a generated id via the existing
`detail::make_id`, so scope-suffix and callsite-counter behavior match
op-generated ids. It seeds neither `prov_vars` nor `prov_consts`.

### 2. Rationale for auto-ids (not empty, not names)

An anonymous scalar shouldn't need a name to be *pointed at*. Giving it an id
keeps the value graph fully connected (`in` never contains `""`, `parents()` and
`trace_ancestors()` see it) while keeping it out of attribution (`trace_sources()`
still returns only source variables). The raw `Tracked<T>(v)` constructor stays
as the deliberate opt-out for callers who want neither.

### 3. Id shape and the `_lit@` marker

```
_lit@<file>:<line>#<counter>[@<scope>]   // with SourceLocation
_lit@?#<counter>[@<scope>]               // without
```

The `_lit` op tag is deliberate so downstream tooling can distinguish literals
from op-generated ids with a single string check (`id.rfind("_lit@", 0) == 0`)
and filter or aggregate them.

### 4. `complex.hpp` migration — the root fix

- **Anonymous scalars → `literal()`**: the `Complex(T re, T im)` component
  promotions, the eight mixed-scalar operators, `sqrt`'s runtime-selected
  `sign(im)` (±1, *not* a named constant — `constant()` dedups by name, so a
  shared `"sign"` id would conflate `+1` and `-1`), and the imaginary padding of
  a real-promoted complex (`Complex(Tracked<T> re)`), which is a structural
  artifact rather than a zero the user's math named.
- **Named mathematical constants → `constant()`**: `sqrt`'s
  `0.5 → constant("half")` and `2 → constant("two")` — coefficients the algorithm
  invokes by name.

Rule of thumb: the split is **intent, not spelling**. A constant the computation
actually referred to by name (`half`, `two`) is a `constant()`; a scalar that is
machinery — a scratch value, a runtime-selected sign, or structural padding —
is a `literal()`, even when it's a hardcoded `T(0)`. `prov_consts` should reflect
the constants in the user's math, not the library's plumbing.

All three factories (and the raw constructor) are numerically identical — same
`value`, `rel_err`, `cond` — so no calibration assertion moves; only ids and
provenance categories change.

## Non-goals

- No change to `Tracked<T>(T v)` or the default constructor — both stay as-is;
  reaching for them directly still opts out of graph traceability.
- No changes to `tracked.hpp` op bodies or `ops.hpp` — they already generate
  proper ids.
- `literal()` takes **no** user name. If a value warrants a name, that's
  `constant()`. Keeping `literal()` name-free enforces the taxonomy.

## Downstream (separate follow-up)

Bumping the Tracked subtree in Agentic-Mixed-Precision-Demo and re-running B13 is
out of scope here. Expected outcome once bumped: the 6,286 anonymous records
become valid `_lit@…` ids in their `in` fields, total record count is unchanged,
and `trace_sources` on the top log record still returns the same six source
variables (literals excluded by design).
