# Provenance and attribution (v0.3)

This document describes the provenance model introduced in the **v0.3** hard
cutover: what a value's identity is, how attribution works, and how to migrate
from v0.2. It is the reference for the `id` / `prov_vars` / `prov_consts` fields
in the journal and for the graph-walk helpers in `tracked::journal`.

## Why v0.3 exists — the aliasing bug at the root

In v0.2 every journal record's `in` field was computed by
`detail::primary_id(operand.provenance_)`, which returned `*p.begin()` on a
`std::set<std::string>` — the **alphabetically first** provenance name. The
codebase used an underscore-prefix convention for library constants (`_four`,
`_half`, `_pi`), and `_`-prefixed names sort before real variable names
(`m1[79]`, `sibar`). So on any derived value, `in` reported a *constant* instead
of the actual operand:

```
log(2.15e-8)  →  in = ["_four"]      # obviously not log of "_four"
```

This was not a formatting bug — it was a **data-model** bug. A derived
`Tracked` value had no stable per-value identity, so when the journal wanted to
say "this specific intermediate value was the operand," it had nowhere to point.
Any single-name pick from the flat provenance set is a guess.

v0.3 fixes the data model: **every value carries a stable id**, and `in` records
the operands' ids verbatim. With real identities to point at, the aliasing
dissolves — there is no longer a "pick one name" step to get wrong.

## The factory taxonomy

Three factories cover the roles a leaf value can play. Pick by what the value
*is*, not by how it happens to be spelled at the call site:

| Factory | Role | Seeds | Is an attribution root? |
|---|---|---|---|
| `tracked::track(name, value)` | source variable | `prov_vars` | **yes** |
| `tracked::constant(name, value)` | named constant | `prov_consts` | no |
| `tracked::literal(value [, loc])` | anonymous scalar | *(neither)* | no |

A **source variable** is an input whose value your computation is *about*
(`m1[79]`, `mu2`, a measurement, a sample). Attribution — "which inputs
ultimately fed this result?" — resolves to source variables.

A **named constant** is a literal with a name for readability and audit (`pi`,
`two`, `half`). It participates in provenance so you can see it was involved,
but it is never reported as the *origin* of a value; asking "where did this
number come from?" should not answer "from the constant 2."

An **anonymous literal** (v0.4) is a bare scalar that enters the graph without a
semantic name — a scratch value the caller doesn't care to name. It gets an
auto-generated id so downstream ops can point at it, but seeds neither provenance
category. See [Literals](#literals-anonymous-scalars) below.

Provenance is unioned per category as ops combine values: an op's result carries
the union of its operands' `prov_vars` and, independently, the union of their
`prov_consts`. Literals contribute to neither, so they leave both sets untouched.

### The underscore convention is retired

v0.2 leaned on the `_`-prefix convention both cosmetically *and* structurally
(it determined sort order, which determined `in`). v0.3 removes the structural
dependency entirely: **no library code inspects the name prefix.** You may still
name constants with a leading underscore if you like the visual cue — it is now
purely cosmetic. The honest signal that "this is a constant" is that it was
created with `constant()` and lives in `prov_consts`.

## Value ids

Every `Tracked` value has a stable `TrackedId` (`= std::string`) assigned at
construction.

- **Source variables / constants**: the id *is* the user-given name — `m1[79]`,
  `pi`.
- **Derived values**: a generated id of the form

  ```
  <op>@<file>:<line>#<callsite_counter>[@<scope>]
  ```

  for example `sub@physics.cpp:127#3@sample=79`.

The **callsite counter** is thread-local and keyed on `(file, line, op)`. The
first op produced at a given call site is `#1`, the next `#2`, and so on. The
counter resets on `tracked::journal::clear()`, so a fresh run reproduces the same
ids for the same sequence of calls (see *ID stability*, below).

Operator-form calls (`a - b`) capture no source location, so their ids use a
placeholder file and a separate anonymous counter: `sub@?#<n>`. Prefer the named
free functions with `TRACKED_HERE` (`sub(a, b, TRACKED_HERE)`) when you want
readable, source-anchored ids.

Values built with the bare `Tracked<T>(v)` constructor (e.g. a `0.5` created
inside a kernel) have an **empty id** and empty provenance — the constructor is
an ergonomic escape hatch, and reaching for it directly opts out of graph
traceability. Prefer `literal(v)` for anonymous scalars (auto id, still
walkable) or `constant()` for anything that deserves a name. See below.

## Literals (anonymous scalars)

`tracked::literal(value, loc = {})` promotes a bare scalar into the tracked graph
**without** giving it a semantic name. It is the third corner of the factory
taxonomy: unlike `track()` and `constant()`, it seeds no provenance, but unlike
the raw `Tracked<T>(v)` constructor it *does* get a stable id, so an op that
consumes it emits a record whose `in` points at a real node instead of an empty
string.

**Why it's distinct from `constant()`.** A constant has a semantic name (`pi`,
`half`) and lands in `prov_consts` for audit. A literal has neither — it is a
scratch value whose identity doesn't matter to attribution. Use it precisely
when a scalar *doesn't* warrant a name; if it does, that's what `constant()` is
for.

**Id shape.** Literals reuse the same generator as derived values, with a `_lit`
op tag:

```
_lit@<file>:<line>#<counter>[@<scope>]     // with a SourceLocation (TRACKED_HERE)
_lit@?#<counter>[@<scope>]                 // without (operator-form / default loc)
```

The `_lit@` prefix is deliberate: downstream tooling can **filter or aggregate**
literals with a single string check (`id.rfind("_lit@", 0) == 0`) — e.g. to hide
scratch nodes from a graph view, or to roll up all literals into one bucket.

**Graph-walk semantics.**

- `trace_sources(id)` — literals **never** appear. They are not source variables,
  so attribution ignores them (behavior unchanged from v0.3).
- `trace_ancestors(id)` — literals **do** appear. They are valid causal nodes;
  they simply show up now that they carry ids.

**Where it's used internally.** `Complex<T>`'s bare-scalar promotions route
through `literal()` — the `Complex(T re, T im)` component constructors, the
mixed-scalar operators (`z * 3.0`, etc.), and the imaginary padding of a
real-promoted complex (`Complex(Tracked<T> re)`, whose `im` is a structural zero,
not a zero the user's math named). This is the root fix for the empty-`in`
records that anonymous scalar-to-complex promotions produced before v0.4.

By contrast, `half`/`two` in `sqrt` *are* named `constant()`s — they are
mathematical coefficients the algorithm invokes by name. The line is intent:
`prov_consts` records constants the computation referred to, not machinery.

**Guidance.** Prefer `constant()` for anything with a semantic name; reach for
`literal()` only when the scalar truly doesn't warrant one. The raw
`Tracked<T>(v)` constructor remains available but is the opt-out — it gives
neither an id nor provenance.

## Scopes

`tracked::scope` is an RAII helper that pushes a context string onto a
thread-local stack. The joined stack (nested scopes separated by `/`) is
appended to every generated id as `@<scope>`:

```cpp
tracked::scope run("run=A");
{
    tracked::scope s("s=1");
    auto r = sub(a, b, TRACKED_HERE);   // id ends "@run=A/s=1"
}
// here, ids end "@run=A" again
```

Scopes are **lexical** and owned by the RAII objects; they are *not* reset by
`journal::clear()`. They are the natural way to tag per-sample or per-iteration
work so that ids from different iterations don't collide and can be filtered
apart downstream.

## Journal schema (v0.3)

```json
{
  "op":          "sub",
  "at":          "physics.cpp:compute:127",
  "id":          "sub@physics.cpp:127#3@sample=79",
  "in":          ["mul@physics.cpp:125#1@sample=79", "pi"],
  "val":         2.15e-8,
  "cond":        1.0,
  "rel_err":     2.22e-16,
  "prov_vars":   ["m1[79]", "mu2"],
  "prov_consts": ["pi", "two"]
}
```

Changes versus v0.2:

- **new `id`** — the stable identity of the produced value.
- **`in` carries direct-operand ids verbatim** — no `primary_id` heuristic. For
  a binary op it is `[a.id, b.id]`; for a unary op `[a.id]`.
- **`prov` split into `prov_vars` + `prov_consts`** — both are `std::set`-backed
  (dedup + fast membership) and serialized as JSON arrays.

### Why sets, not lists, for `prov_vars` / `prov_consts`

Attribution is a **bag** question: *did source X contribute? which set of
sources fed this?* Dedup and membership matter; order does not. A flat list
would invite readers to attach meaning to element order that the model does not
guarantee. Users who need ordered causal ancestry call `trace_ancestors(id)`,
which returns a topologically-ordered vector. The graph walk is the honest place
to recover order.

## Graph model and walk helpers

The journal is a DAG. Each record's `id` is a node; its `in` lists the ids of
the operands that produced it. Leaf ids (source variables, constants, literals,
opaque markers) never appear as a produced `id`, so they have no `in` edges. The
library owns the traversal:

```cpp
namespace tracked::journal {
  const std::unordered_map<std::string, std::size_t>& index();   // id → record idx (cached)
  std::vector<std::string> parents(std::string_view id);         // direct operands
  std::set<std::string>    trace_sources(std::string_view id);   // variable roots
  std::vector<std::string> trace_ancestors(std::string_view id); // full causal order
}
```

- **`index()`** builds and caches an `id → record-index` map over the current
  buffer. It is rebuilt lazily on the first query after any invalidating `emit`
  (or `clear`), so repeated queries between emits are cheap.
- **`parents(id)`** returns the record's `in`, or empty for a leaf / unknown id.
- **`trace_sources(id)`** BFS-walks backward through `in` edges and returns the
  set of ids that are **source variables** — i.e. that appear in some record's
  `prov_vars`. Constants (which appear only in `prov_consts`), literals, and
  opaque markers (which appear in neither) are excluded. This is the answer to
  "which source variables ultimately fed this value?"
- **`trace_ancestors(id)`** returns every causal ancestor (including `id`) in
  topological order: each id appears *after* all of its parents, with roots
  first and `id` last.

All three are O(depth) once the index is built.

## Opaque markers

`opaque(fn_name, value, inputs...)` records a value crossing a non-`Tracked`
boundary (a vendor/framework math call the library can't see into). The
`fn_name` is a **marker** — it is neither a source variable nor a named
constant, so it does not fit either provenance category cleanly.

For v0.3 we **defer** a first-class category and take the least-invasive path:

- The opaque record's `in` is `[fn_name, input_ids...]`: the boundary marker
  leads, followed by the forwarded operand ids so the value graph stays
  traversable *through* the barrier.
- `fn_name` is kept **out of** `prov_vars` and `prov_consts` (this retires
  v0.2's behavior of folding it into the flat provenance set).
- Because the marker is not in `prov_vars`, `trace_sources` naturally ignores it
  and still resolves to the real upstream source variables forwarded through the
  boundary.

**Follow-up (deferred):** promote the marker to a first-class `prov_opaque`
category + journal field if opaque-provenance querying becomes important. This
was weighed against folding it into `prov_consts` (rejected — a black-box
boundary is not a constant) and chosen to keep the v0.3 schema surface tight.
See the tracking issue.

## Migrating from v0.2

1. **Split your leaves.** Replace `track("pi", 3.14159)` for genuine constants
   with `constant("pi", 3.14159)`. Keep `track()` for the inputs your
   computation is actually about.
2. **Drop the `_`-prefix crutch.** It no longer affects anything structurally.
   Rename `_four` → `four` (as a `constant`) if you like; it's cosmetic now.
3. **Read the new fields.** `record.provenance` → `record.prov_vars` +
   `record.prov_consts`. `Tracked::provenance()` → `prov_vars()` /
   `prov_consts()`. The accessor `id()` is new.
4. **Stop guessing operands.** Anywhere you previously read `in` and tried to
   reconstruct the operand, `in` now *is* the operand ids. To recover attribution
   or ancestry, call `trace_sources` / `trace_ancestors` instead of
   post-processing the flat set.

## Worked example: a Monte-Carlo integrator with per-sample scopes

```cpp
#include <tracked/tracked.hpp>
#include <tracked/ops.hpp>

using tracked::track;
using tracked::constant;

double estimate(const std::vector<double>& xs) {
    auto half = constant("half", 0.5);
    tracked::Tracked<double> acc(0.0);

    for (std::size_t s = 0; s < xs.size(); ++s) {
        tracked::scope samp("sample=" + std::to_string(s));   // ids get "@sample=<s>"

        auto x   = track("x[" + std::to_string(s) + "]", xs[s]);
        auto x2  = mul(x, x, TRACKED_HERE);                   // integrand
        auto w   = mul(half, x2, TRACKED_HERE);              // weighted
        acc      = add(acc, w, TRACKED_HERE);
    }
    tracked::journal::flush("mc.jsonl");

    // Which sample inputs fed the final estimate?
    auto srcs = tracked::journal::trace_sources(acc.id());   // {"x[0]", "x[1]", ...}
    return acc.value();
}
```

Each sample's ops carry `@sample=<s>` in their ids, so records from different
samples never collide and can be grouped or filtered by scope downstream. The
`half` constant shows up in every record's `prov_consts` but never in
`trace_sources` — attribution correctly points only at the `x[s]` inputs.
