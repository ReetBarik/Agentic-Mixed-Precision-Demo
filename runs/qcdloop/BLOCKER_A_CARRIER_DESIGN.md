# Blocker A — Chain-Emission Carrier Under-Widening: Design Notes

Status: **design finalized** (2026-07-25). Implementation follows this document.
Scope: Phase 2f Tier-B Stage-1 chain-scoped double-double (dd) promotion.

## 1. The defect

`chain_promote` widens a promoted variant's region *bodies* at the chain's listed
lines, but leaves **carrier declarations** at caller precision. A **carrier** is a
variable declared OUTSIDE the chain's line set, written by one interior chain line,
and read by another interior chain line. The interior write lands in a
caller-precision carrier decl → the extended (dd) value is truncated back to double
→ the interior 2d-B `chain_write_truncation` gate correctly observes a lossy
write-boundary and rejects the patch (`patcher_status: write_truncation`).

The current run outcomes confirm this exactly:

| Integral | outcome | patcher_status |
|----------|---------|----------------|
| B10 | apply_failed | write_truncation |
| B13 | apply_failed | write_truncation |
| B14 | apply_failed | write_truncation |

### Worked example — B10 `kokkosUtils.h` ddilog

```
157   TMass Y, S, A;                                   // <-- carrier decl (multi-declarator, no init)
...
174   Y = (TMass(-ql::Constants<TMass>::_one()) - T) / T;   // interior chain line: WRITES Y
177   A = TMass(... + A * (...));                            // interior chain line: WRITES A, READS A
199   const TMass H = Y + Y - TMass(ql::Constants<TMass>::_one());  // interior chain line: READS Y
212   return -(S * (B0 - H * B2) + A);                      // interior chain line: READS A
```

`Y` is written at :174 and read at :199 — both interior chain lines → `Y` is a
strict carrier. `A` is written at :177 and read at :177/:212 — strict carrier.
Their declaration at :157 is a bare multi-declarator `TMass Y, S, A;`. The promoted
variant widens the bodies of :174/:177/:199/:212 to dd but leaves :157 at `TMass`
→ the `Y =`/`A =` writes truncate → gate rejects.

**The fix belongs in chain emission (widen the carrier declaration alongside the
line bodies), NOT in the gate. No gate is weakened.**

## 2. Carrier definition (finalized)

A name `v` is a **strict carrier** of chain `C` iff ALL of:

1. `v` is **written** by at least one interior chain line of `C`;
2. `v` is **read** by at least one *other* chain line of `C`
   (the write-line and the read-line differ, OR the same line both reads and
   writes `v` AND `v` is also read/written on another chain line — i.e. the
   value genuinely crosses a chain-line boundary);
3. the declaration of `v` lies **outside** the chain's line set (a decl *inside*
   the line set is already widened by the body transform — out of scope, see §6);
4. `v` is not a write target of the **outermost** (min-depth) region — the
   outermost region's writes are the designed chain exit boundary and are demoted
   to caller precision on purpose (excluded from the carrier closure, §5).

"Interior chain line" = a chain line whose enclosing region is NOT the outermost
(min-depth) region.

### Conservative policies (v1, all confirmed)

- **Conditional writes**: any *potential* carrier-write (a write under a branch,
  loop, or guard) is treated as a carrier-write. We over-widen rather than miss.
- **Multi-declarator siblings**: widening `TMass Y, S, A;` to dd widens `S` too
  even if `S` is not itself a carrier. Permitted by the conservative policy
  (over-widening a same-type sibling is safe; it never truncates).
- **Function-parameter carrier** (decl traces to a function parameter): REFUSE.
  Terminal status `chain_carrier_unwidenable`.
- **Globally-visible carrier** (decl traces to a module/class member, or an output
  container like `res(i,0)`): REFUSE. Terminal status `chain_carrier_external`.

## 3. The B14 `fac` tension — resolved

B14 `B2m.h`: `fac` is declared `TOutput fac;` (:396), written on chain line :401
(`fac = TOutput(...)`), and read only at the NON-chain output stores
`res(i,1)=fac/...` (:404) and `res(i,0)=fac*wlogtmu` (:405).

`fac` is written on a chain line but **not read by another chain line** → it fails
carrier condition (2) → it is **NOT a strict carrier**. Fix A leaves `fac` alone.
Its fate is decided by the *existing* machinery:

- If `fac`'s write region is the outermost/min-depth region → exempt from
  `chain_write_truncation` (designed output boundary) → patch proceeds.
- If interior → genuine `write_truncation` terminal (correct; not a carrier bug).

`chain_carrier_external` is reserved strictly for a name that IS a strict carrier
(conditions 1–3 hold) whose declaration traces to a global/member/output container.
`fac` never reaches that test. **No new status is needed for `fac`.**

Likewise B10 `H` (`const TMass H` declared at :199, read at :212): `H` is declared
*inside* the chain line set → fails condition (3) → out of scope; the body
transform already widens :199. It does not trip the gate because `TMass` is
unrecognized by `write_truncation_inert` (a decl_write of an unrecognized type
bails the gate to `False`).

## 4. Carrier-closure algorithm (chain_promote.py)

Per chain `C`:

1. **Collect interior writes.** For each interior chain line, derive the set of
   names written on that line (`region_scan.region_writes_from_source` over the
   single line; plus bare `name =` Case-B detection).
2. **Collect chain-line reads.** For each chain line, derive names read
   (`region_scan.region_reads_from_function` / token scan).
3. **Candidate carriers** = names written on an interior line AND read on a
   *different* chain line.
4. **Trace each candidate to its declaration** (source-derivable via libclang /
   region_scan / token lexer). Reject candidates whose decl is inside the chain
   line set (condition 3, handled by the body transform).
5. **Exclude outermost writes** (condition 4).
6. **Classify each surviving carrier's decl site**:
   - local var decl in a function body reachable in the variant → **widenable**;
     record `(file, decl_line, name, dd_type)` for §7.
   - function parameter → emit `chain_carrier_unwidenable`, abort this chain.
   - global / class member / output container → emit `chain_carrier_external`,
     abort this chain.
7. Pass the widenable carrier name set into the boundary transform as
   `carrier_names` (§8), and the decl-widen records into the VariantSpec (§7).

## 5. Outermost exclusion

The outermost region (min-depth in `region_meta`, `depth = min(len(p) for p in
paths) - 1`) owns the chain's exit boundary. Its write targets are demoted to
caller precision by design. Carrier condition (4) removes any name that is a
write target of the outermost region from the carrier closure, so we never try to
widen the designed exit sink. This mirrors the existing `chain_write_truncation`
min-depth skip.

## 6. In-scope vs out-of-scope decls

| decl site | in carrier closure? | handled by |
|-----------|---------------------|------------|
| outside chain lines, local var | YES (widen) | §7 VariantSpec decl-widen |
| inside chain lines | NO | existing body transform |
| function parameter | REFUSE | `chain_carrier_unwidenable` |
| global / member / output container | REFUSE | `chain_carrier_external` |
| outermost region write target | NO | designed exit boundary (§5) |

## 7. Emission — VariantSpec carrier decl-widening

`fanout.Promote` / `VariantSpec` gain a carrier decl-widen payload:

- New field (working name `carrier_decls`): a list of
  `(decl_line, orig_type_token, dd_type_token, name?)` records scoped to the
  variant's file, produced by §4 step 6.
- `render_variant(spec)` applies carrier decl-widening in the SAME descending
  line-order pass as `promotes` (descending by line so earlier edits don't shift
  later line numbers), rewriting the type token on the decl line to the chain's
  internal dd type (`Kokkos::Experimental::DoubleDouble` / `DoubleDoubleComplex`).
- Multi-declarator bare decls (`TMass Y, S, A;`) rewrite the leading type token,
  widening all same-type siblings (§2 conservative policy). A dedicated scanner is
  needed because the existing `_scan_decls` requires `<type> <name> =` and misses
  bare/multi-declarator forms.
- `to_json` / `from_json` / `merge` extended to round-trip the new field.

## 8. Boundary transform — `carrier_names` awareness

`carrier_names` threads through `_compute_promotion` → `promote_region_block` →
`write_truncation_inert`. A name in `carrier_names`:

- is **seeded into the `promoted` dataflow set** (its dd value flows through);
- is **excluded** from `pure_reads`, `caseB`, and `decl_writes` (it is neither a
  read-only input nor a truncating sink — its decl is widened in the variant);
- is **NOT renamed / NOT seeded with a `r__`/`w__` boundary alias** and NOT
  demoted on exit;
- a carrier-write **counts as a landing** so the no-op guard
  (`if not decl_writes and not caseB and (two_limb or not pure_reads): return
  ..., False`) does not spuriously fire when the only region effect is a carrier
  write.

`write_truncation_inert` with `carrier_names`: carrier writes are removed from the
`caseB`/`decl_writes` sets it inspects, so a region whose only "truncating" writes
are now-widened carriers no longer reports inert → the gate stops rejecting the
(now correct) patch. **The gate logic is unchanged; it simply sees the corrected
carrier classification.** No gate is weakened.

## 9. New terminal statuses

`agents/patcher/result.py`:

- `CHAIN_CARRIER_UNWIDENABLE = "chain_carrier_unwidenable"` — a strict carrier's
  decl is a function parameter (v1 refuses to rewrite signatures).
- `CHAIN_CARRIER_EXTERNAL = "chain_carrier_external"` — a strict carrier's decl is
  global / class member / output container (v1 refuses to widen shared state).

Both added to the STATUSES frozenset and given `err.kind` entries. Wired in
`dispatch.py::_gen_chain` from the `ChainFanoutResult`.

## 10. Prompt rule C10 (chain integrator)

`agents/chain_integrator/system_prompt.txt` + `agent.py` `_SPEC.constant_note`:

> **C10 (carrier-widening invariant).** A carrier variable — declared outside the
> chain's line set but written by one chain link and read by another — is widened
> to DoubleDouble at its declaration by the boundary/emission layer. Your shim MUST
> treat such a variable as DoubleDouble end-to-end: never re-narrow it to double at an
> assignment or overload return. If a value produced by one link is consumed by a
> later link, it stays DoubleDouble across the whole chain.

## 11. Files touched

| file | change |
|------|--------|
| `agents/patcher/chain_promote.py` | carrier-closure computation (§4), classify decl sites, feed `carrier_names` + decl-widen records |
| `agents/integrator_base/boundary.py` | `carrier_names` param through `_compute_promotion` / `promote_region_block` / `write_truncation_inert`; multi-declarator scanner |
| `agents/patcher/fanout.py` | `Promote`/`VariantSpec` carrier decl-widen field; `render_variant` widening pass; json round-trip |
| `agents/patcher/result.py` | two new terminal statuses + err.kind |
| `agents/patcher/dispatch.py` | wire carrier statuses in `_gen_chain` |
| `agents/chain_integrator/system_prompt.txt` + `agent.py` | C10 rule |
| `tests/patcher/fanout/test_chain_promote.py` (+ boundary tests) | carrier unit tests |

## 12. Success criterion (re-run)

NOT "all 3 accept." Each of B10/B13/B14 must reach a real diagnostic terminal
state: accepted with a measured kernel-scoped lift, OR one of
`chain_no_lift` / `chain_regression` / `chain_carrier_unwidenable` /
`chain_carrier_external` / another well-diagnosed precision state. The carrier fix
removes the *spurious* `write_truncation` rejection so the true precision outcome
becomes observable.

Re-run params: B10,B13,B14 only (skip B12), seed 12345, sample_count 5000, entry
BO, kernel-scope gate, +0.5 lift, STOP-and-report.
