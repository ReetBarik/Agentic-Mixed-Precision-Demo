# STOP #K probe evidence — Subtask 5 forwarding-overload compile/runtime battery

Faithful single-TU probes built against the REAL headers + REAL Kokkos:
  g++ -std=c++20 -w -I<repo>/src -I<repo>/third_party/include -I~/kokkos-install/include \
      probe.cpp -L~/kokkos-install/lib64 -lkokkoscore -lkokkoscontainers -ldl

Real promoted call reproduced (from boundary.promote_region_block on kokkosUtils.h:706):
    ql::Lnrat<Kokkos::complex<double>, double, double>(v__ff /*ddouble*/, x__ff /*ddouble*/)

## Findings (all with real types TOutput=Kokkos::complex<double>, TMass=TScale=double)

| candidate forwarding overload                                   | compile | runtime            |
|-----------------------------------------------------------------|---------|--------------------|
| A: 3tp, (ddouble,ddouble) args, `return ::ql::Lnrat<TOutput,TMass,TScale>(a,b)` (subtask-3 "working" B12 shim) | OK | **INFINITE RECURSION → segfault** (depth counter proves self re-selection) |
| D: same but `return ::ql::Lnrat<ddouble,TMass,TScale>(a,b)`     | OK      | **INFINITE RECURSION** |
| WIDEN: `return ::ql::Lnrat<TOutput,ddouble,ddouble>(a,b)`       | OK      | **INFINITE RECURSION** |
| B: 2tp `<TMass,TScale>`, (ddouble,ddouble) args                 | FAIL (call site names 3 explicit template args) | — |
| C: 2b hand-written `(const TMass&,const TMass&)` args           | FAIL (ambiguous / arg mismatch at real types) | — |
| "recursion-safe": cast args to double, call primary#2 at TScale=double | OK | runs — **but computes at DOUBLE precision (narrowed), C9 violation** |

## Root cause (structural, not variance)

1. C++ overload resolution selects `ql::Lnrat` by ARGUMENT types `(ddouble,ddouble)`, NOT by the
   explicit template-argument list. So ANY injected `ql::Lnrat(ddouble,ddouble)` overload whose body
   calls `ql::Lnrat(ddouble,ddouble)` re-selects ITSELF regardless of the `<...>` it writes → infinite
   recursion.
2. Unlike the `_MATH_FN_NAMES` bridges, there is **no vendored `quad::ddfun::Lnrat` / `ddilog`** to
   forward to (grep of third_party/include: none). The only same-name target is the app primary, and
   the primary is NOT instantiable at extended precision without a large support surface
   (`ql::Imag/Real/Sign/_ipio2/iszero/kLog/kAbs` on ddouble) — its body uses `.imag()`, complex-log
   branch selection, a regulator constant, etc. That surface is exactly the "semantics" the Subtask
   premise assumed away.
3. `Lnrat` is a LEAF callee: it is never on a chain's promoted line-set, so the reroute machinery
   never clones it into a `Lnrat_B10` variant (verified: `Lnrat` appears in ZERO fanout manifests
   across the entire run history; `ddilog`/`Li2omx2` DO get cloned because their bodies ARE on the
   chain). A rename-based escape from the recursion is therefore unavailable on the current design.

## Conclusion

The Subtask-5 premise — "`Ns::fn<Ts>(promoted)` on a visible primary template is a mechanical,
semantics-free template forward" — is **FALSE for `ql::Lnrat` and `ql::ddilog`**. A deterministic
emitter producing the mechanical forward would emit a build/runtime-breaking overload (STOP #K), which
is exactly the false-positive the conservative parser must refuse. Both symbols therefore classify as
**NON-forwardable**, and the emitter emits nothing for them — leaving them on the LLM path (a genuine
capability gap, not variance). See TIER_B_STAGE2_SUBTASK_5 report.
