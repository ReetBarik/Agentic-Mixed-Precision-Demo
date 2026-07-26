#pragma once
// SOURCE_HASH: f3412da736ac75f36f52c06723e6cb406988a423aa165475d6198e41e6ba1eba
#include <dd_math.hpp>
#include <dd_complex.hpp>

// Region:
//   const TOutput lnarg = TOutput(-ql::Lnrat<TOutput, TMass, TScale>(v, x)
//                                 - ql::Lnrat<TOutput, TMass, TScale>(w, y));
//
// TOutput is promoted to quad::ddfun::ddouble on this chain link. The region
// calls ql::Lnrat<TOutput, TMass, TScale>(...) — a namespace-qualified call
// into namespace ql — and constructs a TOutput from the extended sum.
//
// C3 (namespace-qualified math bridge): ADL does not rescue `ql::Lnrat(...)`;
// the primary template would deduce TOutput=ddouble but internally compute at
// double, and even if it forwarded correctly its return path must stay ddouble
// per C9 (this call sits on the promoted chain — its result feeds `lnarg`,
// which then propagates further). We therefore inject a ddouble-returning
// specialization of ql::Lnrat into namespace ql so the qualified call site
// resolves to an extended-precision implementation whose value stays ddouble
// end-to-end.
//
// C5/C7: the target library owns the function template `ql::Lnrat<TOutput,
// TMass, TScale>(a, b)`. We provide a more-specialized overload that fixes
// TOutput to the concrete quad::ddfun::ddouble while leaving TMass/TScale as
// template parameters, so the region's explicit template argument list
// <TOutput=ddouble, TMass, TScale> selects our overload over the primary.
//
// Lnrat(a, b) computes log(a/b) in the underlying library with a small-value
// regulator branch. Its exact source definition is not visible to this shim,
// so we cannot derive a closed-form ddouble implementation here.

// Rule R4 escape hatch: the region calls ql::Lnrat, whose source definition
// (including its regulator constant, branch structure, and mass/scale
// handling) is not supplied to this shim. We cannot emit a ddouble-returning
// specialization that faithfully reproduces its semantics without seeing that
// source, and any placeholder would silently narrow the chain-internal value
// back to double at this seam, defeating the whole-chain promotion.
//
// Rule R3 cascade for the regulator constant Lnrat depends on internally:
//   step 1 rejected: no vendored dd_lnrat_reg() factory exists.
//   step 2 rejected: no known hex (hi, lo) pair — the constant's identity
//                    and value are internal to ql::Lnrat and not exposed.
//   step 3 rejected: the source definition of ql::Lnrat (and any regulator
//                    literal it contains) is not visible or supplied here.
// Therefore step 4 applies.
//
// UNCLASSIFIED: ql::Lnrat<quad::ddfun::ddouble, TMass, TScale>
// Rule R4 unclear because: the source of ql::Lnrat is not supplied, so a
// chain-internal ddouble-returning specialization cannot be emitted without
// silently narrowing at this seam (C9 violation).
// Human review needed before this shim can compile.
#error "DD Chain Integrator: ql::Lnrat<ddouble, TMass, TScale> requires manual classification"