#pragma once
// SOURCE_HASH: f3412da736ac75f36f52c06723e6cb406988a423aa165475d6198e41e6ba1eba
#include <dd_math.hpp>
#include <dd_complex.hpp>

// Region: kokkosUtils.h:702
//   const TOutput lnarg = TOutput(-ql::Lnrat<TOutput, TMass, TScale>(v, x)
//                                 - ql::Lnrat<TOutput, TMass, TScale>(w, y));
//
// This region calls the templated helper ql::Lnrat<TOutput, TMass, TScale>(a, b)
// where TOutput has been promoted to quad::ddfun::ddouble along the chain.
// Per C9 (chain-internal contract), Lnrat is itself a chain link: its result
// feeds directly into the unary '-' and binary '-' that produce `lnarg`, which
// stays ddouble end-to-end. So the ddouble specialization/overload of Lnrat
// MUST return ddouble, never narrow to double.
//
// The vendored ddouble type already provides:
//   - unary operator- (ddouble)
//   - binary operator- (ddouble, ddouble)
//   - converting constructor ddouble(ddouble) (identity)
// so the outer TOutput(...) cast, the unary '-', and the binary '-' need no
// new shim support. The only thing this region references that could bind to
// a double-narrowing primary is the qualified call ql::Lnrat<...>(a, b) itself.
//
// Per C3 (namespace-qualified math bridge) + C5/C7 (function-template
// specialization keyed on the extended scalar), we inject a ddouble overload
// of ql::Lnrat into namespace ql so the qualified call resolves to a
// ddouble-returning function instead of the primary (which would narrow).
//
// We cannot see the body of ql::Lnrat here and it has no vendored
// double-double equivalent in the shim's include set. Per Rule R4, this is
// a genuine C3 gap — the operation itself is missing from quad::ddfun and
// we have no visible source to reimplement it from without risking a
// silent narrowing at the call site.
//
// UNCLASSIFIED: ql::Lnrat<quad::ddfun::ddouble, TMass, TScale>(a, b)
// Rule 2/C3/C9 unclear because: ql::Lnrat is a target-application function
// template whose body is not visible in the shim's closed include set (C1),
// no vendored quad::ddfun equivalent exists, and providing a ddouble
// overload that forwards to the primary would narrow its ddouble arguments
// to double and break the chain-internal ddouble contract (C9). A human
// must supply a ddouble implementation of Lnrat (or expose its body) before
// this region can be promoted.
// Human review needed before this shim can compile.
#error "DD Chain Integrator: ql::Lnrat<ddouble,...> requires manual classification"