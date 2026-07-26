#pragma once
// SOURCE_HASH: a9bbfd757234bf6725a807c8754dcdd636ac2b425b5362f7dc6a06b53487eee3
#include <dd_math.hpp>
#include <dd_complex.hpp>

// Region: box/B2m.h:305
//   ga43pm1 = TOutput(-p3sq + m3sq - m4sq) + root;
//
// The region uses TOutput(...) as a constructor-style cast around a real
// expression. Under chain promotion, the inner expression is ddouble and
// TOutput must stay ddouble to satisfy C9 (chain-internal contract) — the
// result flows into `ga43pm1` (a carrier per C10) and must NOT narrow.
//
// The vendored quad::ddfun::ddouble already provides all arithmetic
// operators (unary -, binary +, binary -) for ddouble op ddouble, so no
// additional operator overloads are required.
//
// No named constants are referenced by this region → no Rule 5 wrappers.
// No qualified math calls are made → no C3 bridges.
// No library class-template specializations are needed → no C5/C7 entries.