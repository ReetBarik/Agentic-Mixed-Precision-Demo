#pragma once
// SOURCE_HASH: 934d6c05ce6f8cfb197028950b49aaac7fc382841256ed4c74b7e6b05d942f83
#include <dd_math.hpp>
#include <dd_complex.hpp>

// Region: box/B1m.h:240
//   res(i,1) = wlog2mu + wlog4mu - wlogsmu - wlogtmu;
//
// The region uses only addition/subtraction on promoted ddouble values
// (wlog2mu, wlog4mu, wlogsmu, wlogtmu). The vendored dd_math.hpp already
// supplies ddouble + ddouble and ddouble - ddouble. The assignment target
// res(i,1) is a carrier written by this chain link (C10) — the boundary/
// emission layer widens its storage; the sum stays ddouble on assignment.
//
// No named constants, no qualified math calls, no complex containers, and
// no missing operators are referenced by this region, so no additional
// overloads, specializations, or constant wrappers are required.
//
// Rule 2: floating-point sum stays in ddouble (vendored operators used).
// C9/C10: result flows through the widened carrier res as ddouble.