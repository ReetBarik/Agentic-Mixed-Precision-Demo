#pragma once
// SOURCE_HASH: 93d1a7ed191eae357cfb3b56d2fc367d76d68f8bbfc7c961616150db18eeebbe
#include <dd_math.hpp>
#include <dd_complex.hpp>

// Region: box/B2m.h:300 -- ga34pm1 = TOutput(-p3sq + m4sq - m3sq) + root;
//
// The region uses `TOutput(...)` as a conversion cast on a floating-point
// expression (-p3sq + m4sq - m3sq). Under promotion, the operands are
// ddouble, so the cast target must also be ddouble to preserve precision
// across the chain (C9/C10 -- ga34pm1 is a carrier/produced value that must
// stay ddouble end-to-end). The vendored ddouble type already supplies
// unary -, binary +, and binary - for ddouble op ddouble, so no arithmetic
// overloads need to be added.
//
// No named constants appear in this region -- Rule 5 / Rule R3 cascade N/A.
// No qualified math calls appear -- C3 bridge N/A.
// No container-of-float constructions -- Rule 3 N/A.

// Rule 2 (floating-point result stays extended) + C9 (chain-internal value
// stays ddouble): route `TOutput(x)` for a ddouble operand to a ddouble
// identity so the cast introduces no rounding and the widened value flows
// on to the next chain link intact. Provided as a free function template
// alias-style helper is not applicable here because TOutput is used as a
// functional cast on a concrete type at the region call site; the boundary
// patch has retyped the surrounding scalar to ddouble, so TOutput is
// already ddouble at this instantiation and the vendored ddouble(ddouble)
// copy constructor suffices. No additional overload required.