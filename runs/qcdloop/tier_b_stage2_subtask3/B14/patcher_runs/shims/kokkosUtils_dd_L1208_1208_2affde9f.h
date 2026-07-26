#pragma once
// SOURCE_HASH: 2affde9f3c316297071e661afa71fef5c254bab12115da63f708d698282db6cd
#include <dd_math.hpp>
#include <dd_complex.hpp>

// Region: res[0] = -invopr * invopr / rat;
// All operands (invopr, rat) are promoted to quad::ddfun::ddouble by the
// deterministic boundary patch. The vendored dd_math.hpp already supplies:
//   - unary operator- on ddouble
//   - operator* (ddouble, ddouble)
//   - operator/ (ddouble, ddouble)
//   - subscripted assignment into a ddouble-typed lvalue (res[0])
// No additional operators, overloads, specializations, or named constants
// are required by this region.
//
// Rule 2: floating-point result stays ddouble end-to-end.
// C9/C10: any value written into a carrier (res[0]) remains ddouble; the
//        boundary patch handles demotion at the chain edge, not this shim.