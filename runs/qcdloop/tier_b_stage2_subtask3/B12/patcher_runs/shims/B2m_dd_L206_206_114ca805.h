#pragma once
// SOURCE_HASH: 114ca8056f641ab70bca0d0049f913f06d90ba76a10c14d86b576f4a7f5fe773
#include <dd_math.hpp>
#include <dd_complex.hpp>

// Region: box/B2m.h:206
//   const TOutput ga43pm1 = TOutput(-p3sq + m3sq - m4sq) + root;
//
// After promotion, TOutput is quad::ddfun::ddouble (or ddcomplex when the
// caller instantiates the complex chain link). p3sq, m3sq, m4sq, root are
// promoted to the extended scalar by the boundary patch.
//
// The vendored dd_math.hpp / dd_complex.hpp already provide:
//   - unary operator- on ddouble / ddcomplex
//   - ddouble +/- ddouble, ddcomplex +/- ddcomplex
//   - mixed ddcomplex +/- ddouble
//   - the explicit conversion constructor TOutput(x) for both scalar and
//     complex targets from the promoted operands
//
// No named constants, no library-owned templates, no qualified math calls,
// and no missing operators are referenced by this single-statement region.
// Nothing further to shim.