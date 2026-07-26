#pragma once
// SOURCE_HASH: 39d61de953a89aca064c666993969c96b47df1f1a4edddb1d5e9c2948805f9a9
#include <dd_math.hpp>
#include <dd_complex.hpp>

// Region: box/B2m.h:301
//   ga34m = TOutput(+p3sq + m4sq - m3sq) - root;
//
// After promotion, TOutput is quad::ddfun::ddouble and the operands p3sq,
// m4sq, m3sq, root are all ddouble (chain-internal, C9/C10). The vendored
// dd_math.hpp already provides unary+, binary +, binary -, and the
// TOutput(x) explicit cast reduces to the ddouble copy-ctor. No additional
// operators, overloads, specializations, or named constants are required.