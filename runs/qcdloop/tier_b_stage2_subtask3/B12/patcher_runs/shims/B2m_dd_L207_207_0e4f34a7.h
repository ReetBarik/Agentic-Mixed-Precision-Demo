#pragma once
// SOURCE_HASH: 0e4f34a74044771bcb278a326d3375f1dad1e379492a791dab1c75eec18cc3c4
#include <dd_math.hpp>
#include <dd_complex.hpp>

// Region: box/B2m.h:207
//   const TOutput ga43m   = TOutput(+p3sq + m3sq - m4sq) - root;
//
// After promotion, TOutput resolves to quad::ddfun::ddouble (chain-internal
// scalar, per C9/C10). All operands (p3sq, m3sq, m4sq, root) arrive as
// ddouble via the boundary patch / carrier widening. The expression uses
// only unary+, binary+, binary-, and an explicit TOutput(...) conversion
// from a ddouble rvalue to ddouble — all of which are already provided by
// the vendored ddouble type (Rule 2). No missing operators, no named
// constants, no library-owned templates to specialize.
//
// Nothing to emit beyond the vendored includes.