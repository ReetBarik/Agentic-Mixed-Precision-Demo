#pragma once
// SOURCE_HASH: b21f2cce8797f31b0f2860185a98e183618926ac0e1c8d5dffbe3833a4e7ceb9
#include <dd_math.hpp>
#include <dd_complex.hpp>

// Region: box/B1m.h:227
//   const TOutput fac = TOutput(si * tabar - mp2sq * m4sqbar);
//
// All operands (si, tabar, mp2sq, m4sqbar) are promoted to
// quad::ddfun::ddouble (or ddcomplex) by the boundary patch. The vendored
// headers already provide the full arithmetic operator set
// (ddouble op ddouble, ddouble op double, double op ddouble, and the
// analogous ddcomplex overloads including mixed ddcomplex/ddouble), plus
// the ddcomplex(ddouble) / ddcomplex(ddcomplex) conversion constructors
// used by the explicit TOutput(...) cast.
//
// No named constants, no qualified math calls, no missing operators.
// Nothing further to emit for this line.