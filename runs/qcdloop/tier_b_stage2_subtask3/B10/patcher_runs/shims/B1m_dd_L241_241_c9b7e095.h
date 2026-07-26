#pragma once
// SOURCE_HASH: c9b7e0952570dbb9b1828c5ba27ac7999cbe513cc1fc1d35d69f4f7bacfbc1d9
#include <dd_math.hpp>
#include <dd_complex.hpp>

// Region: res(i,0) = dilog4 - dilog5;
// The only operation this region performs is a subtraction of two ddouble
// operands (dilog4, dilog5 — carrier/chain-internal values already widened
// to quad::ddfun::ddouble by the boundary/emission layer per C10) and an
// assignment into res(i,0) (Rule R1: boundary patch handles the demotion).
//
// The vendored dd_math.hpp already provides operator-(ddouble, ddouble)
// returning ddouble (Rule 2 / C9: stays extended end-to-end), and the
// assignment sink is boundary-managed. No named constants, no qualified
// math calls, no missing operators, no complex containers — nothing for
// the shim to add.