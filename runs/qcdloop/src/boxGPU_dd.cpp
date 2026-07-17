// runs/qcdloop/src/boxGPU_dd.cpp
//
// Validator DD ground-truth oracle: the qcdloop box integrals in double-double
// precision (~30-31 digits), built with USE_DD_COMPLEX against a
// qcdloop@ddfun_enabled tree (which supplies kokkosMaths_dd.h + the ql::ddfun
// DD headers via kokkosMaths_wrapper.h).  Same mt19937(12345) recipes as the
// vanilla driver (shared boxGPU_app_recipes.hpp) so it evaluates bit-identical
// input points — that is the invariant that makes precise-digits meaningful.
//
// Prints each coeff component as "hi|lo" hex (the two doubles of the DD value)
// so the Validator reconstructs the exact reference in Python Decimal without
// any libquadmath dependency.

#define USE_DD_COMPLEX             // must precede boxGPU.h -> kokkosMaths_wrapper.h
#include <Kokkos_Core.hpp>
#include <string>

#include "boxGPU.h"                 // ddfun_enabled headers; pulls ql::ddfun types
#include "boxGPU_app_recipes.hpp"   // ql_app::dhex, ql_app::run_app

namespace ql_app {
// Component printer: two hex tokens (hi|lo) = the DD value's two doubles.
struct DDPrinter {
    static void emit(std::string& out, const ql::ddfun::ddouble& v) {
        out += dhex(v.hi);
        out += '|';
        out += dhex(v.lo);
    }
};
}  // namespace ql_app

int main(int argc, char* argv[]) {
    return ql_app::run_app<ql::ddfun::ddcomplex, ql::ddfun::ddouble,
                           ql::ddfun::ddouble, ql_app::DDPrinter>(argc, argv);
}
