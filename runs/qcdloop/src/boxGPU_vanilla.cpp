// runs/qcdloop/src/boxGPU_vanilla.cpp
//
// Validator "current baseline" / "candidate" driver: the qcdloop box integrals
// in plain double precision (Kokkos::complex<double>), built against the working
// tree (runs/qcdloop_headers_full, or a patched copy for a candidate).  The
// de-tracked twin of boxGPU_tracked.cpp — same mt19937(12345) recipes via the
// shared boxGPU_app_recipes.hpp, no Tracked type / scopes / journal.  Prints
// each dispatched sample's three coeffs as hex (see the recipe header).

#include <Kokkos_Core.hpp>
#include <Kokkos_Complex.hpp>
#include <string>

#include "boxGPU.h"                 // working-tree qcdloop headers (native double)
#include "boxGPU_app_recipes.hpp"   // ql_app::dhex, ql_app::run_app

namespace ql_app {
// Component printer: one hex token = the double's IEEE-754 bit pattern.
struct VanillaPrinter {
    static void emit(std::string& out, double v) { out += dhex(v); }
};
}  // namespace ql_app

int main(int argc, char* argv[]) {
    return ql_app::run_app<Kokkos::complex<double>, double, double,
                           ql_app::VanillaPrinter>(argc, argv);
}
