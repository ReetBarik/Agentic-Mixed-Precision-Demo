#include <Kokkos_Core.hpp>

#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>

#include "kokkosUtils.h"

using TOutput = Kokkos::complex<double>;
using TMass = double;
using TScale = double;

int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        if (argc < 2) {
            std::cerr << "usage: " << argv[0] << " <batch_size> [output.csv] [seed]\n";
            Kokkos::finalize();
            return 1;
        }

        const std::size_t batch_size = static_cast<std::size_t>(std::stoull(argv[1]));
        const std::string out_path = (argc >= 3) ? argv[2] : "ddilog_out.csv";
        std::uint64_t seed = 1;
        if (argc >= 4) {
            seed = static_cast<std::uint64_t>(std::stoull(argv[3]));
        }

        Kokkos::View<TMass*> x_d("x", batch_size);
        Kokkos::View<TMass*> y_d("y", batch_size);

        auto x_h = Kokkos::create_mirror_view(x_d);
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<TMass> dist(-4.0, 4.0);
        for (std::size_t i = 0; i < batch_size; ++i) {
            x_h(i) = dist(rng);
        }
        Kokkos::deep_copy(x_d, x_h);

        Kokkos::parallel_for(
            "ddilog_batch",
            Kokkos::RangePolicy<Kokkos::IndexType<std::size_t>>(0, batch_size),
            KOKKOS_LAMBDA(std::size_t i) {
                y_d(i) = ql::ddilog<TOutput, TMass, TScale>(x_d(i));
            });
        Kokkos::fence();

        auto y_h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y_d);

        std::ofstream out(out_path);
        if (!out) {
            std::cerr << "failed to open " << out_path << '\n';
            Kokkos::finalize();
            return 1;
        }
        out << "id,real hex\n";
        out << "# target_id=ddilog seed=" << seed << " batch_size=" << batch_size
            << " x_min=-4.0 x_max=4.0\n";
        for (std::size_t i = 0; i < batch_size; ++i) {
            out << i << ',';
            ql::printDoubleBits(y_h(i), out);
            out << '\n';
        }
    }
    Kokkos::finalize();
    return 0;
}
