cmake_minimum_required(VERSION 3.18)
project(BIN3_spike LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Kokkos REQUIRED)

add_executable(BIN3_spike src/micro_driver.cpp)

target_include_directories(BIN3_spike PRIVATE
    # Tracked instrumentation library
    ${CMAKE_SOURCE_DIR}/../../third_party/tracked/include

    # Un-pruned qcdloop headers (3-/4-mass boxes restored; see
    # runs/qcdloop_headers_full). The C8 library patch is applied to this tree
    # at build time and reset afterward.
    ${CMAKE_SOURCE_DIR}/../qcdloop_headers_full
    ${CMAKE_SOURCE_DIR}/../qcdloop_headers_full/box

    # Local: ql_tracked_interop.hpp lives next to the driver
    ${CMAKE_SOURCE_DIR}/src

    # Kokkos install (Serial-only build for the spike)
    $ENV{HOME}/kokkos-install/include
)

target_link_libraries(BIN3_spike PRIVATE Kokkos::kokkos)
