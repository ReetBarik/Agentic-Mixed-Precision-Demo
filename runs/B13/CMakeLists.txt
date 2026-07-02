cmake_minimum_required(VERSION 3.18)
project(B13_spike LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Kokkos REQUIRED)

add_executable(B13_spike src/micro_driver.cpp)

target_include_directories(B13_spike PRIVATE
    # Tracked instrumentation library
    ${CMAKE_SOURCE_DIR}/../../third_party/tracked/include

    # Vendored qcdloop headers (see qcdloop_headers/README.md for upstream sha)
    ${CMAKE_SOURCE_DIR}/qcdloop_headers
    ${CMAKE_SOURCE_DIR}/qcdloop_headers/box

    # Local: ql_tracked_interop.hpp lives next to the driver
    ${CMAKE_SOURCE_DIR}/src

    # Kokkos install (Serial-only build for the spike)
    $ENV{HOME}/kokkos-install/include
)

target_link_libraries(B13_spike PRIVATE Kokkos::kokkos)
