cmake_minimum_required(VERSION 3.18)
project(B12_spike LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Kokkos REQUIRED)

add_executable(B12_spike src/micro_driver.cpp)

target_include_directories(B12_spike PRIVATE
    # Tracked instrumentation library
    ${CMAKE_SOURCE_DIR}/../../third_party/tracked/include

    # Vendored qcdloop headers (shared tree, see runs/B13/qcdloop_headers)
    ${CMAKE_SOURCE_DIR}/../B13/qcdloop_headers
    ${CMAKE_SOURCE_DIR}/../B13/qcdloop_headers/box

    # Local: ql_tracked_interop.hpp lives next to the driver
    ${CMAKE_SOURCE_DIR}/src

    # Kokkos install (Serial-only build for the spike)
    $ENV{HOME}/kokkos-install/include
)

target_link_libraries(B12_spike PRIVATE Kokkos::kokkos)
