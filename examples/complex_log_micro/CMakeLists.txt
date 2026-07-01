cmake_minimum_required(VERSION 3.18)
project(complex_log_micro LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)  # Kokkos 5.1.0 requires C++20
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Kokkos REQUIRED)

add_executable(complex_log_micro complex_log_micro.cpp)

# Tracked headers live two levels up (examples/complex_log_micro/ → repo root).
target_include_directories(complex_log_micro PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

target_link_libraries(complex_log_micro PRIVATE Kokkos::kokkos)
