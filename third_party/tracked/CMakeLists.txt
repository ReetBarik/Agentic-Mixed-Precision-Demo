cmake_minimum_required(VERSION 3.16)

project(tracked-instrumentation LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Debug CACHE STRING "Choose the type of build." FORCE)
endif()

# ---- Tracked<T> header-only library -----------------------------------------

add_library(tracked INTERFACE)
add_library(tracked::tracked ALIAS tracked)
target_include_directories(tracked INTERFACE
  $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_compile_features(tracked INTERFACE cxx_std_17)

# ---- Tests ------------------------------------------------------------------

include(FetchContent)
FetchContent_Declare(
  Catch2
  GIT_REPOSITORY https://github.com/catchorg/Catch2.git
  GIT_TAG        v3.4.0
)
FetchContent_MakeAvailable(Catch2)

enable_testing()
list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
include(Catch)

foreach(test_name
    test_basic_ops
    test_cancellation
    test_naive_variance
    test_log_sum_exp
    test_kahan_vs_naive
    test_trig
    test_complex_basic
    test_complex_sqrt
    test_complex_log
    test_complex_div
    test_provenance_v03
    test_literals
    test_operand_ids)
  add_executable(${test_name} tests/tracked/${test_name}.cpp)
  target_link_libraries(${test_name} PRIVATE tracked::tracked Catch2::Catch2WithMain)
  catch_discover_tests(${test_name})
endforeach()
