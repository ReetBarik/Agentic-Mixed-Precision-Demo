cmake_minimum_required(VERSION 3.18)
project(qcdloop_app LANGUAGES CXX)

# Kokkos' own CMake config bumps this to C++20 where its headers require it; 17
# is the floor the qcdloop sources are written to.
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Kokkos REQUIRED)

# Parameterized so the Validator can build each of its three runs against the
# right tree with one configure per binary:
#   QL_MODE=vanilla  QL_HEADERS=<working-tree headers>          (double)
#   QL_MODE=dd       QL_HEADERS=<qcdloop@ddfun_enabled/src/qcdloop>  (double-double)
set(QL_HEADERS "" CACHE PATH "qcdloop header dir (contains boxGPU.h + box/)")
set(QL_MODE "vanilla" CACHE STRING "vanilla | dd")

if(NOT QL_HEADERS)
    message(FATAL_ERROR "QL_HEADERS must be set to the qcdloop header dir")
endif()

set(_recipes_dir ${CMAKE_CURRENT_SOURCE_DIR}/../src)

# Repo-vendored extended-precision headers (quad::ffun::ffloat / quad::ddfun::ddouble)
# that regional ff/dd shims #include.  Always available regardless of the candidate
# header tree, so a regional shim installed by the Patcher resolves <ff_math.hpp>.
set(_vendored_include ${CMAKE_CURRENT_SOURCE_DIR}/../../../third_party/include)

if(QL_MODE STREQUAL "dd")
    add_executable(boxGPU_app ${_recipes_dir}/boxGPU_dd.cpp)  # #defines USE_DD_COMPLEX itself
elseif(QL_MODE STREQUAL "vanilla")
    add_executable(boxGPU_app ${_recipes_dir}/boxGPU_vanilla.cpp)
else()
    message(FATAL_ERROR "QL_MODE must be 'vanilla' or 'dd' (got '${QL_MODE}')")
endif()

target_include_directories(boxGPU_app PRIVATE
    ${QL_HEADERS}
    ${QL_HEADERS}/box
    ${_recipes_dir}       # boxGPU_app_recipes.hpp
    ${_vendored_include}  # ff_math.hpp / ff_complex.hpp for regional ff/dd shims
)
target_link_libraries(boxGPU_app PRIVATE Kokkos::kokkos)
target_compile_options(boxGPU_app PRIVATE -w)  # qcdloop headers are warning-noisy
