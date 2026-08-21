cmake_minimum_required(VERSION 3.18)
project(qcdloop_tracked LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Kokkos REQUIRED)

add_executable(boxGPU_tracked src/boxGPU_tracked.cpp)

target_include_directories(boxGPU_tracked PRIVATE
    # Tracked instrumentation library
    ${CMAKE_SOURCE_DIR}/../../third_party/tracked/include

    # Un-pruned qcdloop headers (all box families B0m..B4m). The C8 library
    # patch (src/ql_tracked.patch) is applied to this tree at build time and
    # reset afterward; see runs/qcdloop/README or the consolidation spec.
    ${CMAKE_SOURCE_DIR}/../qcdloop_headers_full
    ${CMAKE_SOURCE_DIR}/../qcdloop_headers_full/box

    # Local: ql_tracked_interop.hpp lives next to the driver
    ${CMAKE_SOURCE_DIR}/src

    # Kokkos install (Serial-only build)
    $ENV{HOME}/kokkos-install/include
)

target_link_libraries(boxGPU_tracked PRIVATE Kokkos::kokkos)

# ---------------------------------------------------------------------------
# Build-provenance for journal_meta.json, injected as compile definitions so
# the driver can emit them at the end of main(). Computed at configure time
# from the committed shim/patch and the repo git state.
# ---------------------------------------------------------------------------

# sha256 of the interop shim and the C8 patch. The patch may be absent during
# the diagnostic-capture build (it is derived from that build's output); guard
# accordingly so a pristine/unpatched configure still succeeds.
file(SHA256 ${CMAKE_SOURCE_DIR}/src/ql_tracked_interop.hpp QCDLOOP_SHIM_SHA)
if(EXISTS ${CMAKE_SOURCE_DIR}/src/ql_tracked.patch)
    file(SHA256 ${CMAKE_SOURCE_DIR}/src/ql_tracked.patch QCDLOOP_PATCH_SHA)
else()
    set(QCDLOOP_PATCH_SHA "unpatched")
endif()

# The SHAs above are baked in as compile definitions at configure time, and
# builds may skip configure when a cache exists (run_chunked.py build()) —
# without a configure dependency an edited shim/patch would recompile against
# the new headers while journal_meta.json still reports the OLD hashes.
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    ${CMAKE_SOURCE_DIR}/src/ql_tracked_interop.hpp)
if(EXISTS ${CMAKE_SOURCE_DIR}/src/ql_tracked.patch)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
        ${CMAKE_SOURCE_DIR}/src/ql_tracked.patch)
endif()

# tracked_version: git describe of the repo (Tracked is vendored in-tree).
execute_process(
    COMMAND git -C ${CMAKE_SOURCE_DIR} describe --tags --always --dirty
    OUTPUT_VARIABLE QCDLOOP_TRACKED_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if(NOT QCDLOOP_TRACKED_VERSION)
    set(QCDLOOP_TRACKED_VERSION "unknown")
endif()

# qcdloop_commit: the vendored upstream snapshot (ReetBarik/qcdloop@master).
# See runs/qcdloop_headers_full/README.md. Hardcoded because the headers are a
# static vendored snapshot, not a live submodule.
set(QCDLOOP_COMMIT "8de2089")

target_compile_definitions(boxGPU_tracked PRIVATE
    QCDLOOP_SHIM_SHA="${QCDLOOP_SHIM_SHA}"
    QCDLOOP_PATCH_SHA="${QCDLOOP_PATCH_SHA}"
    QCDLOOP_TRACKED_VERSION="${QCDLOOP_TRACKED_VERSION}"
    QCDLOOP_COMMIT="${QCDLOOP_COMMIT}"
)
