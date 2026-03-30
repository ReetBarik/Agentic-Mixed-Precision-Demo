cmake_minimum_required(VERSION 3.16)
project(AgenticMixedPrecisionDemo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Kokkos REQUIRED)

# Driver ids: each builds driver/<id>_driver.cc -> <id>_driver
# When adding a target, append the id here and add an entry to targets.json.
# Rationale: docs/DESIGN.md
set(DEMO_DRIVERS ddilog)

foreach(_driver_id IN LISTS DEMO_DRIVERS)
  set(_exe "${_driver_id}_driver")
  set(_src "driver/${_driver_id}_driver.cc")
  add_executable(${_exe} ${_src})
  target_include_directories(${_exe} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_link_libraries(${_exe} PRIVATE Kokkos::kokkos)
endforeach()
