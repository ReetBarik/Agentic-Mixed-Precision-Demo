cmake_minimum_required(VERSION 3.18)
project(micro_driver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD {cxx_standard})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

{find_package_lines}

add_executable(micro_driver src/micro_driver.cpp)
target_include_directories(micro_driver PRIVATE
    {tracked_include_dir}
    {extra_include_dirs}
)
target_link_libraries(micro_driver PRIVATE {link_libs})
