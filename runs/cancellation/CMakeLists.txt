cmake_minimum_required(VERSION 3.18)
project(micro_driver LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



add_executable(micro_driver src/micro_driver.cpp)
target_include_directories(micro_driver PRIVATE
    /home/rbarik/Agentic-Mixed-Precision-Demo/third_party/tracked/include
    
)
target_link_libraries(micro_driver PRIVATE )
