cmake_minimum_required(VERSION 3.1)
project(cpp_unet)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall")

add_library(activation src/activation/Activations.cpp)

add_executable(${PROJECT_NAME} main.cpp)
add_subdirectory(src/datahandler)
add_subdirectory(src/layers)

target_link_libraries(${PROJECT_NAME} datahandler layers activation)
