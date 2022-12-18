cmake_minimum_required(VERSION 3.10.0)
project(layers)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall")

# Найдет все файлы cpp и запишет в CPPS
file(GLOB CPPS "*cpp")

add_library(${PROJECT_NAME} STATIC ${CPPS})
add_subdirectory(paramgetter)

target_link_libraries(${PROJECT_NAME} PUBLIC paramgetter)
