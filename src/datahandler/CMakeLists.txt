cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall")
project(datahandler)

set(LODEPNGDIR "../../include/lodepng")
# Найдет все файлы cpp и запишет в CPPS
file(GLOB CPPS "*cpp")

add_library(lodepng STATIC ${LODEPNGDIR}/lodepng.cpp)
add_library(${PROJECT_NAME} STATIC ${CPPS})

target_include_directories(${PROJECT_NAME} PRIVATE ${LODEPNGDIR}/lodepng.h)
target_link_libraries(${PROJECT_NAME} PRIVATE lodepng)