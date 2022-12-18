cmake_minimum_required(VERSION 3.1)
project(paramgetter)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -Wall")


add_library(${PROJECT_NAME} STATIC paramgetter.cpp)

set(HDF5DIR "/home/galahad/Documents/libs/HDF5")
target_include_directories(${PROJECT_NAME} PRIVATE ${HDF5DIR}/include)
target_link_directories(${PROJECT_NAME} PUBLIC ${HDF5DIR}/lib)
target_link_libraries(${PROJECT_NAME} PUBLIC libhdf5.so libhdf5_cpp.so)