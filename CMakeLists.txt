cmake_minimum_required(VERSION 3.1)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -mavx2 -mfma -fopenmp")

project(unet VERSION 0.01 LANGUAGES C CXX)

set(HDF5DIR "/home/galahad/Documents/libs/HDF5")
file(GLOB LAYERS_CPPS_CU "src/layers/*cpp" "src/layers/*cpp")
set(LODEPNGDIR "include/lodepng")
file(GLOB DATAH_CPPS "src/datahandler/*cpp")

# Библиотека с функциями активации
add_library(activation src/activation/Activations.cpp)

# Библиотека для получения параметров нейронной сети
add_library(parametergetter STATIC src/layers/paramgetter/paramgetter.cpp)
target_include_directories(parametergetter PRIVATE ${HDF5DIR}/include)
target_link_directories(parametergetter PUBLIC ${HDF5DIR}/lib)
target_link_libraries(parametergetter PRIVATE libhdf5.so libhdf5_cpp.so)

# Библиотека матричного умножения
add_library(matmul STATIC src/layers/gemm/matmul/matmul.cpp)
add_library(transform STATIC src/layers/gemm/transformation/transform.cpp)

# Библиотека со слоями нейронной сети
add_library(layers STATIC ${LAYERS_CPPS_CU})
target_link_libraries(layers PUBLIC parametergetter matmul transform)
target_include_directories(layers PUBLIC src/layers/gemm/matmul/)

# Библиотека для загрузки изображения
add_library(lodepng STATIC ${LODEPNGDIR}/lodepng.cpp)

# Библиотека для преобразования изображения
add_library(datahandler STATIC ${DATAH_CPPS})
target_include_directories(datahandler PRIVATE ${LODEPNGDIR}/lodepng.h)
target_link_libraries(datahandler PRIVATE lodepng)

# Основной исполняемый файл
add_executable(${PROJECT_NAME} main.cpp)

target_link_libraries(${PROJECT_NAME} datahandler layers activation)
