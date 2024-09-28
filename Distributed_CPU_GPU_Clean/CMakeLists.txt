cmake_minimum_required(VERSION 3.17)

project(Test LANGUAGES CXX CUDA)

find_package(CUDA REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --generate-code=arch=compute_86,code=sm_86 -rdc=true")

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build/bin)
include_directories(include  ${CUDA_INCLUDE_DIRS} )
include_directories("/home/mdnd/boost_1_85_0" )
aux_source_directory(src DIR_SRCS)

add_definitions(-w)

add_executable(${PROJECT_NAME} ${DIR_SRCS})

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})

set_target_properties(${PROJECT_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)