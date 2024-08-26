cmake_minimum_required(VERSION 3.18)
project(Group4HSDL_cpu LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")

file(GLOB_RECURSE SRC_FILES 
    "include/*.h"
    "./*.cpp"
)

enable_language(CUDA)
find_package(CUDA REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

add_library(${PROJECT_NAME} ${SRC_FILES})

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
message("Include directories: ${CMAKE_CURRENT_SOURCE_DIR}/include")

# 链接目录
# -lcpp_redis -ltacopie -pthread
target_link_libraries(${PROJECT_NAME} pthread)

aux_source_directory(src DIR_SRCS)

add_executable(test ${DIR_SRCS})

target_include_directories(test PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(test ${PROJECT_NAME} ${CUDA_LIBRARIES})

#set_target_properties(${EXECUTABLE_NAME_GPU} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
