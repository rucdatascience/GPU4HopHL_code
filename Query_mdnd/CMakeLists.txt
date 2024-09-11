cmake_minimum_required(VERSION 3.17)
project(Group4HSDL_cpu LANGUAGES CXX CUDA)

# 设置 C++ 和 CUDA 的标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 导出编译命令
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# 启用 CUDA 语言支持
enable_language(CUDA)

# 查找 CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# 设置编译标志
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# 搜索头文件和源文件
file(GLOB_RECURSE SRC_FILES 
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cuh"
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
)


# 添加包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CUDAToolkit_INCLUDE_DIRS})
include_directories(/home/mdnd/boost_1_85_0)

# 创建库
add_library(${PROJECT_NAME} ${SRC_FILES})

target_include_directories(${PROJECT_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
# 链接 pthread 和 cudart 库
target_link_libraries(${PROJECT_NAME} pthread CUDA::cudart ${CUDAToolkit_LIBRARIES})

add_definitions(-w)

# 添加可执行文件
add_executable(test main.cu)

# 确保 CUDA 库和头文件可用
target_include_directories(test PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(test ${PROJECT_NAME} CUDA::cudart)

# 确保启用 CUDA 的分离编译功能
set_target_properties(test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
