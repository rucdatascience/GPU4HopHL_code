cmake_minimum_required(VERSION 3.18)
project(Group4HSDL_cpu LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS 
        "-O3 -lpthread"             
)

file(GLOB_RECURSE SRC_FILES 
    "include/*.h"
    "./*.cpp"
)

add_library(${PROJECT_NAME} ${SRC_FILES})

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
message("Include directories: ${CMAKE_CURRENT_SOURCE_DIR}/include")

# 链接目录
# -lcpp_redis -ltacopie -pthread
target_link_libraries(${PROJECT_NAME} pthread)

add_executable(test ${CMAKE_CURRENT_SOURCE_DIR}/test.cpp)

# # Custom command to generate dependencies for test.cpp
# add_custom_command(
#     OUTPUT ${CMAKE_BINARY_DIR}/test.d
#     COMMAND ${CMAKE_CXX_COMPILER} -M ${CMAKE_CURRENT_SOURCE_DIR}/Group4HBPLL/test.cpp -I${CMAKE_CURRENT_SOURCE_DIR}/Group4HBPLL -I${CMAKE_CURRENT_SOURCE_DIR}/include > ${CMAKE_BINARY_DIR}/test.d
#     DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/Group4HBPLL/test.cpp
#     COMMENT "Generating dependencies for test.cpp"
# )

# # Custom target to execute the custom command
# add_custom_target(generate_dependencies
#     DEPENDS ${CMAKE_BINARY_DIR}/test.d
# )

# # Ensure the custom target runs before building the executable
# add_dependencies(test generate_dependencies)