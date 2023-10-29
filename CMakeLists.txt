cmake_minimum_required(VERSION 3.10.2)
project("gpgpu_with_opencl")

set(CMAKE_CXX_STANDARD 17)

find_package(OpenCL REQUIRED)
find_package(GTest REQUIRED)
enable_testing()

include_directories(
    include/
    ${OpenCL_INCLUDE_DIR}
    ${GTEST_INCLUDE_DIRS}
)

file(GLOB lib_src
    "src/*.cc"
)

add_library(${PROJECT_NAME} STATIC
    ${lib_src}
)

target_link_libraries(${PROJECT_NAME}
    ${OpenCL_LIBRARIES}
)


file(GLOB tests_src
    "tests/*.cc"
)

add_executable(${PROJECT_NAME}_tests ${tests_src})
target_link_libraries(${PROJECT_NAME}_tests PRIVATE ${GTEST_LIBRARIES} ${PROJECT_NAME})

