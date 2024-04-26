cmake_minimum_required(VERSION 3.24)

#FetchContent_Declare(googletest
#  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
#  DOWNLOAD_EXTRACT_TIMESTAMP ON
#)

#FetchContent_MakeAvailable(googletest)

find_package(GTest REQUIRED)

file(GLOB_RECURSE SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(unitTests ${SRCS})

target_compile_definitions(unitTests PUBLIC DEBUG)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(COMPILE_FLAGS -g -Wextra -Wall)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(COMPILE_FLAGS /W4)
endif()

get_target_property(potato GTest::gtest_main INCLUDE_DIRECTORIES)
message("Hello ${potato}")

target_link_libraries(unitTests ${RICHARD_LIB_TARGET} GTest::gtest_main GTest::gmock_main)
target_compile_options(unitTests PRIVATE ${COMPILE_FLAGS})

file(GLOB SHADER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.glsl")

compile_shaders(
  testShaders
  "${SHADER_SOURCES}"
  "${PROJECT_BINARY_DIR}/test_shaders"
)

add_dependencies(unitTests testShaders)
