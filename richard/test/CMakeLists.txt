cmake_minimum_required(VERSION 3.5)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

FetchContent_MakeAvailable(googletest)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../src")
file(GLOB_RECURSE SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(unitTests ${SRCS})

target_link_libraries(unitTests ${RICHARD_LIB_NAME} gtest_main gmock_main)
target_compile_options(unitTests PUBLIC -Wall -g -DDEBUG)
