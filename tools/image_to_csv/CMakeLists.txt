cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME ImageToCsv)

project(${PROJECT_NAME})

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

message("Building ${CMAKE_BUILD_TYPE} configuration...")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB
  CPP_SOURCES
  "${PROJECT_SOURCE_DIR}/src/*.cpp"
  "${PROJECT_SOURCE_DIR}/../common/*.cpp")

add_executable(imagetocsv ${CPP_SOURCES})

target_include_directories(
  imagetocsv
  PRIVATE
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/../../common"
)

set(COMPILER_FLAGS -Wextra -Wall)
set(DEBUG_FLAGS ${COMPILER_FLAGS} -g)
set(RELEASE_FLAGS ${COMPILER_FLAGS} -O3)

target_compile_options(imagetocsv PUBLIC "$<$<CONFIG:DEBUG>:${DEBUG_FLAGS}>")
