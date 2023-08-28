cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME Richard)

project(${PROJECT_NAME})
set(RICHARD_LIB_NAME "chard")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

message("Building ${CMAKE_BUILD_TYPE} configuration...")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Boost 1.56 REQUIRED COMPONENTS program_options)

file(GLOB
  CPP_SOURCES
  "${PROJECT_SOURCE_DIR}/src/*.cpp"
  "${PROJECT_SOURCE_DIR}/../common/*.cpp")
list(REMOVE_ITEM CPP_SOURCES "${PROJECT_SOURCE_DIR}/src/main.cpp")

add_library(${RICHARD_LIB_NAME} ${CPP_SOURCES})

target_include_directories(
  ${RICHARD_LIB_NAME}
  PUBLIC
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/../common"
)

set(COMPILER_FLAGS -Wextra -Wall)
set(DEBUG_FLAGS ${COMPILER_FLAGS} -g)
set(RELEASE_FLAGS ${COMPILER_FLAGS} -O3)

target_compile_options(${RICHARD_LIB_NAME} PUBLIC "$<$<CONFIG:DEBUG>:${DEBUG_FLAGS}>")
target_link_libraries(${RICHARD_LIB_NAME} PUBLIC Boost::program_options)

add_executable(richard "${PROJECT_SOURCE_DIR}/src/main.cpp")
target_link_libraries(richard ${RICHARD_LIB_NAME})

add_subdirectory(test)
