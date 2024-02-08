cmake_minimum_required(VERSION 3.24)

set(PROJECT_NAME Richard)
set(RICHARD_VERSION_MAJOR 0)
set(RICHARD_VERSION_MINOR 1)

project(${PROJECT_NAME})

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message("System name is ${CMAKE_SYSTEM_NAME}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

message("Building ${CMAKE_BUILD_TYPE} configuration...")

include(ExternalProject)
set_property(DIRECTORY PROPERTY EP_BASE ${PROJECT_SOURCE_DIR}/dependencies/${CMAKE_BUILD_TYPE})

include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${PROJECT_SOURCE_DIR}/dependencies/${CMAKE_BUILD_TYPE})

include("richard_common")

add_subdirectory(cpputils)
add_subdirectory(librichard)
add_subdirectory(richardcli)
if (BUILD_TOOLS)
  add_subdirectory(tools)
endif()
