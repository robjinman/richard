cmake_minimum_required(VERSION 3.24)

set(PROJECT_NAME Richard)

project(${PROJECT_NAME})

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

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
add_subdirectory(tools)
