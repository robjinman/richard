cmake_minimum_required(VERSION 3.27)

set(PROJECT_NAME Richard)
set(RICHARD_VERSION_MAJOR 0)
set(RICHARD_VERSION_MINOR 1)

project(${PROJECT_NAME} VERSION "${VERSION_STRING}")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

message("System name is ${CMAKE_SYSTEM_NAME}")
message("Building ${CMAKE_BUILD_TYPE} configuration...")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

include("richard_common")

add_subdirectory(cpputils)
add_subdirectory(librichard)
add_subdirectory(richardcli)
add_subdirectory(tools)

set(CPACK_PACKAGE_VENDOR "Rob Jinman")
set(CPACK_PACKAGE_CONTACT "jinmanr@gmail.com")
include(CPack)
