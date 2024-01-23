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

file(GLOB SHADER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/shaders/*.glsl")
set(SHADER_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/shaders")

set(SHADER_BINARIES "")
foreach(SHADER_SOURCE ${SHADER_SOURCES})
  get_filename_component(SHADER_FILENAME ${SHADER_SOURCE} NAME)
  string(REGEX REPLACE "[.]glsl$" ".spv" SHADER_BINARY_NAME ${SHADER_FILENAME})
  set(SHADER_BINARY "${SHADER_BINARY_DIR}/${SHADER_BINARY_NAME}")
  list(APPEND SHADER_BINARIES ${SHADER_BINARY})
  add_custom_command(
    OUTPUT ${SHADER_BINARY}
    COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_BINARY_DIR}"
    COMMAND ${glslc_executable} -fshader-stage=compute ${SHADER_SOURCE} -o ${SHADER_BINARY}
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
    MAIN_DEPENDENCY ${SHADER_SOURCE}
  )
endforeach()
add_custom_target(testShaders DEPENDS ${SHADER_BINARIES})

add_dependencies(unitTests testShaders)
