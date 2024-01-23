cmake_minimum_required(VERSION 3.16)

set(PROJECT_NAME Richard)

project(${PROJECT_NAME})
set(RICHARD_LIB_NAME "librichard")

message("System name is ${CMAKE_SYSTEM_NAME}")

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

message("Building ${CMAKE_BUILD_TYPE} configuration...")

if (CPUPROF)
  message("CPU profiling ON")
  
  find_package(GOOGLE_PERFTOOLS REQUIRED)

  if (NOT DEFINED PROFILE_DURATION)
    set(PROFILE_DURATION -1)
  endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake")

find_package(Vulkan REQUIRED)
find_program(glslc_executable NAMES glslc HINTS Vulkan::glslc)

include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/dependencies/${CMAKE_BUILD_TYPE})

set(BOOST_INCLUDE_LIBRARIES program_options)
set(BOOST_ENABLE_CMAKE ON)

FetchContent_Declare(
  Boost
  URL https://github.com/boostorg/boost/releases/download/boost-1.84.0/boost-1.84.0.tar.xz
  URL_MD5 893b5203b862eb9bbd08553e24ff146a
)

FetchContent_MakeAvailable(Boost)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
)

FetchContent_MakeAvailable(googletest)

FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
)

FetchContent_MakeAvailable(json)

file(GLOB
  CPP_SOURCES
  "${PROJECT_SOURCE_DIR}/src/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/cpu/*.cpp"
  "${PROJECT_SOURCE_DIR}/src/gpu/*.cpp"
  "${PROJECT_SOURCE_DIR}/../common/*.cpp")
list(REMOVE_ITEM CPP_SOURCES "${PROJECT_SOURCE_DIR}/src/main.cpp")

add_library(${RICHARD_LIB_NAME} ${CPP_SOURCES})

target_include_directories(
  ${RICHARD_LIB_NAME}
  PUBLIC
    "${PROJECT_SOURCE_DIR}/src"
    "${PROJECT_SOURCE_DIR}/../common"
)

set(CPUPROF_COMPILE_FLAGS -g)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(COMPILE_FLAGS -Wextra -Wall)
  if (CPUPROF)
    set(COMPILE_FLAGS ${COMPILE_FLAGS} ${CPUPROF_COMPILE_FLAGS})
  endif()
  set(DEBUG_COMPILE_FLAGS ${COMPILE_FLAGS} -g)
  set(RELEASE_COMPILE_FLAGS ${COMPILE_FLAGS} -O3)

  set(CPUPROF_LINK_FLAGS -Wl,--no-as-needed -lprofiler -Wl,--as-needed)
  set(LINK_FLAGS)
  if (CPUPROF)
    set(LINK_FLAGS ${LINK_FLAGS} ${CPUPROF_LINK_FLAGS})
  endif()
  set(DEBUG_LINK_FLAGS ${LINK_FLAGS})
  set(RELEASE_LINK_FLAGS ${LINK_FLAGS})
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(COMPILE_FLAGS)
  set(DEBUG_COMPILE_FLAGS ${COMPILE_FLAGS})
  set(RELEASE_COMPILE_FLAGS ${COMPILE_FLAGS} /O2)
endif()

target_compile_options(${RICHARD_LIB_NAME}
  PUBLIC
    "$<$<CONFIG:DEBUG>:${DEBUG_COMPILE_FLAGS}>"
    "$<$<CONFIG:RELEASE>:${RELEASE_COMPILE_FLAGS}>"
)

target_link_libraries(${RICHARD_LIB_NAME}
  PRIVATE
    Vulkan::Vulkan
    shaderc
  PUBLIC
    nlohmann_json::nlohmann_json
    "$<$<CONFIG:DEBUG>:${DEBUG_LINK_FLAGS}>"
    "$<$<CONFIG:RELEASE>:${RELEASE_LINK_FLAGS}>"
)

file(GLOB SHADER_SOURCES "${PROJECT_SOURCE_DIR}/src/gpu/shaders/*.glsl")
set(SHADER_BINARY_DIR "${PROJECT_BINARY_DIR}/shaders")

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
add_custom_target(shaders DEPENDS ${SHADER_BINARIES})

add_dependencies(${RICHARD_LIB_NAME} shaders)

add_executable(richard "${PROJECT_SOURCE_DIR}/src/main.cpp")

target_link_libraries(richard
  ${RICHARD_LIB_NAME}
  Boost::program_options
)

add_subdirectory(test)

