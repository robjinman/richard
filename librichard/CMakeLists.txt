cmake_minimum_required(VERSION 3.24)

set(RICHARD_LIB_TARGET librichard)
set(RICHARD_LIB_TARGET ${RICHARD_LIB_TARGET} PARENT_SCOPE)

if (CPUPROF)
  message("CPU profiling ON")
  
  find_package(GOOGLE_PERFTOOLS REQUIRED)

  if (NOT DEFINED PROFILE_DURATION)
    set(PROFILE_DURATION -1)
  endif()
endif()

find_package(Vulkan REQUIRED)

include(FetchContent)
set(FETCHCONTENT_BASE_DIR ${CMAKE_SOURCE_DIR}/dependencies/${CMAKE_BUILD_TYPE})

set(BOOST_INCLUDE_LIBRARIES program_options)
set(BOOST_ENABLE_CMAKE ON)

FetchContent_Declare(Boost
  URL https://github.com/boostorg/boost/releases/download/boost-1.84.0/boost-1.84.0.tar.xz
  URL_MD5 893b5203b862eb9bbd08553e24ff146a
  DOWNLOAD_EXTRACT_TIMESTAMP ON
)

FetchContent_MakeAvailable(Boost)

FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz
  DOWNLOAD_EXTRACT_TIMESTAMP ON
)

FetchContent_MakeAvailable(googletest)

FetchContent_Declare(json
  URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz
  DOWNLOAD_EXTRACT_TIMESTAMP ON
)

FetchContent_MakeAvailable(json)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/src/version_in.hpp"
  "${CMAKE_CURRENT_BINARY_DIR}/include/richard/version.hpp"
)

file(GLOB
  CPP_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu/*.cpp"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/*.cpp"
)

add_library(${RICHARD_LIB_TARGET} ${CPP_SOURCES})

target_include_directories(${RICHARD_LIB_TARGET}
  PRIVATE
    "${CMAKE_CURRENT_BINARY_DIR}/include"
  PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
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
  set(COMPILE_FLAGS /W4)
  set(DEBUG_COMPILE_FLAGS ${COMPILE_FLAGS})
  set(RELEASE_COMPILE_FLAGS ${COMPILE_FLAGS} /O2)
endif()

target_compile_options(${RICHARD_LIB_TARGET}
  PUBLIC
    "$<$<CONFIG:DEBUG>:${DEBUG_COMPILE_FLAGS}>"
    "$<$<CONFIG:RELEASE>:${RELEASE_COMPILE_FLAGS}>"
)

target_link_libraries(${RICHARD_LIB_TARGET}
  PRIVATE
    ${CPP_UTILS_TARGET}
    Vulkan::Vulkan
    nlohmann_json::nlohmann_json
  PUBLIC
    "$<$<CONFIG:DEBUG>:${DEBUG_LINK_FLAGS}>"
    "$<$<CONFIG:RELEASE>:${RELEASE_LINK_FLAGS}>"
)

file(GLOB SHADER_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/gpu/shaders/*.glsl")

compile_shaders(
  shaders
  "${SHADER_SOURCES}"
  "${PROJECT_BINARY_DIR}/shaders"
)

add_dependencies(${RICHARD_LIB_TARGET} shaders)

add_subdirectory(test)
