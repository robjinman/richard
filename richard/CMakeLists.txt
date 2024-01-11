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

set(BUILD_TESTING OFF)

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

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(ABSL_ENABLE_INSTALL ON)
  set(ABSL_PROPAGATE_CXX_STD ON)

  FetchContent_Declare(
    absl
    URL https://github.com/abseil/abseil-cpp/archive/refs/tags/20230802.1.tar.gz
  )

  FetchContent_MakeAvailable(absl)

  FetchContent_Declare(
    re2
    URL https://github.com/google/re2/archive/refs/tags/2023-11-01.tar.gz
  )

  FetchContent_MakeAvailable(re2)

  set(EFFCEE_BUILD_TESTS OFF)

  FetchContent_Declare(
    effcee
    GIT_REPOSITORY https://github.com/google/effcee.git
    GIT_TAG 19b4aa87af25cb4ee779a071409732f34bfc305c
  )

  FetchContent_MakeAvailable(effcee)

  FetchContent_Declare(
    spirv-headers
    URL https://github.com/KhronosGroup/SPIRV-Headers/archive/refs/tags/vulkan-sdk-1.3.268.0.tar.gz
  )

  FetchContent_MakeAvailable(spirv-headers)

  FetchContent_Declare(
    spirv-tools
    URL https://github.com/KhronosGroup/SPIRV-Tools/archive/refs/tags/vulkan-sdk-1.3.268.0.tar.gz
  )

  FetchContent_MakeAvailable(spirv-tools)

  FetchContent_Declare(
    glslang
    URL https://github.com/KhronosGroup/glslang/archive/refs/tags/13.1.1.tar.gz
  )

  FetchContent_MakeAvailable(glslang)

  set(SHADERC_SKIP_TESTS ON)

  FetchContent_Declare(
    shaderc
    URL https://github.com/google/shaderc/archive/refs/tags/v2023.7.tar.gz
  )

  FetchContent_MakeAvailable(shaderc)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    FetchContent_Declare(
      shaderc
      URL https://storage.googleapis.com/shaderc/artifacts/prod/graphics_shader_compiler/shaderc/windows/continuous_debug_2019/35/20240103-103943/install.zip
    )
  else()
    FetchContent_Declare(
      shaderc
      URL https://storage.googleapis.com/shaderc/artifacts/prod/graphics_shader_compiler/shaderc/windows/continuous_release_2019/35/20240103-104154/install.zip
    )
  endif()

  FetchContent_MakeAvailable(shaderc)

  add_library(shaderc STATIC IMPORTED)
  set_property(TARGET shaderc PROPERTY
               IMPORTED_LOCATION ${FETCHCONTENT_BASE_DIR}/shaderc-src/lib/shaderc_shared.lib)
endif()

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
    "$<$<CONFIG:RELEASE>:${RELEASE_COMPILE_FLAGS}>")

target_link_libraries(${RICHARD_LIB_NAME}
  PRIVATE
    Vulkan::Vulkan
    shaderc
  PUBLIC
    nlohmann_json::nlohmann_json
    "$<$<CONFIG:DEBUG>:${DEBUG_LINK_FLAGS}>"
    "$<$<CONFIG:RELEASE>:${RELEASE_LINK_FLAGS}>")

add_executable(richard "${PROJECT_SOURCE_DIR}/src/main.cpp")

target_link_libraries(richard
  ${RICHARD_LIB_NAME}
  Boost::program_options
)

add_subdirectory(test)

add_custom_target(
  link_shaders ALL
  COMMAND ${CMAKE_COMMAND} -E create_symlink ${PROJECT_SOURCE_DIR}/src/gpu/shaders ${PROJECT_BINARY_DIR}/shaders
)
