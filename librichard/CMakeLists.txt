cmake_minimum_required(VERSION 3.27)

set(RICHARD_LIB_TARGET librichard)
set(RICHARD_LIB_TARGET ${RICHARD_LIB_TARGET} PARENT_SCOPE)

find_package(Vulkan REQUIRED)
find_package(nlohmann_json REQUIRED)

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

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(COMPILE_FLAGS -Wextra -Wall)
  if (CPU_PROFILE)
    set(COMPILE_FLAGS ${COMPILE_FLAGS} -g)
  endif()
  set(DEBUG_COMPILE_FLAGS ${COMPILE_FLAGS} -g)
  set(RELEASE_COMPILE_FLAGS ${COMPILE_FLAGS} -O3)
  set(LINK_FLAGS)
  if (CPU_PROFILE)
    set(LINK_FLAGS ${LINK_FLAGS} -Wl,--no-as-needed -lprofiler -Wl,--as-needed)
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
