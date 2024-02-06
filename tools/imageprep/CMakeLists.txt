cmake_minimum_required(VERSION 3.24)

set(PROJECT_NAME ImagePrep)

project(${PROJECT_NAME})

if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()

message("Building ${CMAKE_BUILD_TYPE} configuration...")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(IMAGE_PREP_TARGET imageprep)

# Can't use FetchContent for libjpeg-turbo
ExternalProject_Add(libjpeg
  GIT_REPOSITORY https://github.com/libjpeg-turbo/libjpeg-turbo.git
  GIT_TAG 3.0.2
  CMAKE_ARGS -DCMAKE_INSTALL_PREFIX:PATH=<INSTALL_DIR>
)

file(GLOB
  CPP_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_executable(${IMAGE_PREP_TARGET} ${CPP_SOURCES})

ExternalProject_Get_Property(libjpeg install_dir)
set(LIBJSON_DIR ${install_dir})

target_include_directories(${IMAGE_PREP_TARGET}
  PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    ${LIBJSON_DIR}/include
)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(COMPILE_FLAGS -Wextra -Wall)
  set(DEBUG_COMPILE_FLAGS ${COMPILE_FLAGS} -g)
  set(RELEASE_COMPILE_FLAGS ${COMPILE_FLAGS} -O3)
  set(LINK_FLAGS ${LIBJSON_DIR}/lib/libjpeg.a)
  set(DEBUG_LINK_FLAGS ${LINK_FLAGS})
  set(RELEASE_LINK_FLAGS ${LINK_FLAGS})
elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
  set(COMPILE_FLAGS /W4)
  set(DEBUG_COMPILE_FLAGS ${COMPILE_FLAGS})
  set(RELEASE_COMPILE_FLAGS ${COMPILE_FLAGS} /O2)
  set(LINK_FLAGS ${LIBJSON_DIR}/lib/turbojpeg-static.lib)
  set(DEBUG_LINK_FLAGS ${LINK_FLAGS})
  set(RELEASE_LINK_FLAGS ${LINK_FLAGS})
endif()

target_compile_options(${IMAGE_PREP_TARGET}
  PRIVATE
    "$<$<CONFIG:DEBUG>:${DEBUG_COMPILE_FLAGS}>"
    "$<$<CONFIG:RELEASE>:${RELEASE_COMPILE_FLAGS}>"
)

target_link_libraries(${IMAGE_PREP_TARGET}
  PRIVATE
    ${CPP_UTILS_TARGET}
    "$<$<CONFIG:DEBUG>:${DEBUG_LINK_FLAGS}>"
    "$<$<CONFIG:RELEASE>:${RELEASE_LINK_FLAGS}>"
)
