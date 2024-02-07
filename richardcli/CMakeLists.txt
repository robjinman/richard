cmake_minimum_required(VERSION 3.24)

set(RICHARD_CLI_TARGET richardcli)
set(RICHARD_CLI_TARGET ${RICHARD_CLI_TARGET} PARENT_SCOPE)

if (CPUPROF)
  message("CPU profiling ON")
  
  find_package(GOOGLE_PERFTOOLS REQUIRED)

  if (NOT DEFINED PROFILE_DURATION)
    set(PROFILE_DURATION -1)
  endif()
endif()

set(BOOST_INCLUDE_LIBRARIES program_options)
set(BOOST_ENABLE_CMAKE ON)

FetchContent_Declare(Boost
  URL https://github.com/boostorg/boost/releases/download/boost-1.84.0/boost-1.84.0.tar.xz
  URL_MD5 893b5203b862eb9bbd08553e24ff146a
  DOWNLOAD_EXTRACT_TIMESTAMP ON
)

FetchContent_MakeAvailable(Boost)

file(GLOB
  CPP_SOURCES
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

add_executable(${RICHARD_CLI_TARGET} ${CPP_SOURCES})

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

target_compile_options(${RICHARD_CLI_TARGET}
  PRIVATE
    "$<$<CONFIG:DEBUG>:${DEBUG_COMPILE_FLAGS}>"
    "$<$<CONFIG:RELEASE>:${RELEASE_COMPILE_FLAGS}>"
)

target_link_libraries(${RICHARD_CLI_TARGET}
  PRIVATE
    ${RICHARD_LIB_TARGET}
    Boost::program_options
    "$<$<CONFIG:DEBUG>:${DEBUG_LINK_FLAGS}>"
    "$<$<CONFIG:RELEASE>:${RELEASE_LINK_FLAGS}>"
)
