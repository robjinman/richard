cmake_minimum_required(VERSION 3.5)

include(externalGoogleTest)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/../src")
file(GLOB_RECURSE SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

add_executable(unitTests ${SRCS})

target_link_libraries(unitTests ${RICHARD_LIB_NAME} googleTest)
target_compile_options(unitTests PUBLIC -Wall -g -DDEBUG)
