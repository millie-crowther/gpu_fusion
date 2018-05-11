cmake_minimum_required(VERSION 3.5)

# choose compiler
set (CMAKE_CXX_COMPILER "/usr/bin/nvcc")
#set (CMAKE_CXX_COMPILER "/usr/bin/g++")

set (CMAKE_CXX_FLAGS "-lX11 -Dcimg_display=0")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

project(gpufusion CXX)

set(INCLUDE_DIR
    ../include
)

include_directories(${INCLUDE_DIR})

set(SOURCES
    ../src/main.cpp
    ../src/fusion.cpp
    ../src/gpu_fusion.cu
)

add_executable(gpufusion ${SOURCES})
