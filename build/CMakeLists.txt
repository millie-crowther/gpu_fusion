cmake_minimum_required(VERSION 3.5)

# choose compiler
set (CMAKE_CXX_COMPILER "/usr/bin/nvcc")

set (CMAKE_CXX_FLAGS "-lX11 -Dcimg_display=0 -std=c++11")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D_FORCE_INLINES")

project(gpufusion)

find_package(CUDA QUIET REQUIRED)

set(INCLUDE_DIR
    ../include
)

include_directories(${INCLUDE_DIR})

set(SOURCES
    ../src/gpu_fusion.cu
    ../src/fusion.cpp
    ../src/main.cpp
)

# Pass options to NVCC
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -O3 -gencode arch=compute_35,code=sm_35)

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(gpufusion ${SOURCES})

