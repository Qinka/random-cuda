# CMake for random-cuda/learn
# To learninf the CUDA

## Setting CMake
cmake_minimum_required(VERSION 3.3)
find_package(CUDA QUIET REQUIRED)

project(cuda-learn)

enable_language(C C++ CUDA)

include_directories(include)

## hello for CUDA, and output the GPUs' infos
cuda_add_executable(hello-cuda source/hello.cu)

#add_executable(c-example source/c-example.c)
#cuda_add_executable(cuda-example source/cuda-example.cu)

