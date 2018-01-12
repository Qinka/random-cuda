# CMake for random-cuda/learn
# To learninf the CUDA

## Setting CMake
cmake_minimum_required(VERSION 3.3)
find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)


project(cuda-learn)



include_directories(include)

## hello for CUDA, and output the GPUs' infosn
cuda_add_executable(hello-cuda source/hello.cu)

add_library(qinkalc SHARED source/linear-combination.cc)
cuda_add_library(qinkalc-gpu SHARED source/linear-combination.cu)

add_executable(test-lb source/test-lb.cc)
target_link_libraries(test-lb qinkalc)

add_executable(test-lb-gpu source/test-lb.cc)
target_link_libraries(test-lb-gpu qinkalc-gpu)

add_executable(img-lb source/img-lb.cc)
target_link_libraries(img-lb qinka-lc)
target_link_libraries(img-lb ${OpenCV_LIBS})
