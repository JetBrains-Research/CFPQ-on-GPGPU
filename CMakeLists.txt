cmake_minimum_required(VERSION 3.10)
project(CFPQ_CUDA CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v -Xptxas -O3 -v")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(CMAKE_CXX_STANDARD 11)

add_executable(naiveGPU implementations/naive-gpu/GpuMatrix.h implementations/naive-gpu/GpuMatrix.cpp
        implementations/naive-gpu/Multiplication.cu implementations/naive-gpu/Multiplication.h
        implementations/naive-gpu/Parameters.h
        cfpq-cpp/Grammar.cpp cfpq-cpp/Grammar.h cfpq-cpp/Graph.cpp cfpq-cpp/Graph.h cfpq-cpp/Matrix.h
        implementations/naive-gpu/main.cpp implementations/naive-gpu/Parameters.h)
