cmake_minimum_required(VERSION 3.10)
project(CFPQ_CUDA CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v -O3")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(CMAKE_CXX_STANDARD 11)

add_executable(naiveGPU CFPQ_Interface/naive_gpu/gpu_matrix.h CFPQ_Interface/naive_gpu/gpu_matrix.cpp
        CFPQ_Interface/naive_gpu/multiplication.cu CFPQ_Interface/naive_gpu/multiplication.h
        CFPQ_Interface/naive_gpu/parameters.h
        cfpq-cpp/Grammar.cpp cfpq-cpp/Grammar.h cfpq-cpp/Graph.cpp cfpq-cpp/Graph.h cfpq-cpp/Matrix.h
        CFPQ_Interface/naive_gpu/main.cpp CFPQ_Interface/naive_gpu/parameters.h)
