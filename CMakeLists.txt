cmake_minimum_required(VERSION 3.10)
project(CFPQ_CUDA CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --ptxas-options=-v -Xptxas -O3 -v")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

set(CMAKE_CXX_STANDARD 11)

string(APPEND CMAKE_CUDA_FLAGS " -gencode arch=compute_61,code=sm_61")

include_directories(${CMAKE_SOURCE_DIR}/cfpq-cpp)

add_executable(naiveGPU
        cuda/naive-gpu/GpuMatrix.h
        cuda/naive-gpu/GpuMatrix.cpp
        cuda/naive-gpu/Multiplication.cu
        cuda/naive-gpu/Multiplication.h
        cuda/naive-gpu/Parameters.h
        cuda/naive-gpu/main.cpp
        cfpq-cpp/Grammar.cpp cfpq-cpp/Grammar.h cfpq-cpp/Graph.cpp cfpq-cpp/Graph.h cfpq-cpp/Matrix.h
        )

add_executable(naiveGPU-shared
        cuda/naive-gpu-shared/GpuMatrix.h
        cuda/naive-gpu-shared/GpuMatrix.cpp
        cuda/naive-gpu-shared/Multiplication.cu
        cuda/naive-gpu-shared/Multiplication.h
        cuda/naive-gpu-shared/Parameters.h
        cuda/naive-gpu-shared/main.cpp
        cfpq-cpp/Grammar.cpp cfpq-cpp/Grammar.h cfpq-cpp/Graph.cpp cfpq-cpp/Graph.h cfpq-cpp/Matrix.h
        )

add_executable(cutlassGPU
        cuda/cutlass/main.cpp
        cuda/cutlass/CutlassMatrix.cu
        cuda/cutlass/CutlassMatrix.h
        cfpq-cpp/Grammar.cpp cfpq-cpp/Grammar.h cfpq-cpp/Graph.cpp cfpq-cpp/Graph.h cfpq-cpp/Matrix.h
        )

add_executable(cutlassGPU8
        cuda/cutlass8/main.cpp
        cfpq-cpp/Grammar.cpp cfpq-cpp/Grammar.h cfpq-cpp/Graph.cpp cfpq-cpp/Graph.h cfpq-cpp/Matrix.h
        cuda/cutlass8/CutlassMatrix8Bit.cu cuda/cutlass8/CutlassMatrix8Bit.h)

add_executable(m4riGPU
        cuda/methodOf4RusBooleanSemiringGpu/methodOf4RusBooleanSemiringGpu.cu
        cuda/methodOf4RusBooleanSemiringGpu/methodOf4RusBooleanSemiringMatrix.cu
        cuda/methodOf4RusBooleanSemiringGpu/methodOf4RusBooleanSemiringMatrix.h
        cuda/methodOf4RusBooleanSemiringGpu/Constants.h
        cuda/methodOf4RusBooleanSemiringGpu/gpuMemoryManagement.cu
        cuda/methodOf4RusBooleanSemiringGpu/gpuMemoryManagement.h
        cuda/methodOf4RusBooleanSemiringGpu/cfpqM4RusGpu.cpp
        cfpq-cpp/Grammar.cpp cfpq-cpp/Grammar.h cfpq-cpp/Graph.cpp cfpq-cpp/Graph.h cfpq-cpp/Matrix.h
        )

link_directories(/usr/local/include)
add_executable(m4riCPU
        m4ri/main.cpp
        m4ri/m4ri_matrix.h
        m4ri/m4ri_matrix.cpp
        cfpq-cpp/Grammar.cpp cfpq-cpp/Grammar.h cfpq-cpp/Graph.cpp cfpq-cpp/Graph.h cfpq-cpp/Matrix.h
        )

add_executable(graphBlasCPU
        graphBlas/main.cpp
        graphBlas/graphblas_matrix.h
        graphBlas/graphblas_matrix.cpp
        cfpq-cpp/Grammar.cpp cfpq-cpp/Grammar.h cfpq-cpp/Graph.cpp cfpq-cpp/Graph.h cfpq-cpp/Matrix.h
        )

target_link_libraries(graphBlasCPU graphblas)
target_link_libraries(m4riCPU m4ri)
