//
// Created by DmiitriyJarosh on 14.11.2019.
//
#if (!defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 610))

#define __CUDA_LIBDEVICE__
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>

#include "CutlassMatrix8Bit.h"


// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>
#include <chrono>
#include <cmath>


//
// CUTLASS includes needed for single-precision GEMM kernel
//


// Defines cutlass::gemm::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/gemm.h"

// Defines cutlass::gemm::SgemmTraits, the structural components for single-precision GEMM
#include "cutlass/gemm/igemm_traits.h"
using namespace cutlass::gemm;

#pragma warning( disable : 4503)

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassIGemm(
        int M,
        int N,
        int K,
        int8_t alpha,
        int8_t const *A,
        int lda,
        int8_t const *B,
        int ldb,
        int8_t beta,
        int8_t *C,
        int ldc) {

    // Define type definition for single-precision CUTLASS GEMM with column-major
    // input matrices and 128x128x8 threadblock tile size.
    //
    // Note, GemmTraits<> is a generic template defined for various general matrix product
    // computations within CUTLASS. It is intended to be maximally flexible, and consequently
    // it contains numerous template arguments.
    //
    // To keep the interface manageable, several helpers are defined for plausible compositions
    // including the following example for single-precision GEMM. Typical values are used as
    // default template arguments. See `cutlass/gemm/gemm_traits.h` for more details.
    //
    typedef cutlass::gemm::IgemmTraitsBoolVector<
            cutlass::MatrixLayout::kRowMajor,   // layout of A matrix
            cutlass::MatrixLayout::kRowMajor,   // layout of B matrix
            cutlass::Shape<32, 128, 128>,
            int8_t
    > GemmTraits;

    // Define a CUTLASS GEMM type from a GemmTraits<> instantiation.
    typedef cutlass::gemm::Gemm<GemmTraits> Gemm;

    // Construct and initialize CUTLASS GEMM parameters object.
    //
    // One of CUTLASS's design patterns is to define parameters objects that are constructible
    // in host code and passed to kernels by value. These may include pointers, strides, scalars,
    // and other arguments needed by Gemm and its components.
    //
    // The benefits of this pattern are (1.) a structured, composable strategy for passing host-constructible
    // arguments to kernels and (2.) minimized initialization overhead on kernel entry.
    //
    typename Gemm::Params params;

    int result = params.initialize(
            M,     // GEMM M dimension
            N,     // GEMM N dimension
            K,     // GEMM K dimension
            alpha, // scalar alpha
            A,     // matrix A operand
            lda,
            B,     // matrix B operand
            ldb,
            beta,  // scalar beta
            C,     // source matrix C
            ldc,
            C,     // destination matrix C (may be different memory than source C matrix)
            ldc
    );

    if (result) {
        std::cerr << "Failed to initialize CUTLASS Gemm::Params object." << std::endl;
        return cudaErrorInvalidValue;
    }

    // Launch the CUTLASS GEMM kernel.
    Gemm::launch(params);

    // Return any errors associated with the launch or cudaSuccess if no error.
    return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// The source code after this point in the file is generic CUDA using the CUDA Runtime API
// and simple CUDA kernels to initialize matrices and compute the general matrix product.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Kernel to initialize a matrix with small integers.
__global__ void InitializeMatrix_kernel(
        int8_t *matrix,
        int ldm,
        int rows,
        int columns) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < rows && j < columns) {
        int offset = i + j * ldm;

        matrix[offset] = 0;
        if (i >= rows - 2 && j < 1) {
            matrix[offset] = 0x10;
        }
        if (i < 1 && j >= columns - 2) {
            matrix[offset] = 0x01;
        }
    }
}

/// Simple function to initialize a matrix to arbitrary small integers.
cudaError_t InitializeMatrix(int8_t *matrix, int ldm, int rows, int columns, int8_t * matrix_data = nullptr) {

    if (matrix_data != nullptr) {
        cudaMemcpy(matrix, matrix_data, rows * columns * sizeof(int8_t), cudaMemcpyHostToDevice);
        return cudaGetLastError();
    }
    dim3 block(16, 16);
    dim3 grid(
            (rows + block.x - 1) / block.x,
            (columns + block.y - 1) / block.y
    );

    InitializeMatrix_kernel<<< grid, block >>>(matrix, ldm, rows, columns);

    return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocates device memory for a matrix then fills with arbitrary small integers.
cudaError_t AllocateMatrix(int8_t **matrix, int ldm, int rows, int columns, int8_t * matrix_data = nullptr) {
    cudaError_t result;

    size_t sizeof_matrix = sizeof(int8_t) * ldm * columns;

    // Allocate device memory.
    result = cudaMalloc(reinterpret_cast<void **>(matrix), sizeof_matrix);

    if (result != cudaSuccess) {
        std::cerr << "Failed to allocate matrix: "
                  << cudaGetErrorString(result) << std::endl;
        return result;
    }

    // Clear the allocation.
    result = cudaMemset(*matrix, 0, sizeof_matrix);

    if (result != cudaSuccess) {
        std::cerr << "Failed to clear matrix device memory: "
                  << cudaGetErrorString(result) << std::endl;
        return result;
    }

    // Initialize matrix elements
    result = InitializeMatrix(*matrix, ldm, rows, columns, matrix_data);

    if (result != cudaSuccess) {
        std::cerr << "Failed to initialize matrix: "
                  << cudaGetErrorString(result) << std::endl;
        return result;
    }

    return result;
}

__device__ bool * isChanged;

__global__ void resetChanges(bool * isChangedGlobal) {
    isChanged = isChangedGlobal;
    *isChanged = false;
}

// Kernel for matrix sum and checking if they have changed
__global__ void MatAddKernel(int8_t* A, int8_t* B, int8_t* C, int matDim){
    int col = threadIdx.x + blockIdx.x * blockDim.x;
    int row = threadIdx.y + blockIdx.y * blockDim.y;
    int index = col + row * matDim;
    if (col < matDim && row < matDim) {
        C[index] = A[index] | B[index];
        if (C[index] != B[index]) {
            *isChanged = true;
        }
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.
int8_t * CutlassGemmSquare(
        int dim,
        int8_t alpha,
        int8_t beta,
        int8_t * matrixA = nullptr) {
    cudaError_t result;

    using namespace std::chrono;
    high_resolution_clock::time_point begin_time = high_resolution_clock::now();

    // Compute leading dimensions for each matrix.
    int lda = dim;
    int ldc = dim;

    // Compute size in bytes of the C matrix.
    size_t sizeof_C = sizeof(int8_t) * ldc * dim;

    // Define pointers to mathigh_resolution_clock::time_point algorithm_begin_time = high_resolution_clock::now();rices in GPU device memory.
    int8_t *A;
    int8_t *C_cutlass;

    //
    // Allocate matrices in GPU device memory with arbitrary seeds.
    //

    result = AllocateMatrix(&A, lda, dim, dim, matrixA);

    if (result !=  cudaSuccess) {
        return nullptr;
    }

    result = AllocateMatrix(&C_cutlass, ldc, dim, dim, nullptr);

    if (result != cudaSuccess) {
        cudaFree(A);
        return nullptr;
    }

    //
    // Launch Main Loop.
    //

    bool isChangedHost = true;
    bool * isChangedGlobal;
    cudaMalloc((void**)&isChangedGlobal, sizeof(bool));
    dim3 dimBlock(16, 16);
    dim3 dimGrid((int)ceil((double)dim / dimBlock.x), (int)ceil((double)dim / dimBlock.y));

    int8_t * host_cutlass = (int8_t *)calloc(ldc * dim, sizeof(int8_t));
//    int8_t * A_r = (int8_t *)calloc(lda * dim, sizeof(int8_t));

    high_resolution_clock::time_point algorithm_begin_time = high_resolution_clock::now();

    while(isChangedHost) {
        result = CutlassIGemm(dim, dim, dim, alpha, A, lda, A, lda, beta, C_cutlass, ldc);

        if (result != cudaSuccess) {
            std::cerr << "CUTLASS GEMM kernel failed: "
                      << cudaGetErrorString(result) << std::endl;
            cudaFree(C_cutlass);
            cudaFree(A);
            return nullptr;
        }

        resetChanges<<<1,1>>>(isChangedGlobal);
        MatAddKernel<<<dimGrid, dimBlock>>>(C_cutlass, A, C_cutlass, dim);
        cudaMemcpy(&isChangedHost, isChangedGlobal, sizeof(bool), cudaMemcpyDeviceToHost);

        result = cudaMemcpy(A, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);

        if (result != cudaSuccess) {
            std::cerr << "Failed to copy CUTLASS Loop results to next iteration: "
                      << cudaGetErrorString(result) << std::endl;
            cudaFree(C_cutlass);
            cudaFree(A);
            return nullptr;
        }
    }

    high_resolution_clock::time_point algorithm_end_time = high_resolution_clock::now();

    result = cudaMemcpy(host_cutlass, C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);
    if (result != cudaSuccess) {
        std::cerr << "Failed to copy CUTLASS GEMM results: "
                  << cudaGetErrorString(result) << std::endl;

        cudaFree(C_cutlass);
        cudaFree(A);

        return nullptr;
    }

    //
    // Free device memory allocations.
    //

    cudaFree(C_cutlass);
    cudaFree(A);

    high_resolution_clock::time_point end_time = high_resolution_clock::now();

    milliseconds algorithm_elapsed_secs = duration_cast<milliseconds>(algorithm_end_time - algorithm_begin_time);
    milliseconds elapsed_secs = duration_cast<milliseconds>(end_time - begin_time);

    printf(
            "Algorithm time: %u\nTotal time: %u\n",
            (unsigned int)algorithm_elapsed_secs.count(),
            (unsigned int)elapsed_secs.count()
    );

    return host_cutlass;
}

int8_t ** CutlassMatrix8Bit::MultMatrSquare(unsigned char ** A, int size, unsigned char * grammar_body, unsigned int * grammar_tail, int grammar_size) {
    // Scalars used for linear scaling the result of the matrix product.
    unsigned int scalars[2] = { 1, 0 };

    int8_t * matrixA = new int8_t[size * size];
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < size; j++) {
            int offset = i + j * size;
            matrixA[offset] = A[i][j];
        }
    }

    int8_t * global_device_grammar_body = nullptr;
    unsigned int * global_device_grammar_tail = nullptr;

    cudaError_t result;
    result = cudaMalloc((void**)&global_device_grammar_body, grammar_size * sizeof(int8_t));
    if (result != cudaSuccess) {
        std::cerr << "Failed to malloc grammar body: "
                  << cudaGetErrorString(result) << std::endl;
        return nullptr;
    }
    result = cudaMalloc((void**)&global_device_grammar_tail, grammar_size * sizeof(unsigned int));
    if (result != cudaSuccess) {
        std::cerr << "Failed to malloc gramar tail: "
                  << cudaGetErrorString(result) << std::endl;
        return nullptr;
    }
    result = cudaMemcpy(global_device_grammar_body, grammar_body, grammar_size * sizeof(int8_t), cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        std::cerr << "Failed to copy grammar body to device: "
                  << cudaGetErrorString(result) << std::endl;
        return nullptr;
    }
    result = cudaMemcpy(global_device_grammar_tail, grammar_tail, grammar_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (result != cudaSuccess) {
        std::cerr << "Failed to copy grammar tail to device: "
                  << cudaGetErrorString(result) << std::endl;
        return nullptr;
    }

    result = cudaMemcpyToSymbol(device_grammar_body, global_device_grammar_body, grammar_size * sizeof(int8_t));
    if (result != cudaSuccess) {
        std::cerr << "Failed to copy grammar body to __const__: "
                  << cudaGetErrorString(result) << std::endl;
        return nullptr;
    }
    result = cudaMemcpyToSymbol(device_grammar_tail, global_device_grammar_tail, grammar_size * sizeof(unsigned int));
    if (result != cudaSuccess) {
        std::cerr << "Failed to copy grammar tail to __const__: "
                  << cudaGetErrorString(result) << std::endl;
        return nullptr;
    }
    result = cudaMemcpyToSymbol(device_grammar_size, &grammar_size, sizeof(int));
    if (result != cudaSuccess) {
        std::cerr << "Failed to copy grammar size to __const__: "
                  << cudaGetErrorString(result) << std::endl;
        return nullptr;
    }
    cudaDeviceSynchronize();

    //
    // Run the CUTLASS GEMM test.
    //

    int8_t * res = CutlassGemmSquare(
            size,
            scalars[0],     // alpha
            scalars[1],     // beta
            matrixA
    );

    int8_t ** output = new int8_t*[size];
    for (int i = 0; i < size; i++) {
        output[i] = new int8_t[size];
    }

    for (unsigned int i = 0; i < size; i++) {
        for (unsigned int j = 0; j < size; j++) {
            output[i][j] = res[i + j * size];
        }
    }

    return output;
}

#endif