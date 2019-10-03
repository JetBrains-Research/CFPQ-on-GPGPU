//
// Created by adminlinux on 24.09.2019.
//

#define __CUDA_LIBDEVICE__
#include <cuda_runtime_api.h>
#include <cuda.h>
#include <iostream>

#include "CutlassMatrix.h"


// Standard Library includes
#include <iostream>
#include <sstream>
#include <vector>


//
// CUTLASS includes needed for single-precision GEMM kernel
//


// Defines cutlass::gemm::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/gemm.h"

// Defines cutlass::gemm::SgemmTraits, the structural components for single-precision GEMM
#include "cutlass/gemm/sgemm_traits.h"
using namespace cutlass::gemm;

//__global__ void change_magic_num() {
//    device_magic_num = 5;
//}

#pragma warning( disable : 4503)

///////////////////////////////////////////////////////////////////////////////////////////////////
//
// This function defines a CUTLASS GEMM kernel instantiation, constructs its parameters object,
// and launches it on the CUDA device.
//
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassSgemmNN(
        int M,
        int N,
        int K,
        unsigned int alpha,
        unsigned int const *A,
        int lda,
        unsigned int const *B,
        int ldb,
        unsigned int beta,
        unsigned int *C,
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
    typedef cutlass::gemm::SgemmTraits<
            cutlass::MatrixLayout::kColumnMajor,   // layout of A matrix
            cutlass::MatrixLayout::kColumnMajor,   // layout of B matrix
            cutlass::Shape<8, 128, 128>,           // threadblock tile size
            cutlass::gemm::LinearScaling<unsigned int>,
            cutlass::Shape<8, 8, 8>,
            1,
            1,
            int,
            cutlass::gemm::IgemmConfig<cutlass::Shape<8, 128, 128>, cutlass::Shape<8, 8, 8>, 1, 1, false>
    >
            GemmTraits;

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
        unsigned int *matrix,
        int ldm,
        int rows,
        int columns,
        int seed = 0) {

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
cudaError_t InitializeMatrix(unsigned int *matrix, int ldm, int rows, int columns, unsigned int ** matrix_data = nullptr) {

    if (matrix_data != nullptr) {
        unsigned int * matrix_in_line = new unsigned int[rows * columns];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < columns; j++) {
                int offset = i + j * ldm;
                matrix_in_line[offset] = matrix_data[i][j];
            }
        }
        cudaMemcpy(matrix, matrix_in_line, rows * columns * sizeof(unsigned int), cudaMemcpyHostToDevice);
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
cudaError_t AllocateMatrix(unsigned int **matrix, int ldm, int rows, int columns, unsigned int ** matrix_data = nullptr) {
    cudaError_t result;

    size_t sizeof_matrix = sizeof(unsigned int) * ldm * columns;

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

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Naive reference GEMM computation.
__global__ void ReferenceGemm_kernel(
        int M,
        int N,
        int K,
        unsigned int alpha,
        unsigned int const *A,
        int lda,
        unsigned int const *B,
        int ldb,
        unsigned int beta,
        unsigned int *C,
        int ldc) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < M && j < N) {
        unsigned int accumulator = 0;

        for (int k = 0; k < K; ++k) {
            accumulator += A[i + k * lda] * B[k + j * ldb];
        }

        C[i + j * ldc] = alpha * accumulator + beta * C[i + j * ldc];
    }
}

/// Reference GEMM computation.
cudaError_t ReferenceGemm(
        int M,
        int N,
        int K,
        unsigned int alpha,
        unsigned int const *A,
        int lda,
        unsigned int const *B,
        int ldb,
        unsigned int beta,
        unsigned int *C,
        int ldc) {

    dim3 block(16, 16);
    dim3 grid(
            (M + block.x - 1) / block.x,
            (N + block.y - 1) / block.y
    );

    ReferenceGemm_kernel<<< grid, block >>>(M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);

    return cudaGetLastError();
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Allocate several matrices in GPU device memory and call a single-precision
/// CUTLASS GEMM kernel.
unsigned int * TestCutlassGemm(int M, int N, int K, unsigned int alpha, unsigned int beta, unsigned int ** matrixA = nullptr,
                            unsigned int ** matrixB = nullptr) {
    cudaError_t result;

    //
    // Define several matrices to be used as operands to GEMM kernels.
    //

    // Compute leading dimensions for each matrix.
    int lda = M;
    int ldb = K;
    int ldc = M;

    // Compute size in bytes of the C matrix.
    size_t sizeof_C = sizeof(unsigned int) * ldc * N;
    size_t sizeof_A = sizeof(unsigned int) * lda * N;
    size_t sizeof_B = sizeof(unsigned int) * ldb * N;

    // Define pointers to matrices in GPU device memory.
    unsigned int *A;
    unsigned int *B;
    unsigned int *C_cutlass;
    unsigned int *C_reference;

    //
    // Allocate matrices in GPU device memory with arbitrary seeds.
    //

    result = AllocateMatrix(&A, lda, M, K, matrixA);

    if (result !=  cudaSuccess) {
        return nullptr;
    }

    result = AllocateMatrix(&B, ldb, K, N, matrixB);

    if (result !=  cudaSuccess) {
        cudaFree(A);
        return nullptr;
    }

    result = AllocateMatrix(&C_cutlass, ldc, M, N, nullptr);

    if (result != cudaSuccess) {
        cudaFree(A);
        cudaFree(B);
        return nullptr;
    }

    result = AllocateMatrix(&C_reference, ldc, M, N, nullptr);

    if (result != cudaSuccess) {
        cudaFree(A);
        cudaFree(B);
        cudaFree(C_cutlass);
        return nullptr;
    }

    result = cudaMemcpy(C_reference, C_cutlass, sizeof_C, cudaMemcpyDeviceToDevice);

    if (result != cudaSuccess) {
        std::cerr << "Failed to copy C_cutlass matrix to C_reference: "
                  << cudaGetErrorString(result) << std::endl;

        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);

        return nullptr;
    }

    //
    // Launch CUTLASS GEMM.
    //

    result = CutlassSgemmNN(M, N, K, alpha, A, lda, B, ldb, beta, C_cutlass, ldc);

    if (result != cudaSuccess) {
        std::cerr << "CUTLASS GEMM kernel failed: "
                  << cudaGetErrorString(result) << std::endl;

        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);

        return nullptr;
    }

    // Copy to host and verify equivalence.
    unsigned int * host_cutlass = (unsigned int *)calloc(ldc * N, sizeof(unsigned int));
//    unsigned int * host_reference = (unsigned int *)calloc(ldc * N, sizeof(unsigned int));
    unsigned int * A_r = (unsigned int *)calloc(lda * N, sizeof(unsigned int));
    unsigned int * B_r = (unsigned int *)calloc(ldb * N, sizeof(unsigned int));

    result = cudaMemcpy(A_r, A, sizeof_A, cudaMemcpyDeviceToHost);
    result = cudaMemcpy(B_r, B, sizeof_B, cudaMemcpyDeviceToHost);
    result = cudaMemcpy(host_cutlass, C_cutlass, sizeof_C, cudaMemcpyDeviceToHost);

    if (result != cudaSuccess) {
        std::cerr << "Failed to copy CUTLASS GEMM results: "
                  << cudaGetErrorString(result) << std::endl;

        cudaFree(C_reference);
        cudaFree(C_cutlass);
        cudaFree(B);
        cudaFree(A);

        return nullptr;
    }

//    result = cudaMemcpy(host_reference, C_reference, sizeof_C, cudaMemcpyDeviceToHost);

//    if (result != cudaSuccess) {
//        std::cerr << "Failed to copy Reference GEMM results: "
//                  << cudaGetErrorString(result) << std::endl;
//
//        cudaFree(C_reference);
//        cudaFree(C_cutlass);
//        cudaFree(B);
//        cudaFree(A);
//
//        return nullptr;
//    }

//    for (int i = 0; i < 16; i++) {
//        if (i % 4 == 0) {
//            std::cout << std::endl;
//        }
//        std::cout << A_r[i] << " ";
//    }
//    std::cout << std::endl;
//
//    for (int i = 0; i < 16; i++) {
//        if (i % 4 == 0) {
//            std::cout << std::endl;
//        }
//        std::cout << B_r[i] << " ";
//    }
//    std::cout << std::endl;
//
//    for (int i = 0; i < 16; i++) {
//        if (i % 4 == 0) {
//            std::cout << std::endl;
//        }
//        std::cout << host_cutlass[i] << " ";
//    }
//    std::cout << std::endl;

    //
    // Free device memory allocations.
    //

    cudaFree(C_reference);
    cudaFree(C_cutlass);
    cudaFree(B);
    cudaFree(A);

    return host_cutlass;
}

// function which set grammar to device for special multiplication
__global__ void set_grammar(unsigned int * global_device_grammar_body, unsigned long long * global_device_grammar_tail, int grammar_size) {
    device_grammar_body = global_device_grammar_body;
    device_grammar_tail = global_device_grammar_tail;
    device_grammar_size = grammar_size;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

/// Entry point to basic_gemm example.
//
// usage:
//
//   my_mult_example <M> <N> <K> <alpha> <beta>
//

unsigned int ** CutlassMatrix::MultMatr(unsigned int ** matrixA, unsigned int ** matrixB, int size, unsigned int * grammar_body, unsigned long long * grammar_tail, int grammar_size) {

    //
    // Parse the command line to obtain GEMM dimensions and scalar values.
    //

    // GEMM problem dimensions.
    int problem[3] = { size, size, size };

    // Scalars used for linear scaling the result of the matrix product.
    unsigned int scalars[2] = { 1, 0 };

//    unsigned int ** matrixA = new unsigned int*[4];
//    matrixA[0] = new unsigned int[4]{0,0,0,0x1};
//    matrixA[1] = new unsigned int[4]{0,0,0,0};
//    matrixA[2] = new unsigned int[4]{0,0,0,0};
//    matrixA[3] = new unsigned int[4]{0x10,0,0,0};
//
//    unsigned int ** matrixB = new unsigned int*[4];
//    matrixB[0] = new unsigned int[4]{0,0,0,0x1};
//    matrixB[1] = new unsigned int[4]{0,0,0,0};
//    matrixB[2] = new unsigned int[4]{0,0,0,0};
//    matrixB[3] = new unsigned int[4]{0x10,0,0,0};

//    int grammar_size = 3;
//    unsigned int * grammar_body = new unsigned int[grammar_size]{0x1,0x100,0x10};
//    unsigned long long * grammar_tail = new unsigned long long[grammar_size]{0x100000010,0x100000010,0x10};

    unsigned int * global_device_grammar_body = nullptr;
    unsigned long long * global_device_grammar_tail = nullptr;

    cudaMalloc((void**)&global_device_grammar_body, grammar_size * sizeof(unsigned int));
    cudaMalloc((void**)&global_device_grammar_tail, grammar_size * sizeof(unsigned long long));

    cudaMemcpy(global_device_grammar_body, grammar_body, grammar_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(global_device_grammar_tail, grammar_tail, grammar_size * sizeof(unsigned long long), cudaMemcpyHostToDevice);

    set_grammar<<<1,1>>>(global_device_grammar_body, global_device_grammar_tail, grammar_size);

    cudaDeviceSynchronize();

    //
    // Run the CUTLASS GEMM test.
    //


    unsigned int * res = TestCutlassGemm(
            problem[0],     // GEMM M dimension
            problem[1],     // GEMM N dimension
            problem[2],     // GEMM K dimension
            scalars[0],     // alpha
            scalars[1],     // beta
            matrixA,
            matrixB
    );

    unsigned int ** output = new unsigned int*[size];
    for (int i = 0; i < size; i++) {
        output[i] = new unsigned int[size];
    }

    for (unsigned int i = 0; i < size; i++) {
        for (unsigned int j = 0; j < size; j++) {
            output[i][j] = res[i * size + j];
        }
    }

    return output;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
