
#include <iostream>
#include "multiplication.h"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        std::cout << "GPUassert: " << cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        if (abort) {
            exit(code);
        }
    }
}

#define largest_pow2 32
#define threads_x 32

__device__ bool is_changed;

// size_t matrix_memsize;
// uint *tmp_matrix;

// void initialize(int N_inp) {
// 	N = N_inp;
// 	rows = N;
// 	cols = N / largest_pow2 + (N % largest_pow2 ? 1 : 0);
// 	matrix_memsize = rows * cols * sizeof(uint);

// 	gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&tmp_matrix), matrix_memsize));
// }

inline int rows(int N) {
	return N;
}

inline int cols(int N) {
	return N / largest_pow2 + (N % largest_pow2 ? 1 : 0);
}

inline size_t matrix_memsize(int N) {
	return rows(N) * cols(N) * sizeof(uint);
}

__global__ void DummyMulAdd(uint *A, uint *B, uint *C, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y;

    if (x >= cols) {
        return;
    }

	uint acc = 0;
	uint a_el;
	for (uint i = 0; i < cols; ++i) {
		a_el = A[y * cols + i];
		#pragma unroll
		for (uint b = 0; b < 32; ++b) {
			if (a_el & 1) {
				acc |= B[x + 32 * cols * i + cols * (31 - b)];
			}
			a_el >>= 1;
		}
	}

	if (acc == 0) {
		return;
	}

	uint c_old = C[y * cols + x];
	if (c_old != (acc | c_old)) {
		is_changed = true;
		C[y * cols + x] = acc | c_old;
	}
}

__global__ void DummyMul(uint *A, uint *B, uint *C, int cols) {
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y;

    if (x >= cols) {
        return;
    }

	uint acc = 0;
	uint a_el;
	for (uint i = 0; i < cols; ++i) {
		a_el = A[y * cols + i];
		#pragma unroll
		for (uint b = 0; b < 32; ++b) {
			if (a_el & 1) {
				acc |= B[x + 32 * cols * i + cols * (31 - b)];
			}
			a_el >>= 1;
		}
	}

	C[y * cols + x] = acc;
}


__global__ void AddToLeft(uint *A, uint *B, int cols) {
	int index = blockIdx.y * cols + blockIdx.x * blockDim.x + threadIdx.x;

    if ((blockIdx.x * blockDim.x + threadIdx.x) >= cols) {
        return;
    }

	uint A_el = A[index];
	uint res = B[index] | A_el;
	if (res != A_el) {
		is_changed = true;
		A[index] = res;
	}
}

void synchronize() {
	cudaDeviceSynchronize();
}

void set_value(int N, uint *d_M, int val) {
	gpuErrchk(cudaMemset(d_M, val, matrix_memsize(N)));
}

uint * device_matrix_alloc(int N) {
	uint *d_M;
	gpuErrchk(cudaMalloc(reinterpret_cast<void **>(&d_M), matrix_memsize(N)));

	return d_M;
}

uint * host_matrix_calloc(int N) {
    uint *M;
    gpuErrchk(cudaMallocHost(reinterpret_cast<void **>(&M), matrix_memsize(N)));
    set_value(N, M, 0);
	return M;
}

void gpu2cpu(int N, uint *d_M, uint *h_M) {
	gpuErrchk(cudaMemcpyAsync(h_M, d_M, matrix_memsize(N), cudaMemcpyDeviceToHost));
}

void cpu2gpu(int N, uint *h_M, uint *d_M) {
	gpuErrchk(cudaMemcpyAsync(d_M, h_M, matrix_memsize(N), cudaMemcpyHostToDevice));
}

void setFlag() {
	bool flag = false;
	gpuErrchk(cudaMemcpyToSymbol(is_changed, &flag, sizeof(bool)))
}

bool getFlag() {
	bool flag;
	gpuErrchk(cudaMemcpyFromSymbol(&flag, is_changed, sizeof(bool)))

	return flag;
}

bool MatrixMulAdd(uint *A, uint *B, uint *C, int N, uint *tmp_matrix) {
	bool safe = (A == C) || (B == C);
	dim3 mul_threads(threads_x);
	dim3 mul_blocks(cols(N) / threads_x + (cols(N) % threads_x ? 1 : 0), rows(N));

    setFlag();
	if (safe) {
		DummyMul <<<mul_blocks, mul_threads>>> (A, B, tmp_matrix, cols(N));
		synchronize();
		gpuErrchk(cudaGetLastError());
		AddToLeft <<<mul_blocks, mul_threads>>> (C, tmp_matrix, cols(N));
		synchronize();
		gpuErrchk(cudaGetLastError());
	}
	else {
		DummyMulAdd <<<mul_blocks, mul_threads>>> (A, B, C, cols(N));
		synchronize();
		gpuErrchk(cudaGetLastError());
	}

	return getFlag();
}
