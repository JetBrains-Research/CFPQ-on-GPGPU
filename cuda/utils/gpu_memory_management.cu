
#include "gpu_memory_management.h"
#include <stdio.h>
#include <iostream> 

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
   if (code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

uint32_t ** allocate_tables(int num_tables, int num_rows, int num_cols) {    
    uint32_t **d_ppcPtr, *d_pcPtr;
    gpuErrchk(cudaMalloc(&d_ppcPtr, sizeof(uint32_t *) * num_tables));

    for(int i = 0; i < num_tables; i ++) {
        gpuErrchk(cudaMalloc(&d_pcPtr, sizeof(uint32_t) * num_rows * num_cols));
        gpuErrchk(cudaMemset(d_pcPtr, 0, sizeof(uint32_t) * num_rows * num_cols));
        gpuErrchk(cudaMemcpy(&d_ppcPtr[i], &d_pcPtr, sizeof(uint32_t *), cudaMemcpyHostToDevice));
    }
    return d_ppcPtr;
}

void delete_tables(uint32_t **tables, int num_tables) { 
    uint32_t **someHost;
    gpuErrchk(cudaMallocHost((void **) &someHost, sizeof(uint32_t *) * num_tables)); 
    gpuErrchk(cudaMemcpy(someHost, tables, num_tables * sizeof(uint32_t *), cudaMemcpyDeviceToHost));

    for(int i = 0; i < num_tables; i ++) {
        gpuErrchk(cudaFree((void *) someHost[i]));
    }
    gpuErrchk(cudaFree(tables));
    gpuErrchk(cudaFreeHost(someHost));  
}

uint32_t * allocate_matrix_host(int rows, int cols) {
    uint32_t *matrix;
    gpuErrchk(cudaMallocHost((void **) &matrix, sizeof(uint32_t) * rows * cols));
    return matrix;
}

void delete_matrix_host(uint32_t *matrix) {
    gpuErrchk(cudaFreeHost(matrix));
}

uint32_t * allocate_matrix_device(int rows, int cols) {
    uint32_t *matrix;
    gpuErrchk(cudaMalloc((void **) &matrix, sizeof(uint32_t) * rows * cols));
    return matrix;
}

void delete_matrix_device(uint32_t *matrix) {
    gpuErrchk(cudaFree(matrix));
}

void copy_device_to_host_sync(uint32_t *device, uint32_t *host, int elems) {
    gpuErrchk(cudaMemcpy(host, device, sizeof(uint32_t) * elems, cudaMemcpyDeviceToHost));
}

void copy_host_to_device_sync(uint32_t *host, uint32_t *device, int elems) {
    gpuErrchk(cudaMemcpy(device, host, sizeof(uint32_t) * elems, cudaMemcpyHostToDevice));
}

void copy_device_to_device_sync(uint32_t *src, uint32_t *dst, int elems) {
    gpuErrchk(cudaMemcpy(dst, src, sizeof(uint32_t) * elems, cudaMemcpyDeviceToDevice));
}

void copy_device_to_host_async(uint32_t *device, uint32_t *host, int elems) {
    gpuErrchk(cudaMemcpyAsync(host, device, sizeof(uint32_t) * elems, cudaMemcpyDeviceToHost));
}

void copy_host_to_device_async(uint32_t *host, uint32_t *device, int elems) {
    gpuErrchk(cudaMemcpyAsync(device, host, sizeof(uint32_t) * elems, cudaMemcpyHostToDevice));
}

void synchronize_with_gpu() {
    cudaDeviceSynchronize();
}
