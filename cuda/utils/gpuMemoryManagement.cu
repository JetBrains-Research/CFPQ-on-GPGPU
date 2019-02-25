
#include "gpuMemoryManagement.h"
#include <stdio.h>
#include <iostream> 

namespace gpu_m4ri {

    #define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }

    inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
        if (code != cudaSuccess) {
            fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
            if (abort) exit(code);
        }
    }

    TYPE ** allocate_tables(int num_tables, int num_rows, int num_cols) {    
        TYPE **d_ppcPtr, *d_pcPtr;
        gpuErrchk(cudaMalloc(&d_ppcPtr, sizeof(TYPE *) * num_tables));

        for(int i = 0; i < num_tables; i ++) {
            gpuErrchk(cudaMalloc(&d_pcPtr, sizeof(TYPE) * num_rows * num_cols));
            gpuErrchk(cudaMemset(d_pcPtr, 0, sizeof(TYPE) * num_rows * num_cols));
            gpuErrchk(cudaMemcpy(&d_ppcPtr[i], &d_pcPtr, sizeof(TYPE *), cudaMemcpyHostToDevice));
        }
        return d_ppcPtr;
    }

    void delete_tables(TYPE **tables, int num_tables) { 
        TYPE **someHost;
        gpuErrchk(cudaMallocHost((void **) &someHost, sizeof(TYPE *) * num_tables)); 
        gpuErrchk(cudaMemcpy(someHost, tables, num_tables * sizeof(TYPE *), cudaMemcpyDeviceToHost));

        for(int i = 0; i < num_tables; i ++) {
            gpuErrchk(cudaFree((void *) someHost[i]));
        }
        gpuErrchk(cudaFree(tables));
        gpuErrchk(cudaFreeHost(someHost));  
    }

    TYPE * allocate_matrix_host(int rows, int cols) {
        TYPE *matrix;
        gpuErrchk(cudaMallocHost((void **) &matrix, sizeof(TYPE) * rows * cols));
        return matrix;
    }

    void delete_matrix_host(TYPE *matrix) {
        gpuErrchk(cudaFreeHost(matrix));
    }

    TYPE * allocate_matrix_device(int rows, int cols) {
        TYPE *matrix;
        gpuErrchk(cudaMalloc((void **) &matrix, sizeof(TYPE) * rows * cols));
        return matrix;
    }

    void delete_matrix_device(TYPE *matrix) {
        gpuErrchk(cudaFree(matrix));
    }

    void copy_device_to_host_sync(TYPE *device, TYPE *host, int elems) {
        gpuErrchk(cudaMemcpy(host, device, sizeof(TYPE) * elems, cudaMemcpyDeviceToHost));
    }

    void copy_host_to_device_sync(TYPE *host, TYPE *device, int elems) {
        gpuErrchk(cudaMemcpy(device, host, sizeof(TYPE) * elems, cudaMemcpyHostToDevice));
    }

    void copy_device_to_device_sync(TYPE *src, TYPE *dst, int elems) {
        gpuErrchk(cudaMemcpy(dst, src, sizeof(TYPE) * elems, cudaMemcpyDeviceToDevice));
    }

    void copy_device_to_host_async(TYPE *device, TYPE *host, int elems) {
        gpuErrchk(cudaMemcpyAsync(host, device, sizeof(TYPE) * elems, cudaMemcpyDeviceToHost));
    }

    void copy_host_to_device_async(TYPE *host, TYPE *device, int elems) {
        gpuErrchk(cudaMemcpyAsync(device, host, sizeof(TYPE) * elems, cudaMemcpyHostToDevice));
    }

    void synchronize_with_gpu() {
        cudaDeviceSynchronize();
    }
    
}
