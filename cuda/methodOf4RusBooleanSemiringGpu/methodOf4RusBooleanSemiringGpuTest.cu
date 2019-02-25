
#include "methodOf4RusBooleanSemiringGpu.h"
#include "gpuMemoryManagement.h"
#include "gpuTimer.cu"
#include <time.h>
#include <assert.h>
#include <stdlib.h>
#include <limits.h>
#include "Constants.h"

#define BLOCK_SIZE 32
#define BITS sizeof(TYPE) * 8// aka 32

using namespace gpu_m4ri;

/*
 *squeeze src to dst by 32 in rows
 */
void squeeze_to_bits_rows(const TYPE *src, int src_rows, int src_cols, TYPE *dst, int dst_cols) {
    for (int i = 0; i < src_rows; i++) {
        for (int j = 0; j < dst_cols; j++) {
            TYPE value = 0;
            for (int n = 0; n < BITS; n++) {
                if (src[i * src_cols + j * BITS + n] != 0) {
                    value |= 1ULL << (31 - n);
                }
            }

            dst[i * dst_cols + j] = value;
        }
    }
}

// dummy mul for testing method of four russian
__global__ void dummy_gpu_semiring_mul(TYPE *a, TYPE *b, TYPE *c, int m, int n, int k) { 
    int row = blockIdx.y * blockDim.y + threadIdx.y; 
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int sum = 0;
    if( col < k && row < m) 
    {
        for(int i = 0; i < n; i++) 
        {
            sum |= a[row * n + i] & b[i * k + col];
        }
        c[row * k + col] |= sum;
    }
}

void wrapper_sdummy_semiring_mul(TYPE *a, TYPE *b, TYPE *c, int rows, int cols) {
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (rows + BLOCK_SIZE - 1) / BLOCK_SIZE); 
    dummy_gpu_semiring_mul<<<dimGrid,dimBlock>>>(a, b, c, rows, cols, cols);
    cudaDeviceSynchronize();
}
 
void rand_fill(int rows, int sparsity, TYPE *matrix) {
    for (int i = 0; i < rows * rows; i++) {
        if (rand() % sparsity == 0) {
            matrix[i] = 1;
        }
        else {
            matrix[i] = 0;
        }
    }
}

int method_of_4rus_test(int rows, int table_cols_max, int sparsity) {
    if(rows % SQUEEZE != 0) {
        rows +=  (SQUEEZE - (rows % SQUEEZE));
    }

    int cols = rows / SQUEEZE;
    Tables tables;
    GpuTimer gpuTimer = GpuTimer();
    float elapsedTime;
    tables.initialize(rows, cols, table_cols_max);
    
    TYPE *unsqueezed_matrixA   = allocate_matrix_host(rows, rows);
    TYPE *unsqueezed_matrixB   = allocate_matrix_host(rows, rows);
    TYPE *unsqueezed_matrixC   = allocate_matrix_host(rows, rows);
    TYPE *unsqueezed_matrixAXB = allocate_matrix_host(rows, rows);

    TYPE *squeezed_matrixA   = allocate_matrix_host(rows, cols);
    TYPE *squeezed_matrixB   = allocate_matrix_host(rows, cols);
    TYPE *squeezed_matrixC   = allocate_matrix_host(rows, cols);    
    TYPE *squeezed_matrixAXB = allocate_matrix_host(rows, cols);
    
    //rand fill matrices  
    srand(time(NULL));
    rand_fill(rows, sparsity, unsqueezed_matrixA);
    rand_fill(rows, sparsity, unsqueezed_matrixB);
    rand_fill(rows, sparsity, unsqueezed_matrixC);
    for (int i = 0; i < rows * rows; i++) {
        unsqueezed_matrixAXB[i] = unsqueezed_matrixC[i];
    } 

    // device matrices for dummy multiplication
    TYPE *a_d  = allocate_matrix_device(rows, rows);
    TYPE *b_d  = allocate_matrix_device(rows, rows);
    TYPE *axb_d  = allocate_matrix_device(rows, rows);

    copy_host_to_device_sync(unsqueezed_matrixA, a_d, rows * rows);
    copy_host_to_device_sync(unsqueezed_matrixB, b_d, rows * rows);
    copy_host_to_device_sync(unsqueezed_matrixAXB, axb_d, rows * rows);
    
    wrapper_sdummy_semiring_mul(a_d, b_d, axb_d, rows, rows);
    copy_device_to_host_sync(axb_d, unsqueezed_matrixAXB, rows * rows);

    delete_matrix_device(a_d);
    delete_matrix_device(b_d);
    delete_matrix_device(axb_d);
    
    // squeeze dummy matrix as right answer to check correctness of multiplication
    squeeze_to_bits_rows(unsqueezed_matrixAXB, rows, rows, squeezed_matrixAXB, cols);
    
    // squeeze matrices for mul
    squeeze_to_bits_rows(unsqueezed_matrixA, rows, rows, squeezed_matrixA, cols);
    squeeze_to_bits_rows(unsqueezed_matrixC, rows, rows, squeezed_matrixC, cols);   
    squeeze_to_bits_rows(unsqueezed_matrixB, rows, rows, squeezed_matrixB, cols);
    
    
    // allocate device memory for squeezed
    TYPE *squeezed_matrixB_device = allocate_matrix_device(rows, cols);
    TYPE *squeezed_matrixA_device = allocate_matrix_device(rows, cols);
    TYPE *squeezed_matrixC_device = allocate_matrix_device(rows, cols);

    copy_host_to_device_sync(squeezed_matrixB, squeezed_matrixB_device, rows * cols);
    copy_host_to_device_sync(squeezed_matrixA, squeezed_matrixA_device, rows * cols);
    copy_host_to_device_sync(squeezed_matrixC, squeezed_matrixC_device, rows * cols);
     
    gpuTimer.Start();    
    wrapper_method_of_4rus_bool_semiring(squeezed_matrixA_device, squeezed_matrixB_device, 
                                              squeezed_matrixC_device, tables, rows, cols);
    gpuTimer.Stop();
    elapsedTime = gpuTimer.ElapsedMs();
    
    copy_device_to_host_sync(squeezed_matrixC_device, squeezed_matrixC, rows * cols);

    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < rows / SQUEEZE; j++) {
           assert(squeezed_matrixC[i * rows / SQUEEZE + j] == squeezed_matrixAXB[i * rows / SQUEEZE + j]);
       }
    }
    
    delete_matrix_host(unsqueezed_matrixB);
    delete_matrix_host(unsqueezed_matrixA);
    delete_matrix_host(unsqueezed_matrixC);
    delete_matrix_host(unsqueezed_matrixAXB);

    delete_matrix_host(squeezed_matrixAXB);
    delete_matrix_host(squeezed_matrixA);
    delete_matrix_host(squeezed_matrixB);
    delete_matrix_host(squeezed_matrixC);

    delete_matrix_device(squeezed_matrixB_device);
    delete_matrix_device(squeezed_matrixA_device);
    delete_matrix_device(squeezed_matrixC_device);
    tables.free();
    
    printf("Test passed for sparsity=%d, rows = %d is %f ms.\n", sparsity, rows, elapsedTime);
    return 1;
}

/*
* ./program_name initial_size max_size max_sparsity size_step sparsity_step table_size
* 
*
*/
int main(int argc, char *argv[]) {
    int initial_size = strtol(argv[1], NULL, 10);
    int max_size = strtol(argv[2], NULL, 10);
    int max_sparsity = strtol(argv[3], NULL, 10);
    int size_step = strtol(argv[4], NULL, 10);
    int sparsity_step = strtol(argv[5], NULL, 10);
    int table_size = strtol(argv[6], NULL, 10);
    for(int sparsity = 2; sparsity < max_sparsity; sparsity += sparsity_step) {
        for(int size = initial_size; size < max_size; size += size_step) {
            method_of_4rus_test(size, table_size, sparsity); 
        }  
    }
}
