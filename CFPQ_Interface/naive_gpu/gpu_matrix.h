
#ifndef GPU_MATRIX_H
#define GPU_MATRIX_H

#include "Matrix.h"
#include "multiplication.h"

class gpuMatrix : public Matrix {

    static int N;
    uint32_t *matrix_host;
    uint32_t *matrix_device;
    static uint32_t *tmp_matrix;
    
public:
    bool changed_prev;
    bool changed;

    explicit gpuMatrix(unsigned int n) : Matrix(n) {
        matrix_host = host_matrix_calloc(N);
    }

    ~gpuMatrix() {
        //dealloc
    }

    static void set_N(int n) {
        N = n;
    }

    void set_bit(unsigned int row, unsigned col) {
        matrix_host[row * cols(N) + (col / 32)] |= 1U << (31 - (col % 32));
    }

    unsigned int get_bit(unsigned int row, unsigned col) {
        return (matrix_host[row * cols(N) + (col / 32)] & 1U << (31 - (col % 32))) == 0;
    }

    bool add_mul(Matrix *left, Matrix *right) {
        auto *A = dynamic_cast<gpuMatrix *>(left);
        auto *B = dynamic_cast<gpuMatrix *>(right);
        return MatrixMulAdd(A->matrix_device, B->matrix_device, this->matrix_device, N, tmp_matrix);
    }

    void allocate_device_matrix() {
        matrix_device = device_matrix_alloc(N);
    }

    void deallocate_device_matrix() {
        
    }

    static void allocate_tmp_matrix() {
        gpuMatrix::tmp_matrix = device_matrix_alloc(N);
    }

    static void deallocate_tmp_matrix() {

    }

    void transfer_to_gpu() {
        cpu2gpu(N, matrix_host, matrix_device);
    }

    void transfer_from_gpu() {
        gpu2cpu(N, matrix_device, matrix_host);
    }
};

class gpuMatricesEnv : public MatricesEnv {
public:
    gpuMatricesEnv() {};

    ~gpuMatricesEnv() {};

    void environment_preprocessing(const std::vector<Matrix *> &matrices) {
        for (Matrix *matrix: matrices) {
            auto *gpu_matrix = dynamic_cast<gpuMatrix *>(matrix);
            gpu_matrix->allocate_device_matrix();
            gpu_matrix->transfer_to_gpu();
        }
        gpuMatrix::allocate_tmp_matrix();
        synchronize();
    };

    void environment_postprocessing(const std::vector<Matrix *> &matrices) {
        gpuMatrix::deallocate_tmp_matrix();
        for (Matrix *matrix: matrices) {
            auto *gpu_matrix = dynamic_cast<gpuMatrix *>(matrix);
            gpu_matrix->transfer_from_gpu();
        }
        synchronize();
        for (Matrix *matrix: matrices) {
            auto *gpu_matrix = dynamic_cast<gpuMatrix *>(matrix);
            gpu_matrix->deallocate_device_matrix();
        }
    };
};


#endif //GPU_MATRIX_H
