
#ifndef GPU_MATRIX_H
#define GPU_MATRIX_H

#include "Grammar.h"
#include "multiplication.h"
#include "gpu_matrix.h"

class gpuMatrix : public Matrix {
public:
    uint32_t *matrix_host;
    uint32_t *matrix_device;
    bool changed_prev;
    bool changed;

    explicit gpuMatrix(unsigned int n) : Matrix(n) {
        matrix_host = host_matrix_calloc();
    }

    ~gpuMatrix() {
        //dealloc
    }

    void set_bit(unsigned int row, unsigned col) {
        matrix_host[row * cols + (col / 32)] |= 1U << (31 - (col % 32));
    }

    unsigned int get_bit(unsigned int row, unsigned col) {
        return (matrix_host[row * cols + (col / 32)] & 1U << (31 - (col % 32))) == 0;
    }

    bool add_mul(Matrix *left, Matrix *right) {
        auto *A = dynamic_cast<gpuMatrix *>(left);
        auto *B = dynamic_cast<gpuMatrix *>(right);
        return MatrixMulAdd(A->matrix_device, B->matrix_device, this->matrix_device);
    }
};

class gpuMatricesEnv : public MatricesEnv {
public:
    gpuMatricesEnv() {}

    ~gpuMatricesEnv() {}

    void environment_preprocessing(const std::vector<Matrix *> &matrices) {
        for (Matrix *matrix: matrices) {
            auto *gpu_matrix = dynamic_cast<gpuMatrix *>(matrix);
            gpu_matrix->matrix_device = device_matrix_alloc();
            cpu2gpu(gpu_matrix->matrix_host, gpu_matrix->matrix_device);
        }
        synchronize();
    }

    void environment_postprocessing(const std::vector<Matrix *> &matrices) {
        for (Matrix *matrix: matrices) {
            auto *gpu_matrix = dynamic_cast<gpuMatrix *>(matrix);
            gpu2cpu(gpu_matrix->matrix_device, gpu_matrix->matrix_host);
        }
        synchronize();
        // dealloc
    }
};


#endif //GPU_MATRIX_H
