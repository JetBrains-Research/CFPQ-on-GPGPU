
#pragma once

#include <vector>
#include "Matrix.h"
#include "gpu_memory_management.h"
#include "methodOf4RusBooleanSemiringGpu.h"

#define SQUEEZE 32

class MethodOf4RusMatricesEnv : public MatricesEnv {
public:

    int size_multiple_by_32;
    int cols;
    uint32_t *extra_matrix_device;
    Tables tables;

    MethodOf4RusMatricesEnv() {}

    ~MethodOf4RusMatricesEnv() override;

    void environment_preprocessing(const std::vector<Matrix *> &matrices) override;

    void environment_postprocessing(const std::vector<Matrix *> &matrices) override;
};

class MethodOf4RusMatrix : public Matrix {
public:
    
    int size_multiple_by_32;
    int cols;
    uint32_t *matrix_host;
    uint32_t *matrix_device;
    MethodOf4RusMatricesEnv *env;

    explicit MethodOf4RusMatrix(unsigned int n) : Matrix(n) {
        size_multiple_by_32 = n;

        if (n % SQUEEZE != 0) {
            int part = SQUEEZE - (n % SQUEEZE);
            size_multiple_by_32 += part;
        }
        cols = size_multiple_by_32 / SQUEEZE;

        matrix_device = allocate_matrix_device(size_multiple_by_32, cols);
        matrix_host = allocate_matrix_host(size_multiple_by_32, cols);
    }

    ~MethodOf4RusMatrix() {
        delete_matrix_device(matrix_device);
        delete_matrix_host(matrix_host);
    }

    void set_bit(unsigned int row, unsigned col) override;

    unsigned int get_bit(unsigned int row, unsigned col) override;

    bool add_mul(Matrix *left, Matrix *right) override;
};
