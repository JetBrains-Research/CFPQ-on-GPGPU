
#ifndef GPU_MATRIX_H
#define GPU_MATRIX_H

#include "multiplication.h"
#include "../../cfpq-cpp/Matrix.h"

class gpuMatrix : public Matrix {

    uint32_t *matrix_host;
    uint32_t *matrix_device;
    
public:
    static int N;
    static uint32_t *tmp_matrix;
    bool changed_prev;
    bool changed;

    explicit gpuMatrix(unsigned int n) : Matrix(n) {
        matrix_host = host_matrix_calloc(N);
    };

    ~gpuMatrix() {
        //dealloc
    };

    static void set_N(int n);

    void set_bit(unsigned int row, unsigned col) override;

    unsigned int get_bit(unsigned int row, unsigned col) override;

    bool add_mul(Matrix *left, Matrix *right) override;

    void allocate_device_matrix();

    void deallocate_device_matrix();

    static void allocate_tmp_matrix();

    static void deallocate_tmp_matrix();

    void transfer_to_gpu();

    void transfer_from_gpu();
};

class gpuMatricesEnv : public MatricesEnv {
public:
    gpuMatricesEnv() = default;

    ~gpuMatricesEnv() = default;

    void environment_preprocessing(const std::vector<Matrix *> &matrices) override;

    void environment_postprocessing(const std::vector<Matrix *> &matrices) override;
};


#endif //GPU_MATRIX_H
