
#ifndef GPU_M4RUS_MATRIX
#define GPU_M4RUS_MATRIX

#include <vector>
#include "Matrix.h"
#include "gpuMemoryManagement.h"
#include "methodOf4RusBooleanSemiringGpu.h"
#include "Constants.h"

class MethodOf4RusMatricesEnv : public MatricesEnv {
public:

    int size_multiple_by_32;
    int cols;
    TYPE *extra_matrix_device;
    gpu_m4ri::Tables tables;

    MethodOf4RusMatricesEnv() {}

    ~MethodOf4RusMatricesEnv() override;

    void environment_preprocessing(const std::vector<Matrix *> &matrices) override;

    void environment_postprocessing(const std::vector<Matrix *> &matrices) override;
};

class MethodOf4RusMatrix : public Matrix {
public:
    
    int size_multiple_by_32;
    int cols;
    TYPE *matrix_host;
    TYPE *matrix_device;
    MethodOf4RusMatricesEnv *env;

    explicit MethodOf4RusMatrix(unsigned int n) : Matrix(n) {
        size_multiple_by_32 = n;

        if (n % SQUEEZE != 0) {
            int part = SQUEEZE - (n % SQUEEZE);
            size_multiple_by_32 += part;
        }
        cols = size_multiple_by_32 / SQUEEZE;

        matrix_host = gpu_m4ri::allocate_matrix_host(size_multiple_by_32, cols);
    }

    ~MethodOf4RusMatrix() {
        gpu_m4ri::delete_matrix_host(matrix_host);
    }

    void set_bit(unsigned int row, unsigned col) override;

    unsigned int get_bit(unsigned int row, unsigned col) override;

    bool add_mul(Matrix *left, Matrix *right) override;
};

#endif //#GPU_M4RUS_MATRIX
