
#ifndef GPU_MATRIX_H
#define GPU_MATRIX_H

#include "Grammar.h"

class gpuMatrix : public Matrix {
public:
    explicit gpuMatrix(unsigned int n);

    ~gpuMatrix();

    void set_bit(unsigned int row, unsigned col);

    unsigned int get_bit(unsigned int row, unsigned col);

    bool add_mul(Matrix *left, Matrix *right);

};

class gpuMatricesEnv : public MatricesEnv {
public:
    gpuMatricesEnv();

    ~gpuMatricesEnv();

    void environment_preprocessing(const std::vector<gpuMatrix *> &matrices);

    void environment_postprocessing(const std::vector<gpuMatrix *> &matrices);
};


#endif //GPU_MATRIX_H
