
#ifndef CFPQ_MATRIX_H
#define CFPQ_MATRIX_H

#include <vector>

class Matrix {
public:
    explicit Matrix(unsigned int n) {};

    virtual ~Matrix() = default;

    virtual void set_bit(unsigned int row, unsigned col) = 0;

    virtual unsigned int get_bit(unsigned int row, unsigned col) = 0;
};

class MatricesEnv {
public:
    MatricesEnv() = default;

    virtual ~MatricesEnv() {
        delete[] changed_matrices;
    }

    virtual void environment_preprocessing(const std::vector<Matrix *> &matrices) {
        changed_matrices = new bool[matrices.size()];
    };

    virtual void environment_postprocessing(const std::vector<Matrix *> &matrices) {};

    virtual void add_mull(u_int32_t i, Matrix *A, Matrix *B, Matrix *C) = 0;

    virtual void get_changed_matrices() = 0;

    bool *changed_matrices = nullptr;
};


#endif //CFPQ_MATRIX_H
