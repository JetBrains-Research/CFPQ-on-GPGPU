
#ifndef CFPQ_MATRIX_H
#define CFPQ_MATRIX_H

#include <vector>

class Matrix {
public:
    explicit Matrix(unsigned int n) {};

    virtual ~Matrix() = default;

    virtual void set_bit(unsigned int row, unsigned col) = 0;

    virtual unsigned int get_bit(unsigned int row, unsigned col) = 0;

    virtual bool add_mul(Matrix *left, Matrix *right) = 0;
};

class MatricesEnv {
public:
    MatricesEnv() = default;

    virtual ~MatricesEnv() = default;

    virtual void environment_preprocessing(const std::vector<Matrix *> &matrices) {};

    virtual void environment_postprocessing(const std::vector<Matrix *> &matrices) {};
};


#endif //CFPQ_MATRIX_H
