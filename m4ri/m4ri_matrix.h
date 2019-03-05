
#ifndef CFPQ_M4RI_MATRIX_H
#define CFPQ_M4RI_MATRIX_H

#include "Grammar.h"
#include "vector"
#include <m4ri/m4ri.h>

class M4riMatrix : public Matrix {
public:
    explicit M4riMatrix(unsigned int n) : Matrix(n) {
        m = mzd_init(n, n);
    }

    ~M4riMatrix() override;

    void set_bit(unsigned int row, unsigned col) override;

    unsigned int get_bit(unsigned int row, unsigned col) override;

    bool add_mul(Matrix *left, Matrix *right) override;

private:
    mzd_t *m = nullptr;
};

#endif //CFPQ_M4RI_MATRIX_H
