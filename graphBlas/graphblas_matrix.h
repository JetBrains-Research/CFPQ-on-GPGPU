#ifndef GRAPHBLAS_MATRIX_H
#define GRAPHBLAS_MATRIX_H

#include "Grammar.h"
extern "C" {
#include "GraphBLAS.h"
};

class GbMatrix : public Matrix {
public:
    explicit GbMatrix(unsigned int n) : Matrix(n) {
        GrB_Info info;
        info = GrB_Matrix_new(&m, GrB_BOOL, n, n);
        info = GrB_Monoid_new_BOOL(&monoid, GrB_LOR, false);
        info = GrB_Semiring_new(&semiring, monoid, GrB_LAND);
    }

    ~GbMatrix() override;

    void set_bit(unsigned int row, unsigned col) override;

    unsigned int get_bit(unsigned int row, unsigned col) override;

    bool add_mul(Matrix *left, Matrix *right) override;

private:
    GrB_Matrix m;
    GrB_Monoid monoid;
    GrB_Semiring semiring;
};

#endif // GRAPHBLAS_MATRIX_H
