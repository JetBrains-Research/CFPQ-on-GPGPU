
#include "graphblas_matrix.h"

void GbMatrix::set_bit(unsigned int row, unsigned col) {
    GrB_Matrix_setElement_BOOL(m, true, row, col);
}

unsigned int GbMatrix::get_bit(unsigned int row, unsigned col) {
    bool x = false;
    GrB_Matrix_extractElement_BOOL(&x, m, row, col);
    return static_cast<unsigned int>(x);
}

GbMatrix::~GbMatrix() {
    GrB_Matrix_free(&m);
    GrB_Monoid_free(&monoid);
    GrB_Semiring_free(&semiring);
}

bool GbMatrix::add_mul(Matrix *left, Matrix *right) {
    auto A = dynamic_cast<GbMatrix *>(left);
    auto B = dynamic_cast<GbMatrix *>(right);
    GrB_Matrix m_old;
    GrB_Matrix_dup(&m_old, m);
    GrB_mxm(m, GrB_NULL, GrB_LOR, semiring, A->m, B->m, GrB_NULL);
    GrB_Index nvals_new, nvals_old;
    GrB_Matrix_nvals(&nvals_new, m);
    GrB_Matrix_nvals(&nvals_old, m_old);
    bool changed = nvals_new != nvals_old;
    GrB_Matrix_free(&m_old);
    return changed;
}
