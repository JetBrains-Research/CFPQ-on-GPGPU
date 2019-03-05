
#include "m4ri_matrix.h"

using std::vector;
using std::get;
using std::tuple;

void M4riMatrix::set_bit(unsigned int row, unsigned col) {
    mzd_write_bit(m, row, col, 1);
}

unsigned int M4riMatrix::get_bit(unsigned int row, unsigned col) {
    return static_cast<unsigned int>(mzd_read_bit(m, row, col));
}

M4riMatrix::~M4riMatrix() {
    mzd_free(m);
}

bool M4riMatrix::add_mul(Matrix *left, Matrix *right) {
    auto A = dynamic_cast<M4riMatrix *>(left);
    auto B = dynamic_cast<M4riMatrix *>(right);
    mzd_t *c_old = mzd_copy(nullptr, m);
    m = mzd_sr_addmul_m4rm(m, A->m, B->m, 0);
    bool changed = mzd_equal(c_old, m) == 0;
    mzd_free(c_old);
    return changed;
}
