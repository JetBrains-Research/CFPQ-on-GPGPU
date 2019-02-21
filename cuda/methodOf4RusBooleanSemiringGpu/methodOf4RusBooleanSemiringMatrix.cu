
#include "methodOf4RusBooleanSemiringMatrix.h"

#define MAXIMUM_PARTITION 32

MethodOf4RusMatricesEnv::~MethodOf4RusMatricesEnv() {
    tables.free();
    delete_matrix_device(extra_matrix_device);
}

void MethodOf4RusMatricesEnv::environment_preprocessing
                                (const std::vector<Matrix *> &matrices) {
    if (!matrices.empty()) {
        auto * tmp = dynamic_cast<MethodOf4RusMatrix *>(matrices[0]);
        cols = tmp->cols;
        size_multiple_by_32 = tmp->size_multiple_by_32;
    } else {
        //raise Error;
        return;
    }

    tables.initialize(size_multiple_by_32, cols, MAXIMUM_PARTITION);
    for (auto &m : matrices) {
        auto *A = dynamic_cast<MethodOf4RusMatrix *>(m);
        copy_host_to_device_async(A->matrix_host, A->matrix_device, 
                  A->size_multiple_by_32 * A->cols);
    }    
    synchronize_with_gpu();
}

void MethodOf4RusMatricesEnv::environment_postprocessing
                                (const std::vector<Matrix *> &matrices) {
    for (auto &m : matrices) {
        auto *A = dynamic_cast<MethodOf4RusMatrix *>(m);
        copy_device_to_host_async(A->matrix_device, A->matrix_host, 
                  A->size_multiple_by_32 * A->cols);
    }
    synchronize_with_gpu();
}

void MethodOf4RusMatrix::set_bit(unsigned int row, unsigned col) {
    matrix_host[row * cols + (col / SQUEEZE)] |= 
                                    1 << (31 - (col % SQUEEZE));
}

unsigned int MethodOf4RusMatrix::get_bit(unsigned int row, unsigned col) {
    auto bit = matrix_host [row * cols + (col / SQUEEZE)] &
                                     1 << (31 - (col % SQUEEZE));
    return static_cast<unsigned int>(bit);
}

bool MethodOf4RusMatrix::add_mul(Matrix *left, Matrix *right) {
    auto *A = dynamic_cast<MethodOf4RusMatrix *>(left);
    auto *B = dynamic_cast<MethodOf4RusMatrix *>(right);
    
    if(A->matrix_device == matrix_device) {
        // we need extra matrix
        copy_device_to_device_sync(A->matrix_device, env->extra_matrix_device,
                                                        size_multiple_by_32 * cols);
        return wrapper_method_of_4rus_bool_semiring(env->extra_matrix_device, 
            B->matrix_device, matrix_device, env->tables, size_multiple_by_32, cols);    
    } else {
        return wrapper_method_of_4rus_bool_semiring(A->matrix_device, 
            B->matrix_device, matrix_device, env->tables, size_multiple_by_32, cols);
        }
}
