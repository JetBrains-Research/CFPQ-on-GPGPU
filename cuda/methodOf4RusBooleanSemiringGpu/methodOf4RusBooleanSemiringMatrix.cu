
#include "methodOf4RusBooleanSemiringMatrix.h"
#include "gpuMemoryManagement.h"
#include <stdexcept>

MethodOf4RusMatricesEnv::~MethodOf4RusMatricesEnv() { }

void MethodOf4RusMatricesEnv::environment_preprocessing
                                (const std::vector<Matrix *> &matrices) {
    if (!matrices.empty()) {
        auto * tmp = dynamic_cast<MethodOf4RusMatrix *>(matrices[0]);
        cols = tmp->cols;
        size_multiple_by_32 = tmp->size_multiple_by_32;
    } else {
        throw std::runtime_error("Error: Empty vector of matrices");
    }

    extra_matrix_device = gpu_m4ri::allocate_matrix_device(size_multiple_by_32, cols);

    tables.initialize(size_multiple_by_32, cols, MAXIMUM_PARTITION);
    for (auto &m : matrices) {
        auto *A = dynamic_cast<MethodOf4RusMatrix *>(m);
        A->env = this;
        A->matrix_device = gpu_m4ri::allocate_matrix_device(size_multiple_by_32, cols);
        gpu_m4ri::copy_host_to_device_async(A->matrix_host, A->matrix_device, 
                  A->size_multiple_by_32 * A->cols);
    }
    gpu_m4ri::synchronize_with_gpu();
}

void MethodOf4RusMatricesEnv::environment_postprocessing
                                (const std::vector<Matrix *> &matrices) {
    for (auto &m : matrices) {
        auto *A = dynamic_cast<MethodOf4RusMatrix *>(m);
        gpu_m4ri::copy_device_to_host_async(A->matrix_device, A->matrix_host, 
                  A->size_multiple_by_32 * A->cols);
        gpu_m4ri::delete_matrix_device(A->matrix_device);
    }
    tables.free();
    gpu_m4ri::delete_matrix_device(extra_matrix_device);
    gpu_m4ri::synchronize_with_gpu();
}

void MethodOf4RusMatrix::set_bit(unsigned int row, unsigned col) {
    matrix_host[row * cols + (col / SQUEEZE)] |= 
                                    1U << (31 - (col % SQUEEZE));
}

unsigned int MethodOf4RusMatrix::get_bit(unsigned int row, unsigned col) {
    return (matrix_host [row * cols + (col / SQUEEZE)] &
                                     1U << (31 - (col % SQUEEZE))) ? 1 : 0;
}

bool MethodOf4RusMatrix::add_mul(Matrix *left, Matrix *right) {
    auto *A = dynamic_cast<MethodOf4RusMatrix *>(left);
    auto *B = dynamic_cast<MethodOf4RusMatrix *>(right);
    
    if(A->matrix_device == matrix_device) { 
        // we need extra matrix
        gpu_m4ri::copy_device_to_device_sync(A->matrix_device, env->extra_matrix_device,
                                                        size_multiple_by_32 * cols);
        return gpu_m4ri::wrapper_method_of_4rus_bool_semiring(env->extra_matrix_device, 
            B->matrix_device, matrix_device, env->tables, size_multiple_by_32, cols);    
    } else {
        return gpu_m4ri::wrapper_method_of_4rus_bool_semiring(A->matrix_device, 
            B->matrix_device, matrix_device, env->tables, size_multiple_by_32, cols);
        }
}
