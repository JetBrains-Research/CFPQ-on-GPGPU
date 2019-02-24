
#include "multiplication.h"
#include "../../cfpq-cpp/Matrix.h"
#include "gpu_matrix.h"


int gpuMatrix::N;

uint32_t *gpuMatrix::tmp_matrix;

void gpuMatrix::set_N(int n) {
    N = n;
}

void gpuMatrix::set_bit(unsigned int row, unsigned col) {
    matrix_host[(row / 32) * cols(N) + col] |= 1U << (31 - (row % 32));
}

unsigned int gpuMatrix::get_bit(unsigned int row, unsigned col) {
    return matrix_host[(row / 32) * cols(N) + col] & 1U << (31 - (row % 32)) ? 1 : 0;
}

bool gpuMatrix::add_mul(Matrix *left, Matrix *right) {
    auto *A = dynamic_cast<gpuMatrix *>(left);
    auto *B = dynamic_cast<gpuMatrix *>(right);
    return MatrixMulAdd(A->matrix_device, B->matrix_device, this->matrix_device, N, tmp_matrix);
}

void gpuMatrix::allocate_device_matrix() {
    matrix_device = device_matrix_alloc(N);
}

void gpuMatrix::deallocate_device_matrix() {

}

void gpuMatrix::allocate_tmp_matrix() {
    gpuMatrix::tmp_matrix = device_matrix_alloc(N);
}

void gpuMatrix::deallocate_tmp_matrix() {

}

void gpuMatrix::transfer_to_gpu() {
    cpu2gpu(N, matrix_host, matrix_device);
}

void gpuMatrix::transfer_from_gpu() {
    gpu2cpu(N, matrix_device, matrix_host);
}

void gpuMatricesEnv::environment_preprocessing(const std::vector<Matrix *> &matrices) {
    for (Matrix *matrix: matrices) {
        auto *gpu_matrix = dynamic_cast<gpuMatrix *>(matrix);
        gpu_matrix->allocate_device_matrix();
        gpu_matrix->transfer_to_gpu();
    }
    gpuMatrix::allocate_tmp_matrix();
    synchronize();
};

void gpuMatricesEnv::environment_postprocessing(const std::vector<Matrix *> &matrices) {
    gpuMatrix::deallocate_tmp_matrix();
    for (Matrix *matrix: matrices) {
        auto *gpu_matrix = dynamic_cast<gpuMatrix *>(matrix);
        gpu_matrix->transfer_from_gpu();
    }
    synchronize();
    for (Matrix *matrix: matrices) {
        auto *gpu_matrix = dynamic_cast<gpuMatrix *>(matrix);
        gpu_matrix->deallocate_device_matrix();
    }
};