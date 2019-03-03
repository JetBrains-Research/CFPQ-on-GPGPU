import math

import numpy as np
import time
from numba import cuda
from scipy.sparse import csr_matrix

from utils import time_measure


threads_size = int(np.sqrt(cuda.get_current_device().MAX_THREADS_PER_BLOCK))
threadsperblock = (threads_size, threads_size)
blockspergrid = None
tpb_x = threadsperblock[0]
tpb_y = threadsperblock[1]
size = 1
matmul_method = None


def update_matrix_cpu(matrices, head, body):
    head_mat = matrices[head]
    body_first_mat, body_second_mat = matrices[body[0]], matrices[body[1]]
    mat_type = 'sparse' if isinstance(head_mat, csr_matrix) else str(head_mat.dtype)
    if mat_type in ['bool', 'sparse']:
        new_matrix = head_mat + body_first_mat.dot(body_second_mat)
        matrices[head] = new_matrix
        comparison = new_matrix != head_mat
        return np.any(comparison) if mat_type == 'bool' else (comparison).nnz != 0
    else:
        raise ValueError('CPU multiplication of matrices type {} is not supported'.format(mat_type))


@time_measure
def initialize_and_compile(mat_size, mat_type):
    mat = cuda.to_device(np.zeros((mat_size, mat_size), dtype=mat_type))
    blockspergrid = tuple(int(math.ceil(mat_size / threadsperblock[i])) for i in (0, 1))
    is_changed = cuda.device_array((1,), dtype=bool)

    global size, matmul_method
    if mat_type == 'bool':
        matmul_method = matmul_bool[blockspergrid, threadsperblock]
    elif mat_type == 'uint8':
        matmul_method = matmul_uint[blockspergrid, threadsperblock]
        size = 8
    elif mat_type == 'uint32':
        matmul_method = matmul_uint[blockspergrid, threadsperblock]
        size = 32
    else:
        raise ValueError('GPU multiplication of matrices type {} is not supported'.format(mat_type))

    matmul_method(mat, mat, mat, is_changed)


def update_matrix_gpu(matrices, head, body):
    head_mat, body_first_mat = matrices[head], matrices[body[0]]
    body_second_mat = matrices[body[1]]
    is_changed = cuda.device_array((1,), dtype=bool)

    matmul_method(body_first_mat, body_second_mat, head_mat, is_changed)

    if is_changed[0]:
        matrices[head] = head_mat
        return True
    else:
        return False


@cuda.jit
def matmul_bool(A, B, C, is_changed):
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = False
        for k in range(A.shape[1]):
            tmp = tmp or (A[row, k] and B[k, col])
        if tmp and not C[row, col]:
            is_changed[0] = True
            C[row, col] = tmp


@cuda.jit
def matmul_uint(A, B, C, is_changed):
    row, col = cuda.grid(2)
    if row >= C.shape[0] or col >= C.shape[1]:
        return
    value = 0
    for k in range(A.shape[1]):
        cur_value_A = A[row, k]
        for j in range(size - 1, -1, -1):
            if cur_value_A & 1:
                value |= (B[k * size + j, col])
            cur_value_A >>= 1
    old_value = C[row, col]
    new_value = old_value | value
    if new_value != old_value:
        C[row, col] = new_value
        is_changed[0] = True
