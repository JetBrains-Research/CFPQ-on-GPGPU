from numba import cuda
import numpy as np
import math


threadsperblock = (16, 16)
tpb_x = threadsperblock[0]
tpb_y = threadsperblock[1]


def update_matrix_cpu(matrices, head, body, shared_memory=False):
    head_mat = matrices[head]
    body_first_mat, body_second_mat = matrices[body[0]], matrices[body[1]]
    if str(head_mat.dtype) == 'bool': 
        new_matrix = head_mat + body_first_mat.dot(body_second_mat)
        matrices[head] = new_matrix
        return np.any(new_matrix != head_mat)
    else:
        raise ValueError('CPU multiplication of matrices type {} is not supported'.format(head_mat.dtype))


def update_matrix_gpu(matrices, head, body, shared_memory=False):
    head_mat, body_first_mat = matrices[head], matrices[body[0]]
    body_second_mat = matrices[body[1]]
    is_changed = cuda.device_array((1,), dtype=bool)

    blockspergrid_x = int(math.ceil(body_first_mat.shape[0] / threadsperblock[0]))
    blockspergrid_y = int(math.ceil(body_second_mat.shape[1] / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    global size
    mat_type = str(head_mat.dtype)

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
