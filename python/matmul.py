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
    if str(head_mat.dtype) == 'bool':
        matmul_bool[blockspergrid, threadsperblock](body_first_mat, body_second_mat, head_mat, is_changed)
        if not is_changed[0]:
            return False
        matrices[head] = head_mat
        return True
    else:
        raise ValueError('GPU multiplication of matrices type {} is not supported'.format(head_mat.dtype))


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
