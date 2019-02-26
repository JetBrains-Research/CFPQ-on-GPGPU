from numba import cuda
import numpy as np


def to_gpu(matrices):
    stream = cuda.stream()
    for nonterminal, matrix in matrices.items():
        matrices[nonterminal] = cuda.to_device(matrix, stream=stream)
    stream.synchronize()


def from_gpu(matrices):
    stream = cuda.stream()
    for nonterminal, matrix in matrices.items():
        matrices[nonterminal] = matrix.copy_to_host(stream=stream)
    stream.synchronize()


def to_type(matrices, mat_type):
    if mat_type in [np.uint8, 'uint8']:
        for key, matrix in matrices.items():
            matrix = np.packbits(matrix, axis=-1)
            matrices[key] = matrix
    elif mat_type in [np.uint32, 'uint32']:
        for key, matrix in matrices.items():
            matrix = np.pad(matrix, [(0, 0), (0, (32 - matrix.shape[1] % 32) % 32)], 'constant').astype(np.uint32)
            packed_matrix = sum(matrix[:, i::32] << (31 - i) for i in range(32))
            matrices[key] = packed_matrix
    else:
        raise ValueError('Casting to type {} is not supported yet'.format(mat_type))


def from_type(matrices, mat_type, nodes_amount):
    if mat_type in [np.uint8, 'uint8']:
        for key, matrix in matrices.items():
            matrix = np.unpackbits(matrix, axis=-1)[:nodes_amount, :nodes_amount]
            matrices[key] = matrix
    elif mat_type in [np.uint32, 'uint32']:
        for key, matrix in matrices.items():
            full_matrix = np.zeros((matrix.shape[0], matrix.shape[1] * 32), dtype=bool)
            for i in range(32):
                full_matrix[:, i::32] = (matrix >> (31 - i)) & 1
            matrices[key] = full_matrix[:nodes_amount, :nodes_amount]
    else:
        raise ValueError('Casting to type {} is not supported yet'.format(mat_type))
