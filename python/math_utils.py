import math
import numpy as np


size = 1


def remove_terminals(grammar, inverse_grammar):
    terminals = [body for body in inverse_grammar.keys() if type(body) is str]
    for terminal in terminals:
        heads = inverse_grammar.pop(terminal)
        for head in heads:
            grammar[head].remove(terminal)
    return len(terminals)


def set_bit_bool(matrix, row, col):
    matrix[row, col] = True


def set_bit_uint(matrix, row, col):
    matrix[row, col // size] |= 1 << (size - (col % size) - 1)


def get_boolean_adjacency_matrices(grammar, inv_grammar, graph, graph_size, mat_type='bool'):
    global size
    if mat_type == 'bool':
        set_bit = set_bit_bool
        matrices = {i: np.zeros((graph_size, graph_size), dtype=np.bool) for i in grammar}
    elif mat_type in ['uint8', 'uint32']:
        size = 8 if mat_type == 'uint8' else 32
        set_bit = set_bit_uint
        matrices = {i: np.zeros((graph_size, math.ceil(graph_size / size)), dtype=mat_type) for i in grammar}
    else:
        raise ValueError('Unsupported type {}'.format(mat_type))

    for row, verts in graph.items():
        for col, value in verts.items():
            if value in inv_grammar:
                for nonterminal in inv_grammar[value]:
                    set_bit(matrices[nonterminal], row, col)
    return matrices
