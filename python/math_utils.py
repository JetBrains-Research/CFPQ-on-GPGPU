import numpy as np
import time
from functools import wraps


def time_measure(f):
    @wraps(f)
    def inner(*args, **kwargs):
        time_start = time.time()
        out = f(*args, **kwargs)
        time_stop = time.time()
        return out, time_stop - time_start
    return inner


def remove_terminals(grammar, inverse_grammar):
    terminals = [body for body in inverse_grammar.keys() if type(body) is str]
    for terminal in terminals:
        heads = inverse_grammar.pop(terminal)
        for head in heads:
            grammar[head].remove(terminal)
    return len(terminals)


def get_boolean_adjacency_matrices(grammar, inv_grammar, graph, graph_size):
    size = graph_size
    matrices = {i: np.zeros((size, size), dtype=np.bool) for i in grammar}
    for row, verts in graph.items():
        for col, value in verts.items():
            if value in inv_grammar:
                for nonterminal in inv_grammar[value]:
                    matrices[nonterminal][row, col] = True
    return matrices
