import argparse
import sys
import time
from collections import defaultdict
from functools import wraps

import numpy as np

from math_utils import get_boolean_adjacency_matrices, remove_terminals
from parsing_utils import parse_graph, parse_grammar, products_set, products_list
from matmul import update_matrix_cpu, update_matrix_gpu
from matrix_utils import to_gpu, from_gpu, to_type, from_type


VERBOSE = False


def time_measure(f):
    @wraps(f)
    def inner(*args, **kwargs):
        time_start = time.time()
        out = f(*args, **kwargs)
        time_stop = time.time()
        return out, time_stop - time_start
    return inner


def main(grammar_file, graph_file, args):
    grammar, inverse_grammar = parse_grammar(grammar_file)
    graph, graph_size = parse_graph(graph_file)

    matrices = get_boolean_adjacency_matrices(grammar, inverse_grammar, graph, graph_size)
    remove_terminals(grammar, inverse_grammar)

    # supposing that matrices being altered in-place
    if args.type != 'bool':
        to_type(matrices, args.type)
    if not args.on_cpu:
        to_gpu(matrices)

    _, time_elapsed = iterate_on_grammar(grammar, inverse_grammar, matrices)

    if not args.on_cpu:
        from_gpu(matrices)
    if args.type != 'bool':
        from_type(matrices, args.type, graph_size)

    get_solution(matrices, args.output)
    print(int(1000 * time_elapsed + 0.5))


def get_solution(matrices, file=sys.stdout):
    if isinstance(file, str):
        file = open(file, 'wt')
    else:
        assert file is sys.stdout, f'Only allowed to print solution in file or stdout, not in {file}'
    
    for nonterminal, matrix in matrices.items():
        pairs = (np.argwhere(matrix) +  1).T
        print(nonterminal, end=' ', file=file)
        print(' '.join(map(lambda pair: ' '.join(pair), pairs.astype('str').tolist())), file=file)


@time_measure
def iterate_on_grammar_tracking(grammar, inverse_grammar, matrices):
    inverse_by_nonterm = defaultdict(set)
    for body, heads in inverse_grammar.items():
        assert type(body) is tuple, 'Left terminals in grammar: {}'.format(body)
        for head in heads:
            if body[0] != head:
                inverse_by_nonterm[body[0]].add((head, body))
            if body[1] != head:
                inverse_by_nonterm[body[1]].add((head, body))

    to_recalculate = products_set(grammar)
    while to_recalculate:
        head, body = to_recalculate.pop()
        assert type(body) is tuple, 'Body is either str or tuple, not {}'.format(type(body))
        is_changed = update_matrix(matrices, head, body)
        if not is_changed:
            continue
        for product in inverse_by_nonterm[head]:
            if product != (head, body):
                to_recalculate.add(product)


@time_measure
def iterate_on_grammar_naive(grammar, inverse_grammar, matrices):
    rules = products_list(grammar)

    is_changed_global = True
    while is_changed_global:
        is_changed_global = False
        for head, body in rules:
            is_changed = update_matrix(matrices, head, body)
            is_changed_global |= is_changed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('grammar', type=str, help='File with grammar in CNF')
    parser.add_argument('graph', type=str, help='Path to a directional graph')
    parser.add_argument('-o', '--output', type=str, help='Path to output file')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print logs into console')
    parser.add_argument('-c', '--on_cpu', action='store_true', help='Naive multiplication on CPU')
    parser.add_argument('-t', '--type', default='bool', choices=['bool', 'uint8', 'uint32'], help='Type for booleans to be packed in')
    parser.add_argument('-n', '--naive_cfpq', action='store_true', help='Iterate through all products instead of tracking updates matrices')
    args = parser.parse_args()

    if args.output is None:
        args.output = sys.stdout
    VERBOSE = args.verbose
    update_matrix = update_matrix_cpu if args.on_cpu else update_matrix_gpu
    iterate_on_grammar = iterate_on_grammar_naive if args.naive_cfpq else iterate_on_grammar_tracking

    main(args.grammar, args.graph, args=args)
