import argparse
import sys
import time
from collections import defaultdict
import numpy as np

from math_utils import get_boolean_adjacency_matrices, remove_terminals
from parsing import parse_graph, parse_grammar, products_set, products_list
from matmul import update_matrix_cpu, update_matrix_gpu
from matrix_utils import to_gpu, from_gpu, to_type, from_type
from utils import time_measure


VERBOSE = False


def main(grammar_file, graph_file, args):
    grammar, inverse_grammar = parse_grammar(grammar_file)
    graph, graph_size = parse_graph(graph_file)

    matrices = get_boolean_adjacency_matrices(grammar, inverse_grammar, graph, graph_size, mat_type=args.type)
    remove_terminals(grammar, inverse_grammar)

    begin_time = time.time()
    if not args.on_cpu:
        to_gpu(matrices)

    _, iteration_time = iterate_on_grammar(grammar, inverse_grammar, matrices)

    if not args.on_cpu:
        from_gpu(matrices)
    end_time = time.time()
    if args.type != 'bool':
        from_type(matrices, args.type, graph_size)

    get_solution(matrices, args.output)
    print(int(1000 * iteration_time + 0.5), int(1000 * (end_time - begin_time) + 0.5))


def get_solution(matrices, file=sys.stdout):
    if isinstance(file, str):
        file = open(file, 'wt')
    else:
        assert file is sys.stdout, f'Only allowed to print solution in file or stdout, not in {file}'
    
    for nonterminal, matrix in matrices.items():
        pairs = np.argwhere(matrix).T
        print(nonterminal, end=' ', file=file)
        print(' '.join(map(lambda pair: ' '.join(pair), pairs.astype('str').tolist())), file=file)


@time_measure
def iterate_on_grammar_tracking(grammar, inverse_grammar, matrices):
    inverse_by_nonterm = defaultdict(set)
    for body, heads in inverse_grammar.items():
        for head in heads:
            if body[0] != head:
                inverse_by_nonterm[body[0]].add((head, body))
            if body[1] != head:
                inverse_by_nonterm[body[1]].add((head, body))

    to_recalculate = products_set(grammar)
    while to_recalculate:
        head, body = to_recalculate.pop()
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
    parser.add_argument('-t', '--type', default='bool', choices=['bool', 'sparse', 'uint8', 'uint32'], help='Type for booleans to be packed in')
    parser.add_argument('-n', '--naive_cfpq', action='store_true', help='Iterate through all products instead of tracking updates matrices')
    args = parser.parse_args()

    if args.output is None:
        args.output = sys.stdout
    assert not args.type == 'sparse' or args.on_cpu, 'Sparse matrices multiplication can be only on CPU. Pass "--on_cpu" / "-c" flag'
    VERBOSE = args.verbose
    update_matrix = update_matrix_cpu if args.on_cpu else update_matrix_gpu
    iterate_on_grammar = iterate_on_grammar_naive if args.naive_cfpq else iterate_on_grammar_tracking

    main(args.grammar, args.graph, args=args)
