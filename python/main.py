import argparse
import sys
from parsing_utils import *
from math_utils import *
from functools import wraps
import time


VERBOSE = False


def time_measure(f):
    @wraps(f)
    def inner(*args, **kwargs):
        time_start = time.time()
        out = f(*args, **kwargs)
        time_stop = time.time()
        return out, time_stop - time_start


def main(grammar_file, graph_file, args):
    grammar, inverse_grammar = parse_grammar(grammar_file)
    graph, graph_size = parse_graph(graph_file)

    matrices = get_boolean_adjacency_matrices(grammar, inverse_grammar, graph, graph_size)
    remove_terminals(grammar, inverse_grammar)

    # supposing that matrices being altered in-place
    _, time_elapsed = iterate_on_grammar(grammar, inverse_grammar, matrices)
    get_solution(args.output)
    print(int(1000 * (time_elapsed + 0.0005)))


def get_solution(file=sys.stdout):
    if type(file) is str:
        file = open(file, 'wt')
    else
        assert file is sys.stdout, f'Only allowed to print solution in file or stdout, not in {file}'
    
    for nonterminal, matrix in matrices.items():
        xs, ys = np.where(matrix)
        # restoring true vertices numbers
        xs += 1
        ys += 1
        pairs = np.vstack((xs, ys)).T
        print(nonterminal, end=' ', file=file)
        print(' '.join(map(lambda pair: ' '.join(pair), pairs.astype('str').tolist())), file=file)


@time_measure
def iterate_on_grammar(grammar, inverse_grammar, matrices, shared_memory=False):
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
        is_changed = ... # Awesome multiplication methods here
        if not is_changed:
            continue
        for product in inverse_by_nonterm[head]:
            if product != (head, body):
                to_recalculate.add(product)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('grammar', type=str, help='File with grammar in CNF')
    parser.add_argument('graph', type=str, help='Path to a directional graph')
    parser.add_argument('--output', type=str, default=None, help='Path to output file', required=False)
    parser.add_argument('-v', '--verbose', action='store_true', help='Print logs into console')
    args = parser.parse_args()
    if args.output is None:
        args.output = sys.stdout
    VERBOSE = args.verbose

    main(args.grammar, args.graph, args=args)
