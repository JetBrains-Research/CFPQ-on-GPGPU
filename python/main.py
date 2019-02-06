import argparse
from parsing_utils import *
from math_utils import *

VERBOSE = False

def main(grammar_file, graph_file, args):
    grammar, inverse_grammar = parse_grammar(grammar_file)
    graph, graph_size = parse_graph(graph_file)

    matrices = get_boolean_adjacency_matrices(grammar, inverse_grammar, graph, graph_size)
    remove_terminals(grammar, inverse_grammar)



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
    parser.add_argument('-v', '--verbose', action='store_true', help='Print logs into console')
    args = parser.parse_args()
    VERBOSE = args.verbose

    main(args.grammar, args.graph, args=args)
