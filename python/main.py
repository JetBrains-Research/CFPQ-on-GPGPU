import argparse
from parsing_utils import *
from math_utils import *

VERBOSE = False

def main(grammar_file, graph_file, args):
    ...


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('grammar', type=str, help='File with grammar in CNF')
    parser.add_argument('graph', type=str, help='Path to a directional graph')
    parser.add_argument('-v', '--verbose', action='store_true', help='Print logs into console')
    args = parser.parse_args()
    VERBOSE = args.verbose

    main(args.grammar, args.graph, args=args)
