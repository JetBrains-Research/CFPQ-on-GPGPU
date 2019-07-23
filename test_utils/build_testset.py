"""Scan data folder and create csv file with tests"""

from os import walk, listdir
from os.path import join
from argparse import ArgumentParser

def main(data_folder):
    paths = [root
        for root, dirnames, filenames in walk(data_folder)
        if all([i in dirnames for i in ['Grammars', 'Matrices']])
    ]
    tests = []
    cache = {}
    for path in paths:
        grammars = listdir(join(path, 'Grammars'))
        matrices = listdir(join(path, 'Matrices'))
        for grammar in grammars:
            for matrix in matrices:
                name = f'{grammar[:-4]}:{matrix[:-4]}'
                path_to_grammar = join(path, 'Grammars', grammar)
                path_to_matrix = join(path, 'Matrices', matrix)
                if path_to_matrix not in cache:
                    with open(path_to_matrix, 'r') as f:
                        n_vert = 0
                        n_edge = 0
                        for line in f:
                            n_edge += 1
                            u, _, v = line.strip().split(' ')
                            n_vert = max(n_vert, max(int(u), int(v)))
                        n_vert += 1
                    cache[path_to_matrix] = (n_vert, n_edge)
                n_vert, n_edge = cache[path_to_matrix]
                if path_to_grammar not in cache:
                    with open(path_to_grammar, 'r') as f:
                        n_grammar = sum(1 for _ in f)
                    cache[path_to_grammar] = n_grammar
                n_grammar = cache[path_to_grammar]
                tests.append([name, path_to_grammar, n_grammar, path_to_matrix, n_vert, n_edge])
    with open('tests.csv', 'w') as f:
        f.write('Name,Grammar,Number of rules,Matrix, Number of vertices, Number of edges\n')
        for test in tests:
            f.write(','.join(map(str, test)) + '\n')

if __name__ == '__main__':
    arg_parser = ArgumentParser()
    arg_parser.add_argument('data', type=str, help='Path to main folder with tests')
    args = arg_parser.parse_args()
    main(args.data)
