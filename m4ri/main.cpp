#include <iostream>
#include "Grammar.h"
#include "Graph.h"
#include "Matrix.h"
#include "m4ri_matrix.h"

int main(int argc, char *argv[]) {
    Grammar grammar = Grammar(argv[1]);
    Graph graph = Graph(argv[2]);
    auto times = grammar.intersection_with_graph<M4riMatrix>(graph);
    std::cout << times.first << ' ' << times.second << std::endl;
    grammar.print_results(argv[3]);

    return 0;
}
