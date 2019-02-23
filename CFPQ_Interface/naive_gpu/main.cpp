#include <iostream>
#include "Grammar.h"
#include "Graph.h"
#include "gpu_matrix.h"
#include "multiplication.h"

int main(int argc, char *argv[]) {
    Grammar grammar = Grammar(argv[1]);
    Graph graph = Graph(argv[2]);
    initialize(graph.vertices_count);
    std::cout << grammar.intersection_with_graph<gpuMatrix, gpuMatricesEnv>(graph);
    grammar.print_results(argv[3]);
    return 0;
}