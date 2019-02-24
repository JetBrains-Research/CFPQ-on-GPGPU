#include <iostream>
#include "gpu_matrix.h"
#include "multiplication.h"
#include "../../cfpq-cpp/Grammar.h"

int main(int argc, char *argv[]) {
    Grammar grammar = Grammar(argv[1]);
    Graph graph = Graph(argv[2]);
    gpuMatrix::set_N(graph.vertices_count);
    std::cout << grammar.intersection_with_graph<gpuMatrix, gpuMatricesEnv>(graph);
    grammar.print_results(argv[3]);
    return 0;
}