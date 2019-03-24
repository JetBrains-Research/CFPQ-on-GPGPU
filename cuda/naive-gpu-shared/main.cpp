
#include <iostream>
#include "GpuMatrix.h"
#include "Grammar.h"

int main(int argc, char *argv[]) {
    Grammar grammar = Grammar(argv[1]);
    Graph graph = Graph(argv[2]);
    gpuMatrix::set_N(graph.vertices_count);
    auto times = grammar.intersection_with_graph<gpuMatrix, gpuMatricesEnv>(graph);
    std::cout << times.first << ' ' << times.second << std::endl;
    grammar.print_results(argv[3]);
    return 0;
}
