#include <iostream>
extern "C" {
#include "GraphBLAS.h"
}
#include "Grammar.h"
#include "Graph.h"
#include "Matrix.h"
#include "graphblas_matrix.h"
using namespace std;

int main(int argc, char *argv[]) {
    GrB_init(GrB_NONBLOCKING);
    Grammar grammar = Grammar(argv[1]);
    Graph graph = Graph(argv[2]);
    auto times = grammar.intersection_with_graph<GbMatrix>(graph);
    std::cout << times.first << ' ' << times.second << std::endl;
    grammar.print_results(argv[3]);
    GrB_finalize();
    return 0;
}
