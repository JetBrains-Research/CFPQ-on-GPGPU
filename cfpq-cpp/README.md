# CFPQ C++ implementation

1. To implement cfpq using your own matrix multiplication algorithm you must inherit own matrix class from `Matrix` and call `Grammar::intersection_with_graph` parameterized by this class:
```cpp
grammar.intersection_with_graph<MyMatrixClass>(graph);
```
2. If your implementation requires preprocessing and post-processing of matrices and the environment, you must create your own environment class, inherited from `MatricesEnv` and call `Grammar::intersection_with_graph` parameterized by your matrix class and this class
```cpp
grammar.intersection_with_graph<MyMatrixClass, MyEnvClass>(graph);
```
* Method `MatricesEnv::environment_preprocessing` will be called before algorithm

* Method `MatricesEnv::environment_postprocessing` will be called after algorithm
3. `main.cpp` example:
```cpp
#include <iostream>
#include "Grammar.h"
#include "Graph.h"
#include "my_matrix.h"

int main(int argc, char *argv[]) {
    Grammar grammar = Grammar(argv[1]);
    Graph graph = Graph(argv[2]);
    std::pair<unsigned int, unsigned int> times = grammar.intersection_with_graph<M4riMatrix>(graph);
    std::cout << times.first << ' ' << times.second << std::endl;
    grammar.print_results(argv[3]);
    return 0;
}
```
