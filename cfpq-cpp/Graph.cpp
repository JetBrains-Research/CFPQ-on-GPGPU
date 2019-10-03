
#include <fstream>
#include <iostream>
#include "Graph.h"
#include "Grammar.h"

using std::string;
using std::ifstream;
using std::max;

Graph::Graph(const string &graph_filename) {
    auto graph_stream = ifstream(graph_filename, ifstream::in);
    unsigned int from, to;
    string terminal;
    while (graph_stream >> from >> terminal >> to) {
        edges.push_back({terminal, {from, to}});
        vertices_count = max(vertices_count, max(from, to) + 1);
    }
    graph_stream.close();
}

void Graph::fillMatrix(unsigned int ** matrix, const std::unordered_map<std::string, std::vector<int>> &terminal_to_nonterminals) {
    for (auto & edge : edges) {
        if (terminal_to_nonterminals.count(edge.first) == 0) {
            continue;
        }
        auto nonterminals = terminal_to_nonterminals.at(edge.first);
        unsigned int bool_vector = 0;
        for (auto nonterminal : nonterminals) {
            bool_vector |= Matrix::toBoolVector(nonterminal);
        }
        matrix[edge.second.first][edge.second.second] = bool_vector;
    }
}
