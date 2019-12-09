
#ifndef CFPQ_GRAPH_H
#define CFPQ_GRAPH_H

#include <string>
#include <vector>
#include <unordered_map>

using edge = std::pair<std::string, std::pair<unsigned int, unsigned int>>;

class Graph {
public:
    explicit Graph(const std::string &graph_filename);

    virtual ~Graph() = default;

    void fillMatrix(unsigned int ** matrix, const std::unordered_map<std::string, std::vector<int>> &terminal_to_nonterminals);

    std::vector<edge> edges;

    unsigned int vertices_count = 0;

    void
    fillMatrix(unsigned char **matrix,
               const std::unordered_map<std::string, std::vector<int>> &terminal_to_nonterminals);
};

#endif //CFPQ_GRAPH_H
