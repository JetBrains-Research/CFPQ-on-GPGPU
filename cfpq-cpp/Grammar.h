
#ifndef CFPQ_CFPQ_H
#define CFPQ_CFPQ_H

#include <string>
#include <map>
#include <unordered_set>
#include <unordered_map>
#include <cmath>
#include <ctime>
#include <vector>
#include "Matrix.h"
#include "Graph.h"

using nonterminals_pair = std::pair<unsigned int, unsigned int>;

class Grammar {

public:
    explicit Grammar(const std::string &grammar_filename);

    virtual ~Grammar();

    template<class T1 = Matrix, class T2 = MatricesEnv>
    std::pair<unsigned int, unsigned int> intersection_with_graph(Graph &graph) {
        T2 *environment = new T2();
        vertices_count = graph.vertices_count;

        matrices.reserve(nonterminals_count);
        for (unsigned int i = 0; i < nonterminals_count; ++i) {
            matrices.push_back(new T1(vertices_count));
        }

        for (auto &edge : graph.edges) {
            for (unsigned int nonterm : terminal_to_nonterminals[edge.first]) {
                matrices[nonterm]->set_bit(edge.second.first, edge.second.second);
            }
        }

        clock_t begin_time = clock();
        environment->environment_preprocessing(matrices);
        clock_t begin_algo_time = clock();

        bool global_changed;
        unsigned long rules_size = rules.size();
        do {
            global_changed = false;
            for (uint32_t i = 0; i < rules_size; ++i) {
                auto c = std::get<0>(rules[i]);
                auto a = std::get<1>(rules[i]);
                auto b = std::get<2>(rules[i]);
                if (environment->changed_matrices[a] || environment->changed_matrices[b]) {
                    global_changed = true;
                    environment->add_mull(c, matrices[c], matrices[a], matrices[b]);
                }
            }
            environment->get_changed_matrices();
        } while (global_changed);

        clock_t end_algo_time = clock();
        environment->environment_postprocessing(matrices);
        clock_t end_time = clock();

        delete environment;
        double elapsed_secs = static_cast<double>(end_time - begin_time) / CLOCKS_PER_SEC;
        double overhead_secs =
                static_cast<double>((begin_algo_time - begin_time) + (end_time - end_algo_time)) / CLOCKS_PER_SEC;
        return std::make_pair(elapsed_secs, overhead_secs);
    }

    void print_results(const std::string &output_filename);

private:
    unsigned int nonterminals_count = 0;
    unsigned int vertices_count = 0;

    std::map<std::string, unsigned int> nonterminal_to_index;
    std::unordered_map<std::string, std::vector<int>> terminal_to_nonterminals;
    std::vector<std::tuple<uint32_t, uint32_t, uint32_t>> rules;
    std::vector<Matrix *> matrices;
};

#endif //CFPQ_CFPQ_H
