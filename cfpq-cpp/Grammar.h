//
// Created by vkutuev on 14.02.19.
//

#ifndef CFPQ_CFPQ_H
#define CFPQ_CFPQ_H

#include <string>
#include <map>
#include <unordered_map>
#include <cmath>
#include <ctime>
#include <vector>
#include "Matrix.h"
#include "Graph.h"

using nonterms_pair_t = std::pair<unsigned int, unsigned int>;

class Grammar {

public:

    explicit Grammar(const std::string &grammar_filename);


    virtual ~Grammar();

    template <class T1 = Matrix, class T2 = MatricesEnv>
    unsigned int intersection_with_graph(Graph &graph) {
        T2 *utils = new T2();
        vertices_count = graph.vertices_count;
        matrices.reserve(nonterm_count);

        for (unsigned int i = 0; i < nonterm_count; ++i) {
            matrices.push_back(new T1(vertices_count));
        }

        for (auto &edge : graph.edges) {
            for (unsigned int nonterm : term_to_nonterms[edge.first]) {
                matrices[nonterm]->set_bit(edge.second.first, edge.second.second);
                matrices[nonterm]->changed_prev = true;
            }
        }

        clock_t begin_time = clock();

        utils->environment_preprocessing(matrices);

        while (true) {
            bool has_changed_global = false;
            for (auto &rule : rules) {
                if (matrices[rule.second.first]->changed_prev | matrices[rule.second.second]->changed_prev) {
                    bool has_changed = matrices[rule.first]->add_mul(matrices[rule.second.first],
                                                                     matrices[rule.second.second]);
                    matrices[rule.first]->changed = has_changed;
                    has_changed_global |= has_changed;
                }
            }
            if (!has_changed_global) {
                break;
            }
            for (unsigned int i = 0; i < nonterm_count; ++i) {
                matrices[i]->changed_prev = matrices[i]->changed;
                matrices[i]->changed = false;
            }
        }

        utils->environment_postprocessing(matrices);

        clock_t end_time = clock();
        delete utils;
        double elapsed_secs = static_cast<double>(end_time - begin_time) / CLOCKS_PER_SEC;
        return static_cast<unsigned int>(round(elapsed_secs * 1000 + 0.5));
    }


    void print_results(const std::string &output_filename);

private:
    unsigned int nonterm_count = 0;
    unsigned int vertices_count = 0;
    std::map<std::string, unsigned int> nonterm_to_index;
    std::unordered_map<std::string, std::vector<int>> term_to_nonterms;
    std::vector<std::pair<int, nonterms_pair_t>> rules;
    std::vector<Matrix *> matrices;
};

#endif //CFPQ_CFPQ_H
