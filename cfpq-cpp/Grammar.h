
#ifndef CFPQ_CFPQ_H
#define CFPQ_CFPQ_H

#include <string>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <chrono>
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
        rules_with_nonterminal.reserve(nonterminals_count);

        for (unsigned int i = 0; i < nonterminals_count; ++i) {
            rules_with_nonterminal.emplace_back();
            matrices.push_back(new T1(vertices_count));
        }

        for (unsigned int i = 0; i < rules.size(); ++i) {
            rules_with_nonterminal[rules[i].second.first].push_back(i);
            rules_with_nonterminal[rules[i].second.second].push_back(i);
            to_recalculate.insert(i);
        }

        for (auto &edge : graph.edges) {
            for (unsigned int nonterm : terminal_to_nonterminals[edge.first]) {
                matrices[nonterm]->set_bit(edge.second.first, edge.second.second);
            }
        }

        for (unsigned int nonterm : epsilon_nonterminals) {
            for (unsigned int i = 0; i < vertices_count; ++i) {
                matrices[nonterm]->set_bit(i, i);
            }
        }

        using namespace std::chrono;
        high_resolution_clock::time_point begin_time = high_resolution_clock::now();
        environment->environment_preprocessing(matrices);
        high_resolution_clock::time_point algorithm_begin_time = high_resolution_clock::now();

        while (!to_recalculate.empty()) {
            unsigned int rule_index = *to_recalculate.begin();
            to_recalculate.erase(to_recalculate.begin());
            unsigned int C = rules[rule_index].first;
            unsigned int A = rules[rule_index].second.first;
            unsigned int B = rules[rule_index].second.second;
            if (matrices[C]->add_mul(matrices[A], matrices[B])) {
                for (unsigned int changed_rule_index: rules_with_nonterminal[C]) {
                    to_recalculate.insert(changed_rule_index);
                }
            }
        }

        high_resolution_clock::time_point algorithm_end_time = high_resolution_clock::now();
        environment->environment_postprocessing(matrices);
        high_resolution_clock::time_point end_time = high_resolution_clock::now();

        delete environment;

        milliseconds algorithm_elapsed_secs = duration_cast<milliseconds>(algorithm_end_time - algorithm_begin_time);
        milliseconds elapsed_secs = duration_cast<milliseconds>(end_time - begin_time);
        return std::make_pair(static_cast<unsigned int>(elapsed_secs.count()),
                              static_cast<unsigned int>(algorithm_elapsed_secs.count()));
    }

    void print_results(const std::string &output_filename);
    unsigned int get_rules_size();
    const std::unordered_map<std::string, std::vector<int>>& get_nonterminal_from_terminal();
    void toArrays(unsigned int * grammar_body, unsigned long long * grammar_tail);
    void toArrays8(unsigned char *grammar_body, unsigned int *grammar_tail);

private:
    unsigned int nonterminals_count = 0;
    unsigned int vertices_count = 0;

    std::unordered_set<unsigned int> to_recalculate;
    std::vector<std::vector<unsigned int>> rules_with_nonterminal;
    std::map<std::string, unsigned int> nonterminal_to_index;
    std::vector<unsigned int> epsilon_nonterminals;
    std::unordered_map<std::string, std::vector<int>> terminal_to_nonterminals;
    std::vector<std::pair<unsigned int, nonterminals_pair>> rules;
    std::vector<Matrix *> matrices;
};

#endif //CFPQ_CFPQ_H
