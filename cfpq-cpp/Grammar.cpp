
#include "Grammar.h"
#include <sstream>
#include <fstream>

using std::istringstream;
using std::ifstream;
using std::ofstream;
using std::string;
using std::vector;

const unsigned int hash_base0 = 11;
const unsigned int hash_base1 = 13;
const unsigned long long hash_mod = 1000000007;


Grammar::Grammar(const string &grammar_filename) {

    auto chomsky_stream = ifstream(grammar_filename, ifstream::in);

    string line, tmp;

    while (getline(chomsky_stream, line)) {
        vector<string> terms;
        istringstream iss(line);
        while (iss >> tmp) {
            terms.push_back(tmp);
        }
        if (!nonterminal_to_index.count(terms[0])) {
            nonterminal_to_index[terms[0]] = nonterminals_count++;
        }
        if (terms.size() == 1) {
            epsilon_nonterminals.push_back(nonterminal_to_index[terms[0]]);
        } else if (terms.size() == 2) {
            if (!terminal_to_nonterminals.count(terms[1])) {
                terminal_to_nonterminals[terms[1]] = {};
            }
            terminal_to_nonterminals[terms[1]].push_back(nonterminal_to_index[terms[0]]);
        } else if (terms.size() == 3) {
            if (!nonterminal_to_index.count(terms[1])) {
                nonterminal_to_index[terms[1]] = nonterminals_count++;
            }
            if (!nonterminal_to_index.count(terms[2])) {
                nonterminal_to_index[terms[2]] = nonterminals_count++;
            }
            rules.push_back(
                    {nonterminal_to_index[terms[0]], {nonterminal_to_index[terms[1]], nonterminal_to_index[terms[2]]}});
        }
    }
    chomsky_stream.close();
}

Grammar::~Grammar() {
    unsigned int size = nonterminals_count < matrices.size() ? nonterminals_count : matrices.size();
    for (unsigned int i = 0; i < size; ++i)
        delete matrices[i];
}

unsigned int Grammar::get_rules_size() {
    return rules.size();
}

void Grammar::toArrays(unsigned int * grammar_body, unsigned long long * grammar_tail) {
    for (size_t i = 0; i < rules.size(); i++) {
        grammar_body[i] = Matrix::toBoolVector(rules[i].first);
        grammar_tail[i] = (((unsigned long long)Matrix::toBoolVector(rules[i].second.first)) << 32) | (unsigned long long)Matrix::toBoolVector(rules[i].second.second);
    }
}

void Grammar::toArrays8(unsigned char * grammar_body, unsigned int * grammar_tail) {
    for (size_t i = 0; i < rules.size(); i++) {
        grammar_body[i] = Matrix::toBoolVector8(rules[i].first);
        grammar_tail[i] = (((unsigned int)Matrix::toBoolVector8(rules[i].second.first)) << 8) | (unsigned int)Matrix::toBoolVector8(rules[i].second.second);
    }
}

const std::unordered_map<std::string, std::vector<int>>& Grammar::get_nonterminal_from_terminal() {
    return terminal_to_nonterminals;
}

void Grammar::print_results(const string &output_filename) {
    auto out_stream = ofstream(output_filename, ofstream::out);
    for (auto &nonterm : nonterminal_to_index) {
        unsigned long long hash = 0;
        unsigned long count = 0;
        for (unsigned int row = 0; row < vertices_count; ++row) {
            unsigned long long line_hash = 0;
            for (unsigned int col = 0; col < vertices_count; ++col) {
                line_hash *= hash_base0;
                if (matrices[nonterm.second]->get_bit(row, col) != 0) {
                    ++count;
                    ++line_hash;
                }
                line_hash %= hash_mod;
            }
            hash = (hash * hash_base1 + line_hash) % hash_mod;
        }
        out_stream << nonterm.first << ' ' << hash << ' ' << count << std::endl;
    }
    out_stream.close();
}
