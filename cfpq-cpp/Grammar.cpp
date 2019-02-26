
#include "Grammar.h"
#include <sstream>
#include <fstream>

using std::istringstream;
using std::ifstream;
using std::ofstream;
using std::string;
using std::vector;
using std::make_tuple;

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
        if (terms.size() == 2) {
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
            rules.emplace_back(nonterminal_to_index[terms[0]], nonterminal_to_index[terms[1]],
                               nonterminal_to_index[terms[2]]);
        }
    }
    chomsky_stream.close();
}

Grammar::~Grammar() {
    for (auto &matrix: matrices)
        delete matrix;
}

void Grammar::print_results(const string &output_filename) {
    auto out_stream = ofstream(output_filename, ofstream::out);
    for (auto &nonterm : nonterminal_to_index) {
        out_stream << nonterm.first;
        for (unsigned int row = 0; row < vertices_count; ++row) {
            for (unsigned int col = 0; col < vertices_count; ++col) {
                if (matrices[nonterm.second]->get_bit(row, col) != 0)
                    out_stream << " " << row << " " << col;
            }
        }
        out_stream << std::endl;
    }
    out_stream.close();
}
