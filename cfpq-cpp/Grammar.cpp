//
// Created by vkutuev on 18.02.19.
//

#include "Grammar.h"
#include <sstream>
#include <fstream>

using std::istringstream;
using std::ifstream;
using std::ofstream;
using std::string;
using std::vector;


Grammar::Grammar(const string &grammar_filename) {

    auto chomsky_stream = ifstream(grammar_filename, ifstream::in);

    string line, tmp;

    while (getline(chomsky_stream, line)) {
        vector<string> terms;
        istringstream iss(line);
        while (iss >> tmp) {
            terms.push_back(tmp);
        }
        if (!nonterm_to_index.count(terms[0])) {
            nonterm_to_index[terms[0]] = nonterm_count++;
        }
        if (terms.size() == 2) {
            if (!term_to_nonterms.count(terms[1])) {
                term_to_nonterms[terms[1]] = {};
            }
            term_to_nonterms[terms[1]].push_back(nonterm_to_index[terms[0]]);
        } else if (terms.size() == 3) {
            if (!nonterm_to_index.count(terms[1])) {
                nonterm_to_index[terms[1]] = nonterm_count++;
            }
            if (!nonterm_to_index.count(terms[2])) {
                nonterm_to_index[terms[2]] = nonterm_count++;
            }
            rules.push_back({nonterm_to_index[terms[0]], {nonterm_to_index[terms[1]], nonterm_to_index[terms[2]]}});
        }
    }
    chomsky_stream.close();
}

Grammar::~Grammar() {
    for (unsigned int i = 0; i < nonterm_count; ++i)
        delete matrices[i];
}

void Grammar::print_results(const string &output_filename) {
    auto out_stream = ofstream(output_filename, ofstream::out);
    for (auto &nonterm : nonterm_to_index) {
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

