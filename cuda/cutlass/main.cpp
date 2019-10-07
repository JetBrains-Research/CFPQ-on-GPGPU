//
// Created by DmiitriiJarosh on 24.09.2019.
//

#include <iostream>
#include "Grammar.h"
#include "Graph.h"
#include "CutlassMatrix.h"

int main(int argc, char *argv[]) {
    Grammar grammar = Grammar(argv[1]);

    size_t grammar_size = grammar.get_rules_size();
    auto grammar_body = new unsigned int[grammar_size];
    auto grammar_tail = new unsigned long long[grammar_size];
    grammar.toArrays(grammar_body, grammar_tail);

    printf("Grammar:\n");
    for (size_t i = 0; i < grammar_size; i++) {
        printf("%p ", grammar_body[i]);
        printf("%p\n", grammar_tail[i]);
    }

    Graph graph = Graph(argv[2]);

    unsigned int ** matrix = new unsigned int*[graph.vertices_count];
    for (unsigned int i = 0; i < graph.vertices_count; i++) {
        matrix[i] = new unsigned int[graph.vertices_count]{0};
    }
    graph.fillMatrix(matrix, grammar.get_nonterminal_from_terminal());

    printf("Matrix:\n");
    for (unsigned int i = 0; i < graph.vertices_count; i++) {
        for (unsigned int j = 0; j < graph.vertices_count; j++) {
            printf("%p ", matrix[i][j]);
        }
        printf("\n");
    }

    unsigned int ** mult_res = CutlassMatrix::MultMatrSquare(matrix, (int)graph.vertices_count, grammar_body, grammar_tail, grammar_size);

    printf("Result:\n");
    for (unsigned int i = 0; i < graph.vertices_count; i++) {
        for (unsigned int j = 0; j < graph.vertices_count; j++) {
            printf("%p ", mult_res[i][j]);
        }
        printf("\n");
    }

    return 0;
}
