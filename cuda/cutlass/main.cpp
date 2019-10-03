//
// Created by DmiitriiJarosh on 24.09.2019.
//

#include <iostream>
#include "Grammar.h"
#include "Graph.h"
#include "CutlassMatrix.h"

bool matrixCompare(unsigned int ** matrix1, unsigned int ** matrix2, unsigned int size) {
    for (unsigned int i = 0; i < size; i++) {
        for (unsigned int j = 0; j < size; j++) {
            if (matrix1[i][j] != matrix2[i][j]) {
                return false;
            }
        }
    }
    return true;
}

void concat(unsigned int ** matrix, unsigned int ** mult_res, int size) {
    for (unsigned int i = 0; i < size; i++) {
        for (unsigned int j = 0; j < size; j++) {
            matrix[i][j] |= mult_res[i][j];
        }
    }
}

int main(int argc, char *argv[]) {
    //TestCutlassGemm(2,2,2,1,0);
    std::cout << argv[1] << std::endl;
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
    std::cout << graph.vertices_count << std::endl;

    unsigned int ** matrixCopy = new unsigned int*[graph.vertices_count];
    for (unsigned int i = 0; i < graph.vertices_count; i++) {
        matrixCopy[i] = new unsigned int[graph.vertices_count]{0};
    }

    while (!matrixCompare(matrix, matrixCopy, graph.vertices_count)) {
        for (unsigned int i = 0; i < graph.vertices_count; i++) {
            for (unsigned int j = 0; j < graph.vertices_count; j++) {
                matrixCopy[i][j] = matrix[i][j];
            }
        }

        unsigned int ** mult_res = CutlassMatrix::MultMatr(matrix, matrix, (int)graph.vertices_count, grammar_body, grammar_tail, grammar_size);

        concat(matrix, mult_res, (int)graph.vertices_count);
        for (unsigned int i = 0; i < graph.vertices_count; i++) {
            for (unsigned int j = 0; j < graph.vertices_count; j++) {
                printf("%p ", matrix[i][j]);
            }
            printf("\n");
        }
        printf("------------------------------------------------\n");
    }
    return 0;
}
