//
// Created by DmiitriiJarosh on 24.09.2019.
//

#ifndef CFPQ_CUDA_CUTLASSMATRIX_H
#define CFPQ_CUDA_CUTLASSMATRIX_H


#include <Matrix.h>

class CutlassMatrix : public Matrix {
public:

//    ~CutlassMatrix() override;
//
//    void set_bit(unsigned int row, unsigned col) override;
//
//    unsigned int get_bit(unsigned int row, unsigned col) override;
//
//    bool add_mul(Matrix *left, Matrix *right) override;

    static unsigned int ** MultMatrSquare(unsigned int ** A, int size, unsigned int * grammar_body, unsigned long long * grammar_tail, int grammar_size);
};


#endif //CFPQ_CUDA_CUTLASSMATRIX_H
