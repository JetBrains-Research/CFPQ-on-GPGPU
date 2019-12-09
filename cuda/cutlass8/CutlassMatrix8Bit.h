//
// Created by DmiitriyJarosh on 14.11.2019.
//

#ifndef CFPQ_CUDA_CUTLASSMATRIX8BIT_H
#define CFPQ_CUDA_CUTLASSMATRIX8BIT_H


#include <Matrix.h>

class CutlassMatrix8Bit : public Matrix {
public:
    static int8_t ** MultMatrSquare(unsigned char ** A, int size, unsigned char * grammar_body, unsigned int * grammar_tail, int grammar_size);
};


#endif //CFPQ_CUDA_CUTLASSMATRIX8BIT_H
