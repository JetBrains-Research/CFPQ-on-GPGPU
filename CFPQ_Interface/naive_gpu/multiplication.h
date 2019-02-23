#pragma once

#include <stdio.h>
#include <stdint.h>
#include <iostream>


int rows(int N);

int cols(int N);

void synchronize();

uint32_t * device_matrix_alloc(int N);

uint32_t * host_matrix_calloc(int N);

void gpu2cpu(int N, uint32_t *d_M, uint32_t *h_M);

void cpu2gpu(int N, uint32_t *h_M, uint32_t *d_M);

bool MatrixMulAdd(uint32_t *A, uint32_t *B, uint32_t *C, int N, uint32_t *tmp);