#pragma once
#include <stdio.h> 
#include <stdint.h>
#include "gpu_memory_management.h"

#define BLOCK_SIZE_COL 32
#define BLOCK_SIZE_ROW 32
#define K 8
#define CHANGED 1
#define NOT_CHANGED 0

/*
*
*
*
*/
struct Tables {
    uint32_t **table_n;
    uint32_t **table_last;
    int cols_n;
    int cols_last;
    int num_tables;
    int num_launches;
    void initialize(int rows, int cols, int max_cols) {
        cols_n = max_cols;
        cols_last = cols % max_cols ;
        num_tables = rows / K ; // rows always :32
        num_launches = cols / max_cols;
        if(cols_last != 0) {
            table_last = allocate_tables(num_tables,rows,cols_last);
        }
        if(num_launches != 0) {            
            table_n = allocate_tables(num_tables,rows,cols_n);
        }		
    }
    void free() {
        if(cols_last != 0) {
            delete_tables(table_last, num_tables);
        }
        if(num_launches != 0) {            
            delete_tables(table_n, num_tables);
        }        
    }

};

/*
*
*
*/
int wrapper_methodOf4Rus_subring(uint32_t *a, uint32_t *b, uint32_t *c, Tables tables, int rows, int cols, uint32_t *is_c_changed);



