
#ifndef GPU_M4RUS_MUL
#define GPU_M4RUS_MUL

#include <stdio.h> 
#include <stdint.h>
#include "gpuMemoryManagement.h"
#include "Constants.h"

#define BLOCK_SIZE_COL 32
#define BLOCK_SIZE_ROW 32

namespace gpu_m4ri {
    
    struct Tables {
        TYPE **table_n;
        TYPE **table_last;
        int cols_n;
        int cols_last;
        int num_tables;
        int num_launches;
    
        void initialize(int rows, int cols, int max_cols) {
            if (max_cols > cols) {
                max_cols = cols;
            } else if (max_cols < 32) {
                max_cols = 32; 
            }
            cols_n = max_cols;
            cols_last = cols % max_cols;
            num_tables = rows / K; // rows always :32
            num_launches = cols / max_cols;

            if(cols_last != 0) { //due to specific alignment in gpu
                table_last = allocate_tables(num_tables, TABLE_ROW, cols_n);
            }
            if(num_launches != 0) {            
                table_n = allocate_tables(num_tables, TABLE_ROW, cols_n);
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

    int wrapper_method_of_4rus_bool_semiring(TYPE *a, TYPE *b, TYPE *c, 
                                              Tables &tables, int rows, int cols);
}

#endif // GPU_M4RUS_MUL
