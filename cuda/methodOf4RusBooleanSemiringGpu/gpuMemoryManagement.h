
#ifndef GPU_MEMORY_MANAGEMENT
#define GPU_MEMORY_MANAGEMENT

#include "Constants.h"

namespace gpu_m4ri {

    TYPE ** allocate_tables(int num_tables, int num_rows, int num_cols);

    void delete_tables(TYPE **tables, int num_tables);

    TYPE * allocate_matrix_host(int rows, int cols);

    void delete_matrix_host(TYPE * matrix);

    TYPE * allocate_matrix_device(int rows, int cols);

    void delete_matrix_device(TYPE *matrix);

    void copy_device_to_host_sync(TYPE *device, TYPE *host, int elems);

    void copy_host_to_device_sync(TYPE *host, TYPE *device, int elems);

    void copy_device_to_device_sync(TYPE *src, TYPE *dst, int elems);

    void copy_device_to_host_async(TYPE *device, TYPE *host, int elems);

    void copy_host_to_device_async(TYPE *host, TYPE *device, int elems);

    void synchronize_with_gpu();

}

#endif //GPU_MEMORY_MANAGEMENT
