
#pragma once

#include <stdint.h>

uint32_t ** allocate_tables(int num_tables, int num_rows, int num_cols);

void delete_tables(uint32_t **tables, int num_tables);

uint32_t * allocate_matrix_host(int rows, int cols);

void delete_matrix_host(uint32_t * matrix);

uint32_t * allocate_matrix_device(int rows, int cols);

void delete_matrix_device(uint32_t *matrix);

void copy_device_to_host_sync(uint32_t *device, uint32_t *host, int elems);

void copy_host_to_device_sync(uint32_t *host, uint32_t *device, int elems);

void copy_device_to_device(uint32_t *src, uint32_t *dst, int elems);

void copy_device_to_host_async(uint32_t *device, uint32_t *host, int elems);

void copy_host_to_device_async(uint32_t *host, uint32_t *device, int elems);

void synchronize_with_gpu();
