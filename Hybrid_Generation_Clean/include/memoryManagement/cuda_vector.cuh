#pragma once

#include <cstdio>
#include <cassert>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

struct label {
    int v, h, d;
};

template <typename T> class base_memory {
    public:
        T* data; // size_per_block * capacity

        int lock = 0;

        size_t size_per_block = 1024;
        size_t capacity; // how many blocks

        int* block_size; // how many elements in each block, if block_size[i] == -1, then the block is not allocated

        __host__ base_memory(size_t size_per_block, size_t capacity);
        __device__ int find_set_available_block();
        __host__ __device__ ~base_memory();
};

template <typename T> class cuda_vector {
    public:
        base_memory<T>* pool;
        int* block_idx_array;
        int blocks;

        int blocks_upper_bound = 128; // how many blocks can be allocated

        size_t current_size;

        __host__ cuda_vector(base_memory<T>* pool, int max_blocks);
        __host__ __device__ T operator[](size_t index);
        __device__ void push_back(const T& value);

        __host__ __device__ ~cuda_vector();
};

template class base_memory<int>;
template class cuda_vector<int>;

template class base_memory<label>;
template class cuda_vector<label>;
