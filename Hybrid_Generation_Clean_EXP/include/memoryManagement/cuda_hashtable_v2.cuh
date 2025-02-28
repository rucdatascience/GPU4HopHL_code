#ifndef CUDA_HASHTABLE_V2_CUH
#define CUDA_HASHTABLE_V2_CUH
#pragma once

#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>

// cuda_hashTable_v2
template <typename ValueType> class cuda_hashTable_v2 {
public:
    ValueType *table = NULL;
    int capacity;

    // constructor
    __host__ cuda_hashTable_v2(const int &capacity) : capacity (capacity) {
        cudaMallocManaged(&table, (long long) capacity * sizeof(ValueType));
        cudaMemset(table, 0, (long long) capacity * sizeof(ValueType));
        for (int i = 0; i < capacity; ++i) {
            table[i] = (1<<14);
        }
    }

    // destructor
    __host__ ~cuda_hashTable_v2() {
        cudaFree(table);
    }

    // Modified function
    __device__ void modify (const int pos, const ValueType val) {
        table[pos] = val;
        // __threadfence();
    }
    __device__ void modify (const int vertex, const int hop, const int hop_cst, const ValueType val) {
        table[vertex * (hop_cst + 1) + hop] = val;
        // __threadfence();
    }

    // Modified Min function
    __device__ void modify_min (const int pos, const ValueType val) {
        table[pos] = min(table[pos], val);
        // __threadfence();
    }
    __device__ void modify_min (const int vertex, const int hop, const int hop_cst, const ValueType val) {
        int x = vertex * (hop_cst + 1) + hop;
        table[x] = min(table[x], val);
        // __threadfence();
    }

    // Query function
    __host__ __device__ ValueType get (const int pos) {
        return table[pos];
    }
    __host__ __device__ ValueType get (const int vertex, const int hop, const int hop_cst) {
        return table[vertex * (hop_cst + 1) + hop];
    }

};

template class cuda_hashTable_v2<short>;

#endif