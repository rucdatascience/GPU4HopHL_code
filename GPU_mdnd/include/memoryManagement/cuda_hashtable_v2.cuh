#ifndef CUDA_HASHTABLE_V2_CUH
#define CUDA_HASHTABLE_V2_CUH

#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>

// cuda_hashTable_v2
template <typename ValueType> class cuda_hashTable_v2 {
public:
    ValueType *table = NULL;
    int capacity;

    // 构造函数
    __host__ cuda_hashTable_v2(const int &capacity) : capacity (capacity) {
        cudaMallocManaged(&table, capacity * sizeof(ValueType));
        for (int i = 0; i < capacity; ++i) {
            table[i] = 1e9;
        }
    }

    // 析构函数
    __host__ ~cuda_hashTable_v2() {
        cudaFree(table);
    }

    // 修改函数
    __host__ __device__ void modify (const int pos, const int val) {
        table[pos] = val;
    }
    __host__ __device__ void modify (const int vertex, const int hop, const int hop_cst, const int val) {
        table[vertex * (hop_cst + 1) + hop] = val;
    }

    // 查询函数
    __host__ __device__ ValueType get (const int pos) {
        return table[pos];
    }
    __host__ __device__ ValueType get (const int vertex, const int hop, const int hop_cst) {
        return table[vertex * (hop_cst + 1) + hop];
    }

};

template class cuda_hashTable_v2<int>;

#endif