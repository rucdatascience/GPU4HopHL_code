#ifndef CUDA_HASHTABLE_CUH
#define CUDA_HASHTABLE_CUH

#include <cassert>
#include <cstdio>
#include <cuda_runtime.h>
//#include <thrust/pair.h>


typedef struct label_index {
    __host__ __device__ label_index(int t){
        start = t;
        end = t;
    }
    __host__ __device__ label_index(int a, int b){
        start = a;
        end = b;
    }
    int start;
    int end;
} label_index;

template <typename T1, typename T2> struct mypair {
    T1 first;
    T2 second;

    // 构造函数
    __host__ __device__ mypair(T1 f, T2 s) : first(f), second(s) {}
};


template <typename KeyType, typename ValueType> class cuda_hashTable {
    public:
    mypair<KeyType, ValueType> *table;
    int capacity;

    __host__ cuda_hashTable(int capacity) : capacity(capacity) {
        cudaMallocManaged(&table, capacity * sizeof(mypair<KeyType, ValueType>));

        for (int i = 0; i < capacity; ++i) {
            table[i] = {KeyType(-1), ValueType(-1)};
            if(table[i].first != -1){
                assert(false);
            }
        }
        //printf("init: table[0].first:%d\n", table[0].first);

    }

    __host__ ~cuda_hashTable() { cudaFree(table); }

    __host__ __device__ int hash(KeyType key) const { return key % capacity; }

    __device__ __host__ void insert(KeyType key, ValueType value) {
        int index = hash(key);
        //printf("table[index].first:%d\n", table[0].first);

        while (table[index].first != KeyType(-1) && table[index].first != key) {
            index = (index + 1) % capacity; // 线性探测
            if (index == hash(key)) {
            //printf("Hash table is full\n");
            //assert(false);
            }
        }
        table[index] = {key, value};
    }

    __device__ __host__ ValueType* find(KeyType key) const {
        int index = hash(key);
        while (table[index].first != KeyType(-1)) {
            if (table[index].first == key) {
                return &table[index].second;
            }
            index = (index + 1) % capacity; // 线性探测
            if (index == hash(key)) {
                //printf("Not Found\n");
                break;
            }
        }
        return NULL; // 表示未找到
    }
};

template class cuda_hashTable<int, int>;
template class cuda_hashTable<int, label_index>;
#endif