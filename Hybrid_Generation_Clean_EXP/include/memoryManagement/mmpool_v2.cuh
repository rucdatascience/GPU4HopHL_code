#ifndef MMPOOL_V2_CUH
#define MMPOOL_V2_CUH
#pragma once

#include "definition/mmpool_size.h"
#include <cuda_runtime.h>
#include <iostream>
#include <mutex>

template <typename T> class mmpool_v2 {

public:
    struct block {
        T data[nodes_per_block];
    }; // 一个块的元素
    block *blocks_pool = NULL; // 指向预分配的所有节点的指针
    int *blocks_state = NULL; // 当前块的状态，0为未使用，-x表示使用了x个，+y表示下一个块的位置是y
    int num_blocks = 0; // 块的数量
    int last_empty_block_idx = 0; // 最后一个空块的索引

    int lock = 0; // gpu mtx

    // 构造函数
    __host__ mmpool_v2(int V, int nb);

    // 获取内存池元素个数
    __host__ __device__ int size();

    // 析构函数
    __host__ __device__ ~mmpool_v2();

    __host__ __device__ bool is_full_block (const int &block_idx);

    __host__ __device__ bool is_valid_block (const int &block_idx);

    // 添加节点到内存池中指定块尾
    __device__ bool push_node (const int &block_idx, const T &node_data);

    // 查找内存池中指定行的指定下标的块
    __device__ T *get_node (const int &block_idx, const int &node_idx);
    __host__ T *get_node_host (const int &block_idx, const int &node_idx);

    // 获取一个新的块
    __device__ int get_new_block (const int &block_idx);

    // 获取块大小
    __device__ int get_block_size (const int &block_idx);
    
    __host__ int get_block_size_host (const int &block_idx);

    // 修改指定 blocks_state 的值
    __device__ void set_blocks_state (const int &block_idx, const int &value);

    // 获取块的数量
    __host__ __device__ int get_num_blocks() { return num_blocks; }

    // 获取每个块的节点数量
    __host__ __device__ int get_nodes_per_block() { return nodes_per_block; }

    // 设置块的节点数量
    __host__ __device__ void set_block_user_nodes (const int &block_idx, const int &num) {
        blocks_state[block_idx] = -num;
    }

    //返回block头指针
    __host__ __device__ block *get_block_head(const int &block_idx) {
        return (blocks_pool + block_idx);
    }

};

// 构造函数
template <typename T> __host__ mmpool_v2<T>::mmpool_v2(int V, int nb) : num_blocks(nb) {
    
    lock = 0;
    cudaMallocManaged(&blocks_pool, (long long) sizeof(block) * nb);
    cudaDeviceSynchronize();
    cudaMallocManaged(&blocks_state, (long long) sizeof(int) * nb);
    cudaDeviceSynchronize();
    // printf("num_blocks: %d\n", num_blocks);
    // 初始化每个块
    for (int i = 0; i < nb; ++i) {
        // printf("mmpool v2 T: %d\n", i);
        blocks_state[i] = 0;
    }
    last_empty_block_idx = V;

}

// 获取内存池元素个数
template <typename T> __host__ __device__ int mmpool_v2<T>::size() {
    int size = 0;
    for (int i = 0; i < num_blocks; ++i) {
        if (blocks_state[i] < 0) {
            size -= blocks_state[i];
        }
    }
    return size;
}

// 析构函数
template <typename T> __host__ __device__ mmpool_v2<T>::~mmpool_v2() {
    cudaFree(blocks_pool);
    cudaFree(blocks_state);
}

// 判断满块
template <typename T> __host__ __device__ bool mmpool_v2<T>::is_full_block(const int &block_idx) {
    return (blocks_state[block_idx] == -nodes_per_block) || (blocks_state[block_idx] > 0);
}

// 判断无效块索引
template <typename T> __host__ __device__ bool mmpool_v2<T>::is_valid_block(const int &block_idx) {
    if (block_idx >= 0 && block_idx < num_blocks){
        return true; // 无效块索引
    }
    return false;
}

// 添加节点到内存池
template <typename T> __device__ bool mmpool_v2<T>::push_node(const int &block_idx, const T &node_data) {
    // 修改块元素内容
    
    blocks_pool[block_idx].data[-blocks_state[block_idx]] = node_data;
    // __threadfence_system();
    // atomicSub(&(blocks_state[block_idx]), 1);
    blocks_state[block_idx]--;
    // __threadfence_system();
    
}

// 获取一个新块，前一个块是 block_idx
template <typename T> __device__ int mmpool_v2<T>::get_new_block(const int &block_idx) {
    bool blocked = true;
    while (blocked) {
        if (atomicCAS(&this->lock, 0, 1) == 0) {
            if (last_empty_block_idx >= num_blocks) {
                printf("error with blocks num!\n");
            }
            blocks_state[last_empty_block_idx] = 0;
            blocks_state[block_idx] = last_empty_block_idx;
            atomicAdd(&last_empty_block_idx, 1);
            // __threadfence_system();
            atomicExch(&this->lock, 0); // 释放锁
            blocked = false;
        }
    }
    int x = blocks_state[block_idx];
    // __threadfence_system();
    return x;
}

template <typename T> __device__ int mmpool_v2<T>::get_block_size(const int &block_idx) {
    int x = 0;
    if (blocks_state[block_idx] > 0) {
        x = nodes_per_block;
        // __threadfence_system();
    }else{
        x = -blocks_state[block_idx];
        // __threadfence_system();
    }
    return x;
}

template <typename T> __host__ int mmpool_v2<T>::get_block_size_host(const int &block_idx) {
    int x = 0;
    if (blocks_state[block_idx] > 0) {
        x = nodes_per_block;
    }else{
        x = -blocks_state[block_idx];
    }
    return x;
}

template <typename T> __device__ void mmpool_v2<T>::set_blocks_state(const int &block_idx, const int &value) {
    blocks_state[block_idx] = value;
    __threadfence_system();
}

// 查找内存池中指定行的指定下标的块
template <typename T> __device__ T *mmpool_v2<T>::get_node(const int &block_idx, const int &node_idx) {
    T *ret = &(blocks_pool[block_idx].data[node_idx]);
    // __threadfence_system();
    // printf(" ");
    return ret;
}

template <typename T> __host__ T *mmpool_v2<T>::get_node_host(const int &block_idx, const int &node_idx) {
    T *ret = &(blocks_pool[block_idx].data[node_idx]);
    return ret;
}

#endif