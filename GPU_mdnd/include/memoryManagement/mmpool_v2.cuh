#ifndef MMPOOL_V2_CUH
#define MMPOOL_V2_CUH

#include "definition/mmpool_size.h"
#include <cuda_runtime.h>
#include <iostream>
#include <mutex>

template <typename T> class mmpool_v2 {

public:
    struct block {
        T data[nodes_per_block];
    }; // 一个块的元素
    block *blocks_pool; // 指向预分配的所有节点的指针
    int *blocks_state; // 当前块的状态，0为未使用，-x表示使用了x个，+y表示下一个块的位置是y
    int num_blocks; // 块的数量
    int last_empty_block_idx; // 最后一个空块的索引

    int lock; // gpu mtx

    // 构造函数
    __host__ mmpool_v2(const int &V, const int &num_blocks = 100);

    // 获取内存池元素个数
    __host__ __device__ size_t size();

    // 析构函数
    __host__ __device__ ~mmpool_v2();

    __host__ __device__ bool is_full_block(const int &block_idx);

    __host__ __device__ bool is_valid_block(const int &block_idx);

    // 添加节点到内存池中指定块尾
    __device__ bool push_node(const int &block_idx, const T &node_data);

    // 查找内存池中指定行的指定下标的块
    __host__ __device__ T *get_node(const int &block_idx, const int &node_idx);

    // 获取一个新的块
    __host__ __device__ int get_new_block(const int &block_idx);

    // 获取块大小
    __host__ __device__ int get_block_size(const int &block_idx);
    
    // 修改指定 blocks_state 的值
    __host__ __device__ void set_blocks_state(const int &block_idx, const int &value);

    // 查找空块
    // __host__ __device__ int find_available_block(bool mark_used = true);

    // 删除块（逻辑删除）
    // __host__ __device__ bool remove_block(int block_idx);

    // // 删除node（逻辑删除)
    // __host__ __device__ bool remove_node(int block_idx, int pos);

    // 获取块的数量
    __host__ __device__ int get_num_blocks() { return num_blocks; }

    // 获取每个块的节点数量
    __host__ __device__ int get_nodes_per_block() { return nodes_per_block; }

    // 设置块的节点数量
    __host__ __device__ void set_block_user_nodes(const int &block_idx, const int &num) {
        blocks_state[block_idx] = -num;
    }

    //返回block头指针
    __host__ __device__ block *get_block_head(const int &block_idx) {
        return (blocks_pool + block_idx);
    }

};

// 构造函数
template <typename T> __host__ mmpool_v2<T>::mmpool_v2(const int &V, const int &num_blocks) : num_blocks(num_blocks) {
    
    lock = 0;
    cudaMallocManaged(&blocks_pool, sizeof(block) * num_blocks);
    cudaMallocManaged(&blocks_state, sizeof(int) * num_blocks);

    // 初始化每个块
    for (int i = 0; i < num_blocks; ++i) {
        blocks_state[i] = 0;
    }
    last_empty_block_idx = V;

}

// 获取内存池元素个数
template <typename T> __host__ __device__ size_t mmpool_v2<T>::size() {
    size_t size = 0;
    for (int i = 0; i < num_blocks; ++i) {
        if (blocks_state[i] < 0){
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
    if (block_idx >= 0 && block_idx < num_blocks) return true; // 无效块索引
    return false;
}

// 添加节点到内存池
template <typename T> __device__ bool mmpool_v2<T>::push_node(const int &block_idx, const T &node_data) {
    
    // 无效块索引
    if (!is_valid_block(block_idx)) {
        return false; 
    }
    // 修改块元素内容
    blocks_pool[block_idx].data[-blocks_state[block_idx]] = node_data;
    blocks_state[block_idx]--;

    return true;
}

// 获取一个新块，前一个块是 block_idx
template <typename T> __host__ __device__ int mmpool_v2<T>::get_new_block(const int &block_idx) {
    while (atomicCAS(&this->lock, 0, 1) != 0);
    // printf("a new block!! %d %d %d\n", block_idx, last_empty_block_idx, num_blocks);
    blocks_state[last_empty_block_idx] = 0;
    blocks_state[block_idx] = last_empty_block_idx ++;
    atomicExch(&this->lock, 0); // 释放锁
    return blocks_state[block_idx];
}

template <typename T> __host__ __device__ int mmpool_v2<T>::get_block_size(const int &block_idx) {

    // 无效块索引
    if (!is_valid_block(block_idx)) {
        return -1; 
    }

    if (blocks_state[block_idx] > 0) return nodes_per_block;
    else return -blocks_state[block_idx];
}

template <typename T> __host__ __device__ void mmpool_v2<T>::set_blocks_state(const int &block_idx, const int &value) {
    blocks_state[block_idx] = value;
}

// 查找内存池中指定行的指定下标的块
template <typename T> __host__ __device__ T *mmpool_v2<T>::get_node(const int &block_idx, const int &node_idx) {
    // 无效块索引
    if (!is_valid_block(block_idx)) {
        return NULL; 
    }
    // 无效节点索引
    // if (node_idx < 0 || (node_idx > -blocks_state[block_idx] || blocks_state[block_idx] > 0)) {
    //     return NULL;
    // }
    return &(blocks_pool[block_idx].data[node_idx]);
}

#endif