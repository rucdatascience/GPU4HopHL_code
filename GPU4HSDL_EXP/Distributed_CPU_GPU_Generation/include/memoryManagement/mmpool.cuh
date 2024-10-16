#ifndef MMPOOL_CUH
#define MMPOOL_CUH

#include <thread>
#include <unistd.h>
#include "definition/mmpool_size.h"
#include <cuda_runtime.h>
#include <iostream>
#include <mutex>
// #include <atomicAdd.h>
// template <typename T> struct node {
//   T data;
//   bool is_used;
//   bool is_end;
//   __host__ __device__ node() : is_used(false), is_end(false) {}
//   __host__ __device__ node(T data) : data(data), is_used(true), is_end(false)
//   {} bool operator<(const node &n) const { return data < n.data; }
// };

template <typename T> class mmpool {

public:
    T *nodes_pool = NULL;         // 指向预分配的所有节点的指针
    int *block_used_nodes = NULL; // 每个块中已使用节点的数量
    int *block_next_index = NULL; // 下一个块的索引
    int num_blocks;        // 块的数量
    // int nodes_per_block;   // 每个块的节点数量,-1表示未分配
    // int now_empty_block_idx;
    int lock; // gpu mtx
    // std::mutex mtx;

    int last_empty_block_idx; //上一个空行的索引

    // 构造函数
    __host__ mmpool(int num_blocks = 100);
    __host__ __device__ size_t size();
    // 析构函数
    __host__ __device__ ~mmpool();

    __host__ __device__ bool is_full_block(int block_idx);

    __host__ __device__ bool is_valid_block(int block_idx);

    // 添加节点到内存池中指定块尾
    __device__ bool push_node(int block_idx, const T &node_data);

    // 查找内存池中指定行的指定下标的块
    __host__ __device__ T *get_node(int block_idx, int node_idx);

    // 查找空块
    __host__ __device__ int find_available_block(bool mark_used = true);

    // 删除块（逻辑删除）
    __host__ __device__ bool remove_block(int block_idx);

    // 删除node（逻辑删除)
    __host__ __device__ bool remove_node(int block_idx, int pos);

    // 获取块的数量
    __host__ __device__ int get_num_blocks() { return num_blocks; }

    // 获取每个块的节点数量
    __host__ __device__ int get_nodes_per_block() { return nodes_per_block; }

    // 设置块的节点数量
    __host__ __device__ void set_block_user_nodes(int block_idx, int num) {
        block_used_nodes[block_idx] = num;
    }

    //返回block头指针
    __host__ __device__ T *get_block_head(int block_idx) {
        return nodes_pool + (block_idx * nodes_per_block);
    }

    __host__ void prefetch(int block_idx) {
        cudaMemPrefetchAsync(nodes_pool + block_idx * nodes_per_block, sizeof(T) * nodes_per_block, 0, 0);
    }
};

template <typename T>
__host__ mmpool<T>::mmpool(int num_blocks) : num_blocks(num_blocks) {
    lock = 0;
    last_empty_block_idx = 0;

    cudaError_t error = cudaMallocManaged(&nodes_pool, sizeof(T) * nodes_per_block * num_blocks);

    if (error != cudaSuccess) {
        printf("CUDA malloc failed: %s\n", cudaGetErrorString(error));
        // 处理错误，如退出程序
    }

    cudaMallocManaged(&block_used_nodes, sizeof(int) * num_blocks);
    cudaMallocManaged(&block_next_index, sizeof(int) * num_blocks);

    // 初始化每个块
    for (int i = 0; i < num_blocks; ++i) {
        block_used_nodes[i] = -1;
        block_next_index[i] = (i == num_blocks - 1) ? -1 : i + 1;
    }
    // now_empty_block_idx = 0;
}

template <typename T> __host__ __device__ size_t mmpool<T>::size() {
    size_t size = 0;
    for (int i = 0; i < num_blocks; ++i) {
        size += block_used_nodes[i];
    }
    return size;
}

// 析构函数
template <typename T> __host__ __device__ mmpool<T>::~mmpool() {
    cudaFree(nodes_pool);
    cudaFree(block_used_nodes);
    cudaFree(block_next_index);
}

template <typename T>
__host__ __device__ bool mmpool<T>::is_full_block(int block_idx) {
    return block_used_nodes[block_idx] == nodes_per_block;
}

template <typename T>
__host__ __device__ bool mmpool<T>::is_valid_block(int block_idx) {
    if (block_idx >= 0 && block_idx < num_blocks) {
        return true; // 无效块索引
    }
    return false;
}

// 添加节点到内存池
template <typename T> __device__ bool mmpool<T>::push_node(int block_idx, const T &node_data) {
    if (!is_valid_block(block_idx)) {
        return false; // 无效块索引
    }
    if (is_full_block(block_idx)) {
        return false; // 块已满
    }

    // 使用直接的设备端内存访问，而不是 cudaMemcpy

    // int index = atomicAdd(&block_used_nodes[block_idx], 1);
    int index = ++ block_used_nodes[block_idx];
    nodes_pool[block_idx * nodes_per_block + index] = node_data;

    // 直接增加 block_used_nodes[block_idx]
    // block_used_nodes[block_idx]++;

    return true;
}

// 查找空块
template <typename T>
__host__ __device__ int mmpool<T>::find_available_block(bool mark_used) {
    //使用锁来保护
    // 如果在gpu上运行，使用原子操作
    // 如果在cpu上运行，使用互斥锁

    //使用宏检测环境
    #ifndef __CUDA_ARCH__
        //获取锁
        // mtx.lock(); // 获取锁

        // int block_idx = -1;
        // for (int i = last_empty_block_idx; i < num_blocks; i++) {
        //     if (block_used_nodes[i] == -1) {
        //         block_idx = i;
        //         last_empty_block_idx = i + 1;
        //         break;
        //     }
        // }
        // if (mark_used && block_idx != -1 && block_used_nodes[block_idx] == -1) {
        //     block_used_nodes[block_idx] = 0;
        // }

        // mtx.unlock(); // 释放锁
        // return block_idx;

    #else
        while (atomicCAS(&this->lock, 0, 1) != 0) ;
        // 获取锁

        int block_idx = -1;
        for (int i = last_empty_block_idx; i < num_blocks; i++) {
            if (block_used_nodes[i] == -1) {
                block_idx = i;
                last_empty_block_idx = i + 1;
                break;
            }
        }
        if (mark_used && block_idx != -1 && block_used_nodes[block_idx] == -1) {
            block_used_nodes[block_idx] = 0;
        }

        atomicExch(&this->lock, 0); // 释放锁
        return block_idx;
    #endif
}

// 查找内存池中指定行的指定下标的块
template <typename T>
__host__ __device__ T *mmpool<T>::get_node(int block_idx, int node_idx) {
    if (!is_valid_block(block_idx)) {
        return NULL; // 无效块索引
    }
    if (node_idx < 0 || node_idx >= block_used_nodes[block_idx]) {
        return NULL; // 无效节点索引
    }
    return &nodes_pool[block_idx * nodes_per_block + node_idx];
}

// 删除块（逻辑删除）
template <typename T>
__host__ __device__ bool mmpool<T>::remove_block(int block_idx) {
    if (!is_valid_block(block_idx)) {
        return false; // 无效块索引
    }

    block_used_nodes[block_idx] = 0; // 逻辑删除，标记为未使用
    return true;
}

#endif