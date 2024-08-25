#ifndef CUDA_VECTOR_V2_CUH
#define CUDA_VECTOR_V2_CUH

#include "memoryManagement/mmpool.cuh"
#include "memoryManagement/mmpool_v2.cuh"
#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>
// include log
#include "definition/hub_def.h"
#include "label/hop_constrained_two_hop_labels_v2.cuh"
#include <iostream>

//#define data_type hop_constrained_two_hop_label

template <typename T> class cuda_vector_v2 {
public:
    mmpool_v2<T> *pool = NULL;
    size_t current_size;
    size_t capacity;      // 当前vector的容量, 以块为单位

    int *block_idx_array = NULL; // unified memory
    int now_block; // 记录当前最后一个 block 的id
    int blocks_num;
    int lock;

    // 构造函数
    __host__ cuda_vector_v2(mmpool_v2<T> *pool, const int &idx, size_t capacity = 100); // 初始块大小可以根据需求调整

    // 析构函数
    __host__ ~cuda_vector_v2();

    // 加入一个元素
    __device__ bool push_back(const int &vid, const T &value);
    
    // 获取元素
    __device__ __host__ T *get(size_t index);
    
    // 清空元素
    __device__ void init(int V, const int &vid);

    __host__ void clear();
    
    __host__ __device__ size_t size() const { return current_size; }
    
    __device__ bool empty() const { return current_size == 0; }
    
    __host__ void sort_label(); //从host使用
    
    __host__ __device__ bool resize(size_t new_size);
};

template <typename T> __host__ cuda_vector_v2<T>::cuda_vector_v2(mmpool_v2<T> *pool, const int &idx, size_t capacity) : pool(pool) {
    this->blocks_num = 0;
    this->current_size = 0;
    this->capacity = capacity;
    this->now_block = idx;
    this->lock = 0;

    // copy to cuda
    cudaMallocManaged(&this->block_idx_array, sizeof(int) * capacity);
    this->block_idx_array[this->blocks_num++] = idx;
};

template <typename T> __device__ bool cuda_vector_v2<T>::push_back(const int &vid, const T &value) {

    // 互斥锁
    while (atomicCAS(&this->lock, 0, 1) != 0);
    // printf("vector_push_back: %d %d\n", vid, this->pool->blocks_state[this->block_idx_array[this->blocks_num-1]]);
    
    // 块满了，申请新的块
    if (this->pool->is_full_block(this->now_block)) {
        this->now_block = pool->get_new_block(this->now_block);
        this->block_idx_array[this->blocks_num++] = this->now_block;
    }
    this->pool->push_node(this->now_block, value);
    this->current_size++;

    // 释放锁
    atomicExch(&this->lock, 0);
    return true;

};

template <typename T> __device__ __host__ T *cuda_vector_v2<T>::get(size_t index) {

    //找到对应的块
    int block_idx = this->block_idx_array[index / pool->get_nodes_per_block()];
    
    //找到对应的节点
    int node_idx = index % pool->get_nodes_per_block();
    
    //返回节点
    return pool->get_node(block_idx, node_idx);

};

template <typename T> __device__ void cuda_vector_v2<T>::init(int V, const int &vid) {
    this->current_size = 0;
    this->block_idx_array[0] = vid;
    this->blocks_num = 1;
    this->now_block = vid;
    
    this->pool->set_blocks_state(vid, 0);
    this->pool->last_empty_block_idx = V;
};

template <typename T> __host__ void cuda_vector_v2<T>::clear() {
    //释放所有块
    // for (int i = 0; i < this->blocks; i++) {
    //     pool->remove_block(this->block_idx_array[i]);
    // }
    //释放block_idx_array
    // delete[] this->block_idx_array;
    this->now_block = 0;
    this->current_size = 0;
};

template <typename T> __host__ cuda_vector_v2<T>::~cuda_vector_v2() {
    clear();
    cudaFree(this->block_idx_array);
    // first_elements->clear();，
    // free(this->first_elements);，数据在cuda label中释放
};

template <typename T> __host__ __device__ bool cuda_vector_v2<T>::resize(size_t new_size) {
    //在初始化后立即调用resize()，因此我们不需要检查是否有足够的块
    if (this->now_block == 0) {
        return false;
    }

    if (this->now_block == 1) {
        //刚刚初始化完，标为满
        pool->set_block_user_nodes(this->block_idx_array[0], nodes_per_block);
    }
    if (new_size <= this->now_block) {
        this->now_block = new_size;
        this->current_size = new_size * nodes_per_block;
        return true;
    }
    while (this->now_block < new_size) {
        int block_idx = pool->find_available_block();
        if (block_idx == -1) {
            //没有空行，申请失败
            printf("No available block in mmpool\n");
            assert(false);
            return false;
        }
        this->block_idx_array[this->now_block++] = block_idx;
        // cudaMemcpy(this->block_idx_array + this->blocks, &block_idx, sizeof(int),
        //            cudaMemcpyHostToDevice);
        // this->blocks += 1;
        pool->set_block_user_nodes(block_idx, nodes_per_block);
    }
    this->current_size = new_size * nodes_per_block;
    return true;
}

//显式声明模板类
template <typename hub_type> class cuda_vector_v2;

#endif