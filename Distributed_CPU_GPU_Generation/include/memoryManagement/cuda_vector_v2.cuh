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
    int last_size, current_size;
    int capacity;      // ��ǰvector������, �Կ�Ϊ��λ

    int *block_idx_array = NULL; // unified memory
    int now_block; // ��¼��ǰ���һ�� block ��id
    int blocks_num;
    int lock;

    // ���캯��
    __host__ cuda_vector_v2(mmpool_v2<T> *pool, const int &idx, int capacity); // ��ʼ���С���Ը����������

    // ��������
    __host__ ~cuda_vector_v2();

    // ����һ��Ԫ��
    __device__ void push_back(const T value);
    
    // ��ȡԪ��
    __device__ T *get(int index);
    
    // ���Ԫ��
    __device__ void init(int V, const int vid);

    __host__ void clear();
    
    __host__ __device__ size_t size() const { return current_size; }
    
    __device__ bool empty() const { return current_size == 0; }
    
};

template <typename T> __host__ cuda_vector_v2<T>::cuda_vector_v2(mmpool_v2<T> *pool, const int &idx, int capacity) : pool(pool) {
    this->blocks_num = 0;
    this->current_size = 0;
    this->capacity = capacity;
    this->now_block = idx;
    this->lock = 0;

    // copy to cuda
    cudaMallocManaged(&this->block_idx_array, sizeof(int) * capacity);
    this->block_idx_array[this->blocks_num++] = idx;
};

template <typename T> __device__ void cuda_vector_v2<T>::push_back(const T value) {
    // ������
    bool blocked = true;
    while (blocked) {
        if (0 == atomicCAS(&(this->lock), 0, 1)) {
            // �����ˣ������µĿ�
            if (this->pool->is_full_block(this->now_block)) {
                this->now_block = pool->get_new_block(this->now_block);
                this->pool->push_node(this->now_block, value);
                this->block_idx_array[this->blocks_num] = this->now_block;
                this->blocks_num ++;
                ++ this->current_size;
            }else{
                this->pool->push_node(this->now_block, value);
                ++ this->current_size;
            }
            // �ͷ���
            atomicExch(&(this->lock), 0);
            blocked = false;
        }
    }
};

template <typename T> __device__ T *cuda_vector_v2<T>::get(int index) {

    //�ҵ���Ӧ�Ŀ�
    int block_idx = this->block_idx_array[index / pool->get_nodes_per_block()];
    //�ҵ���Ӧ�Ľڵ�
    int node_idx = index % pool->get_nodes_per_block();
    //���ؽڵ�
    return pool->get_node(block_idx, node_idx);

};

template <typename T> __device__ void cuda_vector_v2<T>::init(int V, const int vid) {
    this->current_size = 0;
    this->block_idx_array[0] = vid;
    this->blocks_num = 1;
    this->now_block = vid;
    this->pool->set_blocks_state(vid, 0);
    this->pool->last_empty_block_idx = V;
};

template <typename T> __host__ void cuda_vector_v2<T>::clear() {
    this->now_block = 0;
    this->current_size = 0;
};

template <typename T> __host__ cuda_vector_v2<T>::~cuda_vector_v2() {
    clear();
    cudaFree(this->block_idx_array);
};

//��ʽ����ģ����
template <typename hub_type> class cuda_vector_v2;

#endif