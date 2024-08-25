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
    size_t capacity;      // ��ǰvector������, �Կ�Ϊ��λ

    int *block_idx_array = NULL; // unified memory
    int now_block; // ��¼��ǰ���һ�� block ��id
    int blocks_num;
    int lock;

    // ���캯��
    __host__ cuda_vector_v2(mmpool_v2<T> *pool, const int &idx, size_t capacity = 100); // ��ʼ���С���Ը����������

    // ��������
    __host__ ~cuda_vector_v2();

    // ����һ��Ԫ��
    __device__ bool push_back(const int &vid, const T &value);
    
    // ��ȡԪ��
    __device__ __host__ T *get(size_t index);
    
    // ���Ԫ��
    __device__ void init(int V, const int &vid);

    __host__ void clear();
    
    __host__ __device__ size_t size() const { return current_size; }
    
    __device__ bool empty() const { return current_size == 0; }
    
    __host__ void sort_label(); //��hostʹ��
    
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

    // ������
    while (atomicCAS(&this->lock, 0, 1) != 0);
    // printf("vector_push_back: %d %d\n", vid, this->pool->blocks_state[this->block_idx_array[this->blocks_num-1]]);
    
    // �����ˣ������µĿ�
    if (this->pool->is_full_block(this->now_block)) {
        this->now_block = pool->get_new_block(this->now_block);
        this->block_idx_array[this->blocks_num++] = this->now_block;
    }
    this->pool->push_node(this->now_block, value);
    this->current_size++;

    // �ͷ���
    atomicExch(&this->lock, 0);
    return true;

};

template <typename T> __device__ __host__ T *cuda_vector_v2<T>::get(size_t index) {

    //�ҵ���Ӧ�Ŀ�
    int block_idx = this->block_idx_array[index / pool->get_nodes_per_block()];
    
    //�ҵ���Ӧ�Ľڵ�
    int node_idx = index % pool->get_nodes_per_block();
    
    //���ؽڵ�
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
    //�ͷ����п�
    // for (int i = 0; i < this->blocks; i++) {
    //     pool->remove_block(this->block_idx_array[i]);
    // }
    //�ͷ�block_idx_array
    // delete[] this->block_idx_array;
    this->now_block = 0;
    this->current_size = 0;
};

template <typename T> __host__ cuda_vector_v2<T>::~cuda_vector_v2() {
    clear();
    cudaFree(this->block_idx_array);
    // first_elements->clear();��
    // free(this->first_elements);��������cuda label���ͷ�
};

template <typename T> __host__ __device__ bool cuda_vector_v2<T>::resize(size_t new_size) {
    //�ڳ�ʼ������������resize()��������ǲ���Ҫ����Ƿ����㹻�Ŀ�
    if (this->now_block == 0) {
        return false;
    }

    if (this->now_block == 1) {
        //�ոճ�ʼ���꣬��Ϊ��
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
            //û�п��У�����ʧ��
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

//��ʽ����ģ����
template <typename hub_type> class cuda_vector_v2;

#endif