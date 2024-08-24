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
    }; // һ�����Ԫ��
    block *blocks_pool; // ָ��Ԥ��������нڵ��ָ��
    int *blocks_state; // ��ǰ���״̬��0Ϊδʹ�ã�-x��ʾʹ����x����+y��ʾ��һ�����λ����y
    int num_blocks; // �������
    int last_empty_block_idx; // ���һ���տ������

    int lock; // gpu mtx

    // ���캯��
    __host__ mmpool_v2(const int &V, const int &num_blocks = 100);

    // ��ȡ�ڴ��Ԫ�ظ���
    __host__ __device__ size_t size();

    // ��������
    __host__ __device__ ~mmpool_v2();

    __host__ __device__ bool is_full_block(const int &block_idx);

    __host__ __device__ bool is_valid_block(const int &block_idx);

    // ��ӽڵ㵽�ڴ����ָ����β
    __device__ bool push_node(const int &block_idx, const T &node_data);

    // �����ڴ����ָ���е�ָ���±�Ŀ�
    __host__ __device__ T *get_node(const int &block_idx, const int &node_idx);

    // ��ȡһ���µĿ�
    __host__ __device__ int get_new_block(const int &block_idx);

    // ��ȡ���С
    __host__ __device__ int get_block_size(const int &block_idx);
    
    // �޸�ָ�� blocks_state ��ֵ
    __host__ __device__ void set_blocks_state(const int &block_idx, const int &value);

    // ���ҿտ�
    // __host__ __device__ int find_available_block(bool mark_used = true);

    // ɾ���飨�߼�ɾ����
    // __host__ __device__ bool remove_block(int block_idx);

    // // ɾ��node���߼�ɾ��)
    // __host__ __device__ bool remove_node(int block_idx, int pos);

    // ��ȡ�������
    __host__ __device__ int get_num_blocks() { return num_blocks; }

    // ��ȡÿ����Ľڵ�����
    __host__ __device__ int get_nodes_per_block() { return nodes_per_block; }

    // ���ÿ�Ľڵ�����
    __host__ __device__ void set_block_user_nodes(const int &block_idx, const int &num) {
        blocks_state[block_idx] = -num;
    }

    //����blockͷָ��
    __host__ __device__ block *get_block_head(const int &block_idx) {
        return (blocks_pool + block_idx);
    }

};

// ���캯��
template <typename T> __host__ mmpool_v2<T>::mmpool_v2(const int &V, const int &num_blocks) : num_blocks(num_blocks) {
    
    lock = 0;
    cudaMallocManaged(&blocks_pool, sizeof(block) * num_blocks);
    cudaMallocManaged(&blocks_state, sizeof(int) * num_blocks);

    // ��ʼ��ÿ����
    for (int i = 0; i < num_blocks; ++i) {
        blocks_state[i] = 0;
    }
    last_empty_block_idx = V;

}

// ��ȡ�ڴ��Ԫ�ظ���
template <typename T> __host__ __device__ size_t mmpool_v2<T>::size() {
    size_t size = 0;
    for (int i = 0; i < num_blocks; ++i) {
        if (blocks_state[i] < 0){
            size -= blocks_state[i];
        }
    }
    return size;
}

// ��������
template <typename T> __host__ __device__ mmpool_v2<T>::~mmpool_v2() {
    cudaFree(blocks_pool);
    cudaFree(blocks_state);
}

// �ж�����
template <typename T> __host__ __device__ bool mmpool_v2<T>::is_full_block(const int &block_idx) {
    return (blocks_state[block_idx] == -nodes_per_block) || (blocks_state[block_idx] > 0);
}

// �ж���Ч������
template <typename T> __host__ __device__ bool mmpool_v2<T>::is_valid_block(const int &block_idx) {
    if (block_idx >= 0 && block_idx < num_blocks) return true; // ��Ч������
    return false;
}

// ��ӽڵ㵽�ڴ��
template <typename T> __device__ bool mmpool_v2<T>::push_node(const int &block_idx, const T &node_data) {
    
    // ��Ч������
    if (!is_valid_block(block_idx)) {
        return false; 
    }
    // �޸Ŀ�Ԫ������
    blocks_pool[block_idx].data[-blocks_state[block_idx]] = node_data;
    blocks_state[block_idx]--;

    return true;
}

// ��ȡһ���¿飬ǰһ������ block_idx
template <typename T> __host__ __device__ int mmpool_v2<T>::get_new_block(const int &block_idx) {
    while (atomicCAS(&this->lock, 0, 1) != 0);
    // printf("a new block!! %d %d %d\n", block_idx, last_empty_block_idx, num_blocks);
    blocks_state[last_empty_block_idx] = 0;
    blocks_state[block_idx] = last_empty_block_idx ++;
    atomicExch(&this->lock, 0); // �ͷ���
    return blocks_state[block_idx];
}

template <typename T> __host__ __device__ int mmpool_v2<T>::get_block_size(const int &block_idx) {

    // ��Ч������
    if (!is_valid_block(block_idx)) {
        return -1; 
    }

    if (blocks_state[block_idx] > 0) return nodes_per_block;
    else return -blocks_state[block_idx];
}

template <typename T> __host__ __device__ void mmpool_v2<T>::set_blocks_state(const int &block_idx, const int &value) {
    blocks_state[block_idx] = value;
}

// �����ڴ����ָ���е�ָ���±�Ŀ�
template <typename T> __host__ __device__ T *mmpool_v2<T>::get_node(const int &block_idx, const int &node_idx) {
    // ��Ч������
    if (!is_valid_block(block_idx)) {
        return NULL; 
    }
    // ��Ч�ڵ�����
    // if (node_idx < 0 || (node_idx > -blocks_state[block_idx] || blocks_state[block_idx] > 0)) {
    //     return NULL;
    // }
    return &(blocks_pool[block_idx].data[node_idx]);
}

#endif