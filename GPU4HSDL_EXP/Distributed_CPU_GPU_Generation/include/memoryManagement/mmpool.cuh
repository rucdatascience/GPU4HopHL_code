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
    T *nodes_pool = NULL;         // ָ��Ԥ��������нڵ��ָ��
    int *block_used_nodes = NULL; // ÿ��������ʹ�ýڵ������
    int *block_next_index = NULL; // ��һ���������
    int num_blocks;        // �������
    // int nodes_per_block;   // ÿ����Ľڵ�����,-1��ʾδ����
    // int now_empty_block_idx;
    int lock; // gpu mtx
    // std::mutex mtx;

    int last_empty_block_idx; //��һ�����е�����

    // ���캯��
    __host__ mmpool(int num_blocks = 100);
    __host__ __device__ size_t size();
    // ��������
    __host__ __device__ ~mmpool();

    __host__ __device__ bool is_full_block(int block_idx);

    __host__ __device__ bool is_valid_block(int block_idx);

    // ��ӽڵ㵽�ڴ����ָ����β
    __device__ bool push_node(int block_idx, const T &node_data);

    // �����ڴ����ָ���е�ָ���±�Ŀ�
    __host__ __device__ T *get_node(int block_idx, int node_idx);

    // ���ҿտ�
    __host__ __device__ int find_available_block(bool mark_used = true);

    // ɾ���飨�߼�ɾ����
    __host__ __device__ bool remove_block(int block_idx);

    // ɾ��node���߼�ɾ��)
    __host__ __device__ bool remove_node(int block_idx, int pos);

    // ��ȡ�������
    __host__ __device__ int get_num_blocks() { return num_blocks; }

    // ��ȡÿ����Ľڵ�����
    __host__ __device__ int get_nodes_per_block() { return nodes_per_block; }

    // ���ÿ�Ľڵ�����
    __host__ __device__ void set_block_user_nodes(int block_idx, int num) {
        block_used_nodes[block_idx] = num;
    }

    //����blockͷָ��
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
        // ����������˳�����
    }

    cudaMallocManaged(&block_used_nodes, sizeof(int) * num_blocks);
    cudaMallocManaged(&block_next_index, sizeof(int) * num_blocks);

    // ��ʼ��ÿ����
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

// ��������
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
        return true; // ��Ч������
    }
    return false;
}

// ��ӽڵ㵽�ڴ��
template <typename T> __device__ bool mmpool<T>::push_node(int block_idx, const T &node_data) {
    if (!is_valid_block(block_idx)) {
        return false; // ��Ч������
    }
    if (is_full_block(block_idx)) {
        return false; // ������
    }

    // ʹ��ֱ�ӵ��豸���ڴ���ʣ������� cudaMemcpy

    // int index = atomicAdd(&block_used_nodes[block_idx], 1);
    int index = ++ block_used_nodes[block_idx];
    nodes_pool[block_idx * nodes_per_block + index] = node_data;

    // ֱ������ block_used_nodes[block_idx]
    // block_used_nodes[block_idx]++;

    return true;
}

// ���ҿտ�
template <typename T>
__host__ __device__ int mmpool<T>::find_available_block(bool mark_used) {
    //ʹ����������
    // �����gpu�����У�ʹ��ԭ�Ӳ���
    // �����cpu�����У�ʹ�û�����

    //ʹ�ú��⻷��
    #ifndef __CUDA_ARCH__
        //��ȡ��
        // mtx.lock(); // ��ȡ��

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

        // mtx.unlock(); // �ͷ���
        // return block_idx;

    #else
        while (atomicCAS(&this->lock, 0, 1) != 0) ;
        // ��ȡ��

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

        atomicExch(&this->lock, 0); // �ͷ���
        return block_idx;
    #endif
}

// �����ڴ����ָ���е�ָ���±�Ŀ�
template <typename T>
__host__ __device__ T *mmpool<T>::get_node(int block_idx, int node_idx) {
    if (!is_valid_block(block_idx)) {
        return NULL; // ��Ч������
    }
    if (node_idx < 0 || node_idx >= block_used_nodes[block_idx]) {
        return NULL; // ��Ч�ڵ�����
    }
    return &nodes_pool[block_idx * nodes_per_block + node_idx];
}

// ɾ���飨�߼�ɾ����
template <typename T>
__host__ __device__ bool mmpool<T>::remove_block(int block_idx) {
    if (!is_valid_block(block_idx)) {
        return false; // ��Ч������
    }

    block_used_nodes[block_idx] = 0; // �߼�ɾ�������Ϊδʹ��
    return true;
}

#endif