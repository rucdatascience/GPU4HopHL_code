#ifndef CUDA_VECTOR_CUH
#define CUDA_VECTOR_CUH

#include "memoryManagement/mmpool.cuh"
#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>
// include log
#include "definition/hub_def.h"
#include "label/hop_constrained_two_hop_labels.cuh"
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
// log function which can be used in device code
inline __host__ __device__ void mylog(const char *message) {
  printf("%s\n", message);
}
//#define data_type hop_constrained_two_hop_label

template <typename T> class cuda_vector {
public:
    mmpool<T> *pool = NULL;
    size_t current_size;
    size_t capacity;      // ��ǰvector������, �Կ�Ϊ��λ
    int *block_idx_array = NULL; // unified memory
    size_t blocks;
    int lock;
    T *first_elements = NULL; // unified memory
                        // ptr,��thrust������󣬽����ݿ���������Ա���device��ʹ��

    __host__ cuda_vector(mmpool<T> *pool, size_t capacity = 100); // ��ʼ���С���Ը����������

    __host__ ~cuda_vector();
    //__host__ void resize(size_t new_size); //������pool����ǰ����ָ�������Ŀ�

    __device__ bool push_back(const T &value);
    __device__ __host__ T *get(size_t index);
    // const T& operator[](size_t index) const;
    __host__ void clear();
    __host__ __device__ size_t size() const { return current_size; }
    __device__ bool empty() const { return current_size == 0; }
    // __host__ void copy_to_cpu(size_t index, T *cpu_ptr);
    __host__ void sort_label(); //��hostʹ��
    __host__ __device__ bool resize(size_t new_size);
    __host__ void prefetch() {
        for (int i = 0; i < blocks; i++) {
            pool->prefetch(block_idx_array[i]);
        }
    }
};

template <typename T>
__host__ cuda_vector<T>::cuda_vector(mmpool<T> *pool, size_t capacity) : pool(pool) {
    this->blocks = 0;
    this->current_size = 0;
    this->capacity = capacity;
    this->lock = 0;
    
    //�������
    int block_idx = pool->find_available_block();
    // printf("block_idx:%d\n\n", block_idx);
    if (block_idx == -1) {
        //û�п��У�����ʧ��
        mylog("No available block in mmpool");
        assert(false);
        return;
    }

    // copy to cuda
    cudaMallocManaged(&this->block_idx_array, sizeof(int) * capacity);
    this->block_idx_array[this->blocks] = block_idx;
    this->blocks += 1;
};

template <typename T>
__device__ bool cuda_vector<T>::push_back(const T &value) {
    // ���˲�������һ�������У���ȷ���̰߳�ȫ
    // ����ʼ
    while (atomicCAS(&this->lock, 0, 1) != 0);

    //�ҵ���ǰvector�����һ���ڵ�
    int last_block_idx = this->block_idx_array[this->blocks - 1];
    // printf("last_block_idx:%d\n", last_block_idx);
    //�жϵ�ǰ���Ƿ�����
    if (pool->is_full_block(last_block_idx)) {
        //��ǰ�������������¿�
        int block_idx = pool->find_available_block();
        if (block_idx == -1) {
            //û�п��У�����ʧ��
            mylog("No available block in mmpool");
            atomicExch(&this->lock, 0);
            return false;
        }
        this->block_idx_array[this->blocks++] = block_idx;
        last_block_idx = block_idx;
    }
    //��ӽڵ�
    if (this->pool->push_node(last_block_idx, value)) {
        this->current_size++;
        atomicExch(&this->lock, 0);
        return true;
    }

    atomicExch(&this->lock, 0);
    return false;
};

template <typename T> __device__ __host__ T *cuda_vector<T>::get(size_t index) {
    if (index >= this->current_size) {
        mylog("Index out of range");
        // error
        cudaError_t error = cudaGetLastError();

        assert(error == cudaErrorMemoryAllocation);
        // return nullptr;
    }
    //�ҵ���Ӧ�Ŀ�
    int block_idx = this->block_idx_array[index / pool->get_nodes_per_block()];
    //�ҵ���Ӧ�Ľڵ�
    int node_idx = index % pool->get_nodes_per_block();
    //���ؽڵ�
    // printf("block_idx:%d, node_idx:%d\n", block_idx, node_idx);
    return pool->get_node(block_idx, node_idx);
};

template <typename T> __host__ void cuda_vector<T>::clear() {
    //�ͷ����п�
    for (int i = 0; i < this->blocks; i++) {
        pool->remove_block(this->block_idx_array[i]);
    }
    //�ͷ�block_idx_array
    // delete[] this->block_idx_array;
    this->blocks = 0;
    this->current_size = 0;
};

template <typename T> __host__ cuda_vector<T>::~cuda_vector() {
    clear();
    cudaFree(this->block_idx_array);
    // first_elements->clear();��
    // free(this->first_elements);��������cuda label���ͷ�
};

template <typename T> __host__ __device__ bool cuda_vector<T>::resize(size_t new_size) {
    //�ڳ�ʼ������������resize()��������ǲ���Ҫ����Ƿ����㹻�Ŀ�
    if (this->blocks == 0) {
        return false;
    }

    if (this->blocks == 1) {
        //�ոճ�ʼ���꣬��Ϊ��
        pool->set_block_user_nodes(this->block_idx_array[0], nodes_per_block);
    }
    if (new_size <= this->blocks) {
        this->blocks = new_size;
        this->current_size = new_size * nodes_per_block;
        return true;
    }
    while (this->blocks < new_size) {
        int block_idx = pool->find_available_block();
        if (block_idx == -1) {
            //û�п��У�����ʧ��
            mylog("No available block in mmpool");
            assert(false);
            return false;
        }
        this->block_idx_array[this->blocks++] = block_idx;
        // cudaMemcpy(this->block_idx_array + this->blocks, &block_idx, sizeof(int),
        //            cudaMemcpyHostToDevice);
        // this->blocks += 1;
        pool->set_block_user_nodes(block_idx, nodes_per_block);
    }
    this->current_size = new_size * nodes_per_block;
    return true;
}

//��ʽ����ģ����
// template class cuda_vector<int>;
// template class cuda_vector<float>;
template <typename hub_type> class cuda_vector;

#endif