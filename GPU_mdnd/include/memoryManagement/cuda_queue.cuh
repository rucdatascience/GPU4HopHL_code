#ifndef CUDA_QUEUE_CUH
#define CUDA_QUEUE_CUH

#include "definition/hub_def.h"
#include "definition/mmpool_size.h"
#include "memoryManagement/cuda_vector.cuh"
#include "memoryManagement/mmpool.cuh"
#include <cassert>
#include <cstddef>
#include <cuda_runtime.h>
#include <iostream>

template <typename T> class cuda_queue {
public:
    mmpool<T> *pool = NULL;
    cuda_vector<T> *data = NULL;

    int *front = NULL; // �����������ڵ�block idx
    int *rear = NULL;  // ��β�������ڵ�block idx
    int *start = NULL; // ÿ��block����ʼ����
    int *end = NULL;   // ÿ��block�Ľ�������
    size_t size; //����block�����������úܴ�

    cuda_queue(size_t size, mmpool<T> *pool) : pool(pool) {
        cudaMallocManaged(&front, sizeof(int));
        cudaMallocManaged(&rear, sizeof(int));
        cudaMallocManaged(&data, sizeof(cuda_vector<T>));

        cudaMallocManaged(&start, size * sizeof(int));
        cudaMallocManaged(&end, size * sizeof(int));
        new (data) cuda_vector<T>(pool, 1);

        *front = 0;
        *rear = 0;

        if (!data->resize(1)) {
            printf("resize failed\n");
            printf("data->blocks: %d\n", data->blocks);
            assert(false);
        }
        this->size = size;
    }

    ~cuda_queue() {
        cudaFree(front);
        cudaFree(rear);
        data->~cuda_vector();
        cudaFree(data);
    }

    __device__ bool enqueue(const T item) {
        //�ҵ�rear block
        int idx = *rear;
        //������ˣ�����
        if (is_full_block(idx)) {
            if (!extend()) {
                return false;
            }
            idx++;
        }
        //��item���뵽��β
        *(data->get(idx * nodes_per_block + end[idx])) = item;
        end[idx] = (end[idx] + 1) % nodes_per_block;
        return true;
    }

    __device__ __host__ bool extend() {
        if (is_full_block(*rear)) {
            //�����µĿ�
            if (!data->resize(data->blocks + 1)) {
                return false;
            }
            if(size < data->blocks){
                printf("size: %d, data->blocks: %d\n", size, data->blocks);
                assert(false);
            }
            //��β��ָ��ָ���µĿ�
            *rear = data->blocks - 1;
            start[*rear] = 0;
            end[*rear] = 0;
        }
        return true;
    }

    __device__ bool dequeue(T *item) {
        //�ҵ�front block
        int idx = *front;
        //������ˣ�����false
        if (is_empty()) {
            return false;
        }
        //������Ԫ�س���
        *item = *(data->get(idx * nodes_per_block + start[idx]));
        start[idx] = (start[idx] + 1) % nodes_per_block;
        //���blockΪ�գ�frontָ�����
        if (is_empty_block(idx)) {
            *front = (*front + 1) % data->blocks;
        }
        return true;
    }

    __device__ bool is_empty() const { //�����Ϊ��
        int idx = *front;
        return (*front == *rear) && (start[idx] == end[idx]);
    }

    __device__ bool is_full_block(int idx) const {
        // block��front��rear+1��ͬ��˵������
        return ((end[idx] + 1) % nodes_per_block == start[idx]);
    }
    __device__ bool is_empty_block(int idx) const {
        // block��front��rear��ͬ��˵��Ϊ��
        return (end[idx] == start[idx]);
    }

    __device__ __host__ bool clear() {
        *front = 0;
        *rear = 0;
        for (int i = 0; i < data->blocks; i++) {
            start[i] = 0;
            end[i] = 0;
        }
        return true;
    }
    
};

//��ʽ����ģ����
template <typename hub_type> class cuda_queue;

#endif