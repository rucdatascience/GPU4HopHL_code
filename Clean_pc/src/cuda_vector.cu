#include <HBPLL/cuda_vector.cuh>
#include <cuda_runtime.h>

template <typename T>
__host__ base_memory<T>::base_memory(size_t size_per_block, size_t capacity) {
    this->size_per_block = size_per_block;
    this->capacity = capacity;
    cudaMallocManaged(&data, size_per_block * capacity * sizeof(T));
    cudaMallocManaged(&block_size, capacity * sizeof(int));
    cudaMemset(block_size, -1, capacity * sizeof(int));
}

template <typename T>
__device__ int base_memory<T>::find_set_available_block() {
    bool blocked = true;
    int block_idx = -1;
    while (blocked) {
        if (atomicCAS(&lock, 0, 1) == 0) {
            for (int i = 0; i < capacity; i++) {
                if (block_size[i] == -1) {
                    block_size[i] = 0;
                    __threadfence();
                    atomicExch(&lock, 0);
                    blocked = false;
                    block_idx = i;
                    break;
                }
            }
            atomicExch(&lock, 0);
            blocked = false;
        }
    }
    return block_idx;
}

template <typename T>
__host__ __device__ base_memory<T>::~base_memory() {
    cudaFree(data);
    cudaFree(block_size);
}

template <typename T>
__host__ cuda_vector<T>::cuda_vector(base_memory<T>* pool, int max_blocks) : pool(pool) {
    this->blocks = 0;
    this->current_size = 0;
    this->blocks_upper_bound = max_blocks;

    cudaMallocManaged(&this->block_idx_array, sizeof(int) * blocks_upper_bound);
}

template <typename T>
__host__ __device__ T cuda_vector<T>::operator[](size_t index) {
    if (index >= this->current_size) {
        printf("index %lu is out of bound %lu\n", index, this->current_size);
        assert(false);
    }
    int block_idx = index / pool->size_per_block;
    int idx = index % pool->size_per_block;
    return pool->data[pool->size_per_block * block_idx_array[block_idx] + idx];
}

template <typename T>
__device__ void cuda_vector<T>::push_back(const T& value) {
    if (current_size % pool->size_per_block == 0) {
        int block_idx = pool->find_set_available_block();
        if (block_idx == -1) {
            printf("no available block\n");
            assert(false);
            return;
        }
        block_idx_array[blocks] = block_idx;
        blocks += 1;
    }
    int block_idx = current_size / pool->size_per_block;
    int idx = current_size % pool->size_per_block;
    pool->data[block_idx_array[block_idx] * pool->size_per_block + idx] = value;
    pool->block_size[block_idx_array[block_idx]] += 1;
    current_size++;
}

template <typename T>
__host__ __device__ cuda_vector<T>::~cuda_vector() {
    cudaFree(block_idx_array);
}
