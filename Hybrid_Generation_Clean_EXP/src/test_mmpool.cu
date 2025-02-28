#include "test/test_mmpool.cuh"

__global__ void test_hashtable (cuda_hashTable_v2<int> *Has) {
    // Has[0].modify(10, 1);
    // Has[0].modify(10, 1);
    // Has[0].modify(10, 1);
    // printf("test hash 1: %d\n", Has[0].get(10));
    // printf("test hash 2: %d\n", Has[0].get(11));
    // printf("test hash 4: %d\n", Has[1].get(10));
    // Has[1].modify(10, 1);
    // printf("test hash 5: %d\n", Has[1].get(10));
    // Has[1].modify(10, 1e9);
    // printf("test hash 6: %d\n", Has[1].get(10));
    // printf("test hash 7: %d\n", Has[0].get(10));
}

__global__ void test_vector_insert (cuda_vector_v2<hub_type> *L_gpu) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // L_gpu[0].push_back({tid, tid, tid});
    // printf("insert successful!\n");
}

__global__ void test_vector_print (cuda_vector_v2<hub_type> *L_gpu) {
    // printf("test_vector_print !!!\n");
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("test_vector_print: %d\n", tid);
    // cuda_vector_v2<hub_type> *L = (L_gpu + tid);
    // for (int i = 0; i < L->blocks_num; ++i) {
    //     int block_id = L->block_idx_array[i];
    //     int block_siz = L->pool->get_block_size(block_id);
    //     // printf("tid, block_id, block_siz: %d %d %d\n", tid, block_id, block_siz);
    //     for (int j = 0; j < block_siz; ++j) {
    //         hub_type* x = L->pool->get_node(block_id, j);
    //         printf("%d %d %d %d %d\n", tid, x->hop, x->hub_vertex, x->distance);
    //     }
    // }
}

__global__ void test_vector_clear (int V, cuda_vector_v2<hub_type> *L_gpu) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // if (tid < V) {   
    //     L_gpu[tid].init(V, tid);
    // }
}

void test_mmpool (int V, const int &thread_num, const int &test_type, hop_constrained_case_info_v2 *info, cuda_hashTable_v2<int> *L_hash) {
    cudaError_t err;
    
    // if (test_type == 1) {
    //     // 测试 cuda_hash 的部分，修改并查询，并行。
    //     test_hashtable <<< 1, thread_num>>> (L_hash);
    // } else if (test_type == 2) {
    //     // 测试 cuda_vector 的部分，先插入，再输出，并行。
    //     test_vector_insert <<< 30, 30 >>> (info->L_cuda);
    //     cudaDeviceSynchronize();
    //     test_vector_print <<< 1, V >>> (info->L_cuda);
    //     cudaDeviceSynchronize();
    // } else if (test_type == 3) {
    //     // 测试 cuda_vector 的部分，先插入，再clear，再插入，最后输出，并行。
    //     test_vector_insert <<< 2, thread_num / 2 >>> (info->L_cuda);
    //     cudaDeviceSynchronize();
    //     printf("------------------------------------\n");
    //     test_vector_clear <<< 1, thread_num >>> (V, info->L_cuda);
    //     cudaDeviceSynchronize();
    //     printf("------------------------------------\n");
    //     test_vector_insert <<< 1, thread_num >>> (info->L_cuda);
    //     cudaDeviceSynchronize();
    //     printf("------------------------------------\n");
    //     test_vector_print <<< 1, V >>> (info->L_cuda);
    //     cudaDeviceSynchronize();
    // }
}
