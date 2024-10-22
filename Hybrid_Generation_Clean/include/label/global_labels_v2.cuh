#ifndef GLOBAL_LABELS_V2_CUH
#define GLOBAL_LABELS_V2_CUH

#include "definition/hub_def.h"
#include "label/hop_constrained_two_hop_labels_v2.cuh"
#include "memoryManagement/cuda_hashtable_v2.cuh"
#include "memoryManagement/cuda_vector_v2.cuh"
#include "memoryManagement/mmpool_v2.cuh"
#include <cuda_runtime.h>

class hop_constrained_case_info_v2 {
public:
    /* labels */
    mmpool_v2<hub_type> *mmpool_labels = NULL;
    mmpool_v2<T_item> *mmpool_T0 = NULL;
    mmpool_v2<T_item> *mmpool_T1 = NULL;

    cuda_vector_v2<hub_type> *L_cuda = NULL; // gpu res
    cuda_vector_v2<T_item> *T0 = NULL; // T0
    cuda_vector_v2<T_item> *T1 = NULL; // T1

    cuda_hashTable_v2<weight_type> *L_hash;
    cuda_hashTable_v2<weight_type> *D_hash;

    int **nid;
    int *nid_size;

    int *D_vector;
    int *D_pare;
    
    int *Num_T; // Num_T, Test use
    int *Num_L;
    pair<int, int> *T_push_back;
    pair<int, int> *L_push_back;

    /*hop bounded*/
    int thread_num = 1;
    int hop_cst = 0;
    int use_d_optimization = 0;

    /*running time records*/
	double time_initialization = 0;
	double time_generate_labels = 0;
    double time_traverse_labels = 0;
	double time_total = 0;
    double label_size = 0;

    int L_size;
    int G_max;

    hop_constrained_case_info_v2() {}

    // Constructor
    // mmpool_size_block is the total number of elements to store
    // nodes_per_block is the required number of blocks
    __host__ void init (int V, int hop_cst, int G_max, int thread_num, vector<vector<int> > graph_group) {
        
        cudaError_t err;
        size_t free_byte, total_byte;

        L_size = V;
        G_max = G_max;

        // Create three memory pools
        // The first memory pool is used to store labels
        cudaMallocManaged(&mmpool_labels, sizeof(mmpool_v2<hub_type>));
        cudaDeviceSynchronize();
        new (mmpool_labels) mmpool_v2<hub_type> (V, max((long long)V, (long long) G_max * V * (hop_cst) / 2 / nodes_per_block));
        cudaDeviceSynchronize();

        // The second memory pool is used to store T0
        cudaMallocManaged(&mmpool_T0, sizeof(mmpool_v2<T_item>));
        cudaDeviceSynchronize();
        new (mmpool_T0) mmpool_v2<T_item> (G_max, (long long) G_max * V / 2 / nodes_per_block);
        cudaDeviceSynchronize();

        // The second memory pool is used to store T1
        cudaMallocManaged(&mmpool_T1, sizeof(mmpool_v2<T_item>));
        cudaDeviceSynchronize();
        new (mmpool_T1) mmpool_v2<T_item> (G_max, (long long) G_max * V / 2 / nodes_per_block);
        cudaDeviceSynchronize();

        // Allocate the L_cuda memory pool
        cudaMallocManaged(&L_cuda, (long long) V * sizeof(cuda_vector_v2<hub_type>)); // Allocate n cuda_vector Pointers
        cudaDeviceSynchronize();
        for (int i = 0; i < V; i++) {
            new (L_cuda + i) cuda_vector_v2<hub_type> (mmpool_labels, i, G_max * hop_cst / nodes_per_block + 1);
        }
        cudaDeviceSynchronize();

        // Allocate the T0 memory pool
        cudaMallocManaged(&T0, (long long) G_max * sizeof(cuda_vector_v2<T_item>)); // Allocate n cuda_vector Pointers
        cudaDeviceSynchronize();
        for (int i = 0; i < G_max; i++) {
            new (T0 + i) cuda_vector_v2<T_item> (mmpool_T0, i, V / 2 / nodes_per_block + 1);
        }
        cudaDeviceSynchronize();

        // Allocate the T1 memory pool
        cudaMallocManaged(&T1, (long long) G_max * sizeof(cuda_vector_v2<T_item>)); // 分配 n 个cuda_vector指针
        cudaDeviceSynchronize();
        for (int i = 0; i < G_max; i++) {
            new (T1 + i) cuda_vector_v2<T_item> (mmpool_T1, i, V / 2 / nodes_per_block + 1);
        }
        cudaDeviceSynchronize();

        // 准备 L_hash
        cudaMallocManaged(&L_hash, (long long) thread_num * sizeof(cuda_hashTable_v2<weight_type>));
        cudaDeviceSynchronize();
        for (int i = 0; i < thread_num; i++) {
            new (L_hash + i) cuda_hashTable_v2 <weight_type> (G_max * (hop_cst + 1));
        }
        cudaDeviceSynchronize();

        // 准备 D_hashTable
        cudaMallocManaged(&D_hash, (long long) thread_num * sizeof(cuda_hashTable_v2<weight_type>));
        cudaDeviceSynchronize();
        for (int i = 0; i < thread_num; i++) {
            new (D_hash + i) cuda_hashTable_v2 <int> (V);
        }
        cudaDeviceSynchronize();
        
        // 准备 D_parent_vertex
        cudaMallocManaged(&D_pare, (long long) thread_num * V * sizeof(int));
        // cudaMemset(D_pare, 0, (long long) thread_num * V * sizeof(int));
        cudaDeviceSynchronize();

        // 准备 D_vector
        cudaMallocManaged(&D_vector, (long long) thread_num * V * sizeof(int));
        // cudaMemset(D_vector, 0, (long long) thread_num * V * sizeof(int));
        cudaDeviceSynchronize();
        
        // 准备 LT_push_back_table
        // cudaMallocManaged(&LT_push_back, (long long) thread_num * V * sizeof(int));
        // cudaDeviceSynchronize();

        cudaMallocManaged(&Num_T, (long long) sizeof(int) * V);
        // cudaMemset(Num_T, 0, (long long) V * sizeof(int));
        cudaDeviceSynchronize();
        cudaMallocManaged(&T_push_back, (long long) thread_num * V * sizeof(pair<int, int>));
        // cudaMemset(T_push_back, 0, (long long) thread_num * V * sizeof(pair<int, int>));
        cudaDeviceSynchronize();

        cudaMallocManaged(&Num_L, (long long) sizeof(int) * V);
        // cudaMemset(Num_L, 0, (long long) V * sizeof(int));
        cudaDeviceSynchronize();
        cudaMallocManaged(&L_push_back, (long long) thread_num * V * sizeof(pair<int, int>));
        // cudaMemset(T_push_back, 0, (long long) thread_num * V * sizeof(pair<int, int>));
        cudaDeviceSynchronize();

	    // cudaMemGetInfo(&free_byte, &total_byte);
        // printf("Device memory: total %ld, free %ld\n", total_byte, free_byte);

        err = cudaGetLastError(); // Check for kernel memory request errors
        if (err != cudaSuccess) {
            printf("init cuda error !: %s\n", cudaGetErrorString(err));
        }
    }

    // Points in label
    inline int cuda_vector_size() {
        return L_size;
    }

    // destructor
    __host__ void destroy_L_cuda() {
        for (int i = 0; i < L_size; ++i) {
            L_cuda[i].~cuda_vector_v2<hub_type>();
        }
        cudaFree(L_cuda);

        for (int i = 0; i < G_max; ++i) {
            T0[i].~cuda_vector_v2<T_item>();
            T1[i].~cuda_vector_v2<T_item>();
        }
        cudaFree(T0);
        cudaFree(T1);

        for (int i = 0; i < thread_num; ++i) {
            L_hash[i].~cuda_hashTable_v2<int>();
            D_hash[i].~cuda_hashTable_v2<int>();
        }
        cudaFree(L_hash);
        cudaFree(D_hash);

        cudaFree(D_vector);
        cudaFree(Num_T);
        cudaFree(T_push_back);
        cudaFree(Num_L);
        cudaFree(L_push_back);

        mmpool_labels->~mmpool_v2();
        mmpool_T0->~mmpool_v2();
        mmpool_T1->~mmpool_v2();
        cudaFree(mmpool_labels);
        cudaFree(mmpool_T0);
        cudaFree(mmpool_T1);

    }

};

#endif