#ifndef GLOBAL_LABELS_V2_CUH
#define GLOBAL_LABELS_V2_CUH
#pragma once

#include "definition/hub_def.h"
#include "label/hop_constrained_two_hop_labels_v2.cuh"
#include "HBPLL/hop_constrained_two_hop_labels.h"
#include "memoryManagement/cuda_hashtable_v2.cuh"
#include "memoryManagement/cuda_vector_v2.cuh"
#include "memoryManagement/mmpool_v2.cuh"
#include <cuda_runtime.h>
#include <memoryManagement/cuda_vector.cuh>

class hop_constrained_case_info_v2 {
public:
// for generation
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
    std::pair<int, int> *T_push_back;
    std::pair<int, int> *L_push_back;

    /*hop bounded*/
    int thread_num = 1;
    int hop_cst = 0;
    int Distributed_Graph_Num = 0;
    int use_2023WWW_GPU_version = 0;
    int use_new_algo = 0;
    
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
    __host__ void init (int V, int hop_cst, int G_max, int thread_num, std::vector<std::vector<int> > graph_group) {
        
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
            new (D_hash + i) cuda_hashTable_v2 <weight_type> (V);
        }
        cudaDeviceSynchronize();
        
        // 准备 D_parent_vertex
        cudaMallocManaged(&D_pare, (long long) thread_num * V * sizeof(int));
        cudaDeviceSynchronize();

        // 准备 D_vector
        cudaMallocManaged(&D_vector, (long long) thread_num * V * sizeof(int));
        cudaDeviceSynchronize();
        
        cudaMallocManaged(&Num_T, (long long) sizeof(int) * V);
        cudaDeviceSynchronize();
        cudaMallocManaged(&T_push_back, (long long) thread_num * V * sizeof(std::pair<int, int>));
        cudaDeviceSynchronize();

        cudaMallocManaged(&Num_L, (long long) sizeof(int) * V);
        cudaDeviceSynchronize();
        cudaMallocManaged(&L_push_back, (long long) thread_num * V * sizeof(std::pair<int, int>));
        cudaDeviceSynchronize();

	    // cudaMemGetInfo(&free_byte, &total_byte);
        // printf("Device memory: total %ld, free %ld\n", total_byte, free_byte);

        err = cudaGetLastError(); // Check for kernel memory request errors
        if (err != cudaSuccess) {
            printf("init cuda error !: %s\n", cudaGetErrorString(err));
        }
    }

    // set nid
    __host__ void set_nid (int distributed_graph_num, std::vector<std::vector<int> > graph_group) {
        Distributed_Graph_Num = distributed_graph_num;
        cudaMallocManaged(&nid, sizeof(int*) * Distributed_Graph_Num);
        cudaMallocManaged(&nid_size, sizeof(int) * Distributed_Graph_Num);
        for (int j = 0; j < Distributed_Graph_Num; ++ j) {
            cudaMallocManaged(&nid[j], sizeof(int) * graph_group[j].size());
            nid_size[j] = graph_group[j].size();
            for (int k = 0; k < graph_group[j].size(); ++k) {
                nid[j][k] = graph_group[j][k];
            }
        }
    }

    // Points in label
    inline int cuda_vector_size() {
        return L_size;
    }

    // destructor
    __host__ void destroy_L_cuda() {
        for (int i = 0; i < L_size; ++i) {
            L_cuda[i].~cuda_vector_v2 <hub_type> ();
        }
        cudaFree(L_cuda);

        for (int i = 0; i < G_max; ++i) {
            T0[i].~cuda_vector_v2 <T_item> ();
            T1[i].~cuda_vector_v2 <T_item> ();
        }
        cudaFree(T0);
        cudaFree(T1);

        for (int i = 0; i < thread_num; ++i) {
            L_hash[i].~cuda_hashTable_v2 <weight_type> ();
            D_hash[i].~cuda_hashTable_v2 <weight_type> ();
        }
        
        for (int i = 0; i < Distributed_Graph_Num; ++ i) {
            cudaFree(nid[i]);
        }
        cudaFree(nid);
        cudaFree(nid_size);

        cudaFree(L_hash);
        cudaFree(D_hash);

        cudaFree(D_vector);
        cudaFree(D_pare);

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

// for clean
    long long *L_start = nullptr;
    long long *L_end = nullptr;
    int *node_id = nullptr;
    int *nid_to_tid = nullptr;
    hop_constrained_two_hop_label *L = nullptr; // label on gpu
    int *mark = nullptr; // mark the label clean state
    int *hash_array = nullptr;

};

#endif