#ifndef GEN_LABEL_CUH
#define GEN_LABEL_CUH

#pragma once
#include <graph/csr_graph.hpp>
#include <label/global_labels_v2.cuh>
#include "memoryManagement/cuda_hashtable_v2.cuh"
#include "memoryManagement/cuda_vector_v2.cuh"
#include "memoryManagement/mmpool_v2.cuh"
#include "test/test_mmpool.cuh"
#include <text_mining/ThreadPool.h>

__device__ int query_dis_by_hash_table (int u, int v, cuda_hashTable_v2<weight_type> *H, cuda_vector_v2<hub_type> *L, int hop_now, int hop_cst);

void label_gen (CSR_graph<weight_type>& input_graph, hop_constrained_case_info_v2 *info, vector<vector<hub_type> >&L, vector<int>& nid);

__global__ void gen_label_hsdl (int V, int thread_num, int hop_cst, int hop_now, int* out_pointer, int* out_edge, int* out_edge_weight,
            cuda_vector_v2<hub_type> *L_gpu, cuda_hashTable_v2<weight_type> *Has, cuda_vector_v2<T_item> *T0, cuda_vector_v2<T_item> *T1);

__global__ void gen_label_hsdl_v2 (int V, int thread_num, int hop_cst, int hop_now, int* out_pointer, int* out_edge, int* out_edge_weight,
            cuda_vector_v2<hub_type> *L_gpu, cuda_hashTable_v2<weight_type> *Has, cuda_hashTable_v2<weight_type> *Das,
            cuda_vector_v2<T_item> *T0, cuda_vector_v2<T_item> *T1, int *d);

__device__ int query_dis_by_hash_table (cuda_vector_v2<hub_type> *L_gpu);

__global__ void clear_T (int V, cuda_vector_v2<T_item> *T, cuda_vector_v2<T_item> *D);

__global__ void query_parallel (int sv, int st, int ed, int sz, cuda_hashTable_v2<weight_type> *das, int *d,
cuda_hashTable_v2<weight_type> *has, cuda_vector_v2<hub_type> *L_gpu, cuda_vector_v2<T_item> *t1, int hop_now, int hop_cst);

#endif