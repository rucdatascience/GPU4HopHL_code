#ifndef GEN_LABEL_CUH
#define GEN_LABEL_CUH

#include <graph/csr_graph.hpp>
#include <label/global_labels_v2.cuh>
#include "memoryManagement/cuda_hashtable_v2.cuh"
#include "memoryManagement/cuda_vector_v2.cuh"
#include "memoryManagement/mmpool_v2.cuh"
#include "test/test_mmpool.cuh"

__device__ int query_dis_by_hash_table (int u, int v, cuda_hashTable_v2<int> *H, cuda_vector_v2<hub_type> *L, int hop_now, int hop_cst);

void label_gen (CSR_graph<weight_type>& input_graph, hop_constrained_case_info_v2 *info, int hop_cst, 
vector<vector<hub_type> >&L);

__global__ void gen_label_hsdl (int V, int thread_num, int hop_cst, int hop_now, int* out_pointer, int* out_edge, int* out_edge_weight,
            cuda_hashTable_v2<int> *Has, cuda_vector_v2<hub_type> *L_gpu, cuda_vector_v2<hub_type> *T0, cuda_vector_v2<hub_type> *T1);

__device__ int query_dis_by_hash_table (cuda_vector_v2<hub_type> *L_gpu);

__global__ void init_T (int V, cuda_vector_v2<hub_type> *T);

__global__ void clear_T (int V, cuda_vector_v2<hub_type> *T);

#endif