#ifndef TEST_MMPOOL_CUH
#define TEST_MMPOOL_CUH

#include "definition/hub_def.h"
#include "label/hop_constrained_two_hop_labels_v2.cuh"
#include "memoryManagement/cuda_hashtable_v2.cuh"
#include "memoryManagement/cuda_vector_v2.cuh"
#include "memoryManagement/mmpool_v2.cuh"
#include "label/global_labels_v2.cuh"

__global__ void test_hashtable (cuda_hashTable_v2<int> *Has);

__global__ void test_vector_insert (cuda_vector_v2<hub_type> *L_gpu);

__global__ void test_vector_print (cuda_vector_v2<hub_type> *L_gpu);

__global__ void test_vector_clear (int V, cuda_vector_v2<hub_type> *L_gpu);

void test_mmpool(int V, const int &thread_num, const int &test_type, hop_constrained_case_info_v2 *info, cuda_hashTable_v2<int> *L_hash);

#endif