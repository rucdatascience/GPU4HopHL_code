#pragma once

#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <label/global_labels_v2.cuh>
#include <graph_v_of_v/graph_v_of_v.h>
#include <future>
#include <text_mining/ThreadPool.h>
#include <memoryManagement/cuda_vector.cuh>
#include <memoryManagement/graph_pool.hpp>
#include <HBPLL/hop_constrained_two_hop_labels.h>

using std::vector;
using thrust::device_vector;

class gpu_clean_info {
public:
    int hop_cst;
    
    int **nid;
    int *nid_size;

    long long *L_start = nullptr;
    int *node_id = nullptr;
    label *L = nullptr; // label on gpu
    int *mark = nullptr; // mark the label clean state
    int *hash_array = nullptr;
    
};

static std::vector<std::future<int>> results_gpu;
static ThreadPool pool_gpu(100);

__global__ void get_hash (int V, int K, int tc, int start_id, int end_id, hop_constrained_two_hop_label *L, long long *L_start, int *hash_array, int *mark);

__device__ int query_label (hop_constrained_two_hop_label* L, long long start, long long end, int i, int h_v, int* Lc_hashed, int V, int K);

__global__ void clean_kernel (int V, int K, int tc, hop_constrained_two_hop_label* L, long long* L_start, int* hash_array,int *mark);

__global__ void clean_kernel_v2 (int V, int K, int tc, int start_id, int end_id, int *node_id, hop_constrained_two_hop_label *L, long long *L_start, int *hash_array, int *mark);

void gpu_clean_init (graph_v_of_v<int> &input_graph, vector<vector<hop_constrained_two_hop_label>> &input_L, hop_constrained_case_info_v2 * info_gpu, Graph_pool<int>& graph_pool, int tc, int K);

void gpu_clean (graph_v_of_v<int>& input_graph, hop_constrained_case_info_v2 * info_gpu, vector<vector<hop_constrained_two_hop_label>>& res, int tc, int nid_vec_id);