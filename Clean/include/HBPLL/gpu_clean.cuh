#pragma once

#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <graph_v_of_v/graph_v_of_v.h>
#include <tool_functions/cuda_vector.cuh>

//#include "HBPLL/test.h"

using std::vector;
using thrust::device_vector;

__global__ void query_label(label* L, long long start, long long end, int i, int h_v, int* Lc_hashed, int* d_uv, int V, int K);
__global__ void clean_kernel(int V, int K, int tc, label* L, long long* L_start, cuda_vector<label>** Lc, int* hash_array, int* d_uv);

cuda_vector<label>** gpu_clean(graph_v_of_v<int>& input_graph, vector<vector<label>>& uncleaned_L, int tc, int K);
