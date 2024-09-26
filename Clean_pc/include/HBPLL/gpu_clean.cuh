#pragma once

#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <graph_v_of_v/graph_v_of_v.h>
//#include <HBPLL/cuda_vector.cuh>

#include "HBPLL/test.h"


#define label hop_constrained_two_hop_label

// struct node{
//     int node_id; // 此node在原有
// };

// __device__ int query_label(label* L, long long start, long long end, int i, int h_v, int* Lc_hashed, int V, int K);
// __global__ void clean_kernel(int V, int K, int tc, label* L, long long* L_start, int* hash_array,int *mark);

double gpu_clean(graph_v_of_v<int>& input_graph, vector<vector<label>>& input_L,vector<vector<hop_constrained_two_hop_label>>& res, int tc, int K);
double gpu_clean_old(graph_v_of_v<int> &input_graph, vector<vector<label>> &input_L, vector<vector<hop_constrained_two_hop_label>> &res, int tc, int K);