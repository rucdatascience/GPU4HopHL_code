#pragma once

#include <vector>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <graph_v_of_v/graph_v_of_v.h>
#include <HBPLL/cuda_vector.cuh>

#include "HBPLL/test.h"

using std::vector;
using thrust::device_vector;

class query_info {
public:
    // start, end, hop
    int s, t, h;
    query_info (int s, int t, int h) : s(s), t(t), h(h) {}
    query_info () { s = t = h = 0; }
};

double gpu_query(graph_v_of_v<int>& input_graph, vector<vector<label>>& input_L, int query_num, query_info* que, int *ans_gpu, int K);
