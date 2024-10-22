#ifndef GPU_GRAPH_CUH
#define GPU_GRAPH_CUH
#pragma once

#include "definition/hub_def.h"
#include "graph/graph_v_of_v.h"
#include <cuda_runtime.h>
#include <iostream>
#include <vector>

struct Edge {
    int target;
    weight_type weight;
};

class gpu_Graph {
public:
    gpu_Graph();
    ~gpu_Graph();

    void allocateGraphOnGPU(int num_nodes, int num_edges);
    void copyGraphToGPU();
    gpu_Graph(const std::vector<std::vector<std::pair<int, weight_type>>> &ADJs);
    void freeGraphOnGPU();

    //cpu
    int num_nodes;
    int num_edges;
    int max_degree;

    Edge *edges;  // �洢���бߵ�Ŀ��ڵ��Ȩ��
    int *offsets; // ÿ���ڵ�ıߵ���ʼλ��

    // gpu
    Edge *d_edges;
    int *d_offsets;

};

#endif // GPU_GRAPH_H