#ifndef CSR_GRAPH_HPP
#define CSR_GRAPH_HPP
#pragma once

#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include <vector>
// #include <graph_structure/graph_structure.hpp>
#include <graph/ldbc.hpp>
#include <graph_v_of_v/graph_v_of_v.h>

/*for GPU*/
template <typename weight_type>
class CSR_graph
{
public:
    std::vector<int> INs_Neighbor_start_pointers, OUTs_Neighbor_start_pointers, ALL_start_pointers; // Neighbor_start_pointers[i] is the start point of neighbor information of vertex i in Edges and Edge_weights
    /*
        Now, Neighbor_sizes[i] = Neighbor_start_pointers[i + 1] - Neighbor_start_pointers[i].
        And Neighbor_start_pointers[V] = Edges.size() = Edge_weights.size() = the total number of edges.
    */
    std::vector<int> INs_Edges, OUTs_Edges, all_Edges;                       // Edges[Neighbor_start_pointers[i]] is the start of Neighbor_sizes[i] neighbor IDs
    std::vector<weight_type> INs_Edge_weights, OUTs_Edge_weights; // Edge_weights[Neighbor_start_pointers[i]] is the start of Neighbor_sizes[i] edge weights
    int *in_pointer, *out_pointer, *in_edge, *out_edge, *all_pointer, *all_edge;
    int *in_edge_weight, *out_edge_weight;
    int E_all;

    __host__ void destroy_csr_graph () {
        cudaFree(in_pointer), cudaFree(out_pointer);
        cudaFree(in_edge), cudaFree(out_edge);
        cudaFree(out_edge), cudaFree(all_edge);
        cudaFree(in_edge_weight), cudaFree(out_edge_weight);
    }
};

template <typename weight_type>
// CSR_graph<weight_type> toCSR(graph_structure<weight_type>& graph)
CSR_graph<weight_type> toCSR(LDBC<weight_type> &graph)
{

    CSR_graph<weight_type> ARRAY;

    int V = graph.size();
    
    ARRAY.INs_Neighbor_start_pointers.resize(V + 1); // Neighbor_start_pointers[V] = Edges.size() = Edge_weights.size() = the total number of edges.
    ARRAY.OUTs_Neighbor_start_pointers.resize(V + 1);
    ARRAY.ALL_start_pointers.resize(V + 1);
    
    int pointer = 0;
    for (int i = 0; i < V; i++)
    {
        ARRAY.INs_Neighbor_start_pointers[i] = pointer;
        for (auto &xx : graph.INs[i])
        {
            ARRAY.INs_Edges.push_back(xx.first);
            ARRAY.INs_Edge_weights.push_back(xx.second);
        }
        pointer += graph.INs[i].size();
    }
    ARRAY.INs_Neighbor_start_pointers[V] = pointer;
    
    pointer = 0;
    for (int i = 0; i < V; i++)
    {
        ARRAY.OUTs_Neighbor_start_pointers[i] = pointer;
        for (auto &xx : graph.OUTs[i])
        {
            ARRAY.OUTs_Edges.push_back(xx.first);
            ARRAY.OUTs_Edge_weights.push_back(xx.second);
        }
        pointer += graph.OUTs[i].size();
    }
    ARRAY.OUTs_Neighbor_start_pointers[V] = pointer;
    
    pointer = 0;
    for (int i = 0; i < V; i++)
    {
        ARRAY.ALL_start_pointers[i] = pointer;
        for (auto &xx : graph.INs[i])
        {
            ARRAY.all_Edges.push_back(xx.first);
        }
        for (auto &xx : graph.OUTs[i])
        {
            ARRAY.all_Edges.push_back(xx.first);
        }
        pointer += graph.INs[i].size() + graph.OUTs[i].size();
    }
    ARRAY.ALL_start_pointers[V] = pointer;
    
    int E_in = ARRAY.INs_Edges.size();
    int E_out = ARRAY.OUTs_Edges.size();
    int E_all = E_in + E_out;
    ARRAY.E_all = E_all;
    cudaMallocManaged(&ARRAY.in_pointer, (V + 1) * sizeof(int));
    cudaMallocManaged(&ARRAY.out_pointer, (V + 1) * sizeof(int));
    cudaMallocManaged(&ARRAY.all_pointer, (V + 1) * sizeof(int));
    cudaMallocManaged(&ARRAY.in_edge, E_in * sizeof(int));
    cudaMallocManaged(&ARRAY.out_edge, E_out * sizeof(int));
    cudaMallocManaged(&ARRAY.all_edge, E_all * sizeof(int));
    cudaMallocManaged(&ARRAY.in_edge_weight, E_in * sizeof(int));
    cudaMallocManaged(&ARRAY.out_edge_weight, E_out * sizeof(int));
    
    cudaMemcpy(ARRAY.in_pointer, ARRAY.INs_Neighbor_start_pointers.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ARRAY.out_pointer, ARRAY.OUTs_Neighbor_start_pointers.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ARRAY.all_pointer, ARRAY.ALL_start_pointers.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ARRAY.in_edge, ARRAY.INs_Edges.data(), E_in * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ARRAY.out_edge, ARRAY.OUTs_Edges.data(), E_out * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ARRAY.all_edge, ARRAY.all_Edges.data(), E_all * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ARRAY.in_edge_weight, ARRAY.INs_Edge_weights.data(), E_in * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ARRAY.out_edge_weight, ARRAY.OUTs_Edge_weights.data(), E_out * sizeof(int), cudaMemcpyHostToDevice);
    
    return ARRAY;
}

template <typename weight_type>
CSR_graph<weight_type> graph_v_of_v_to_CSR(graph_v_of_v<weight_type> &g) {
    
    CSR_graph<weight_type> ARRAY;

    int V = g.ADJs.size();
    ARRAY.INs_Neighbor_start_pointers.resize(V + 1);
    ARRAY.OUTs_Neighbor_start_pointers.resize(V + 1);
    ARRAY.ALL_start_pointers.resize(V + 1);

    int pointer = 0;
    for (int i = 0; i < V; i++) {
        ARRAY.INs_Neighbor_start_pointers[i] = pointer;
        for (auto &xx : g.ADJs[i]) {
            ARRAY.INs_Edges.push_back(xx.first);
            ARRAY.INs_Edge_weights.push_back(xx.second);
        }
        pointer += g.ADJs[i].size();
    }
    ARRAY.INs_Neighbor_start_pointers[V] = pointer;
    
    ARRAY.OUTs_Edges =  ARRAY.INs_Edges;
    ARRAY.OUTs_Edge_weights = ARRAY.INs_Edge_weights;
    ARRAY.OUTs_Neighbor_start_pointers = ARRAY.INs_Neighbor_start_pointers;


    // pointer = 0;
    // for (int i = 0; i < V; i++) {
    //   ARRAY.OUTs_Neighbor_start_pointers[i] = pointer;
    //   for (auto &xx : g.ADJs[i]) {
    //     ARRAY.OUTs_Edges.push_back(xx.first);
    //     ARRAY.OUTs_Edge_weights.push_back(xx.second);
    //   }
    //   pointer += g.ADJs[i].size();
    // }
    // ARRAY.OUTs_Neighbor_start_pointers[V] = pointer;

    pointer = 0;
    for (int i = 0; i < V; i++) {
        ARRAY.ALL_start_pointers[i] = pointer;
        int IN_start = ARRAY.INs_Neighbor_start_pointers[i];
        int IN_end = ARRAY.INs_Neighbor_start_pointers[i+1];

        ARRAY.all_Edges.insert(ARRAY.all_Edges.end(),ARRAY.INs_Edges.begin()+IN_start,ARRAY.INs_Edges.begin()+IN_end);
        ARRAY.all_Edges.insert(ARRAY.all_Edges.end(),ARRAY.INs_Edges.begin()+IN_start,ARRAY.INs_Edges.begin()+IN_end);
        pointer += (IN_end-IN_start)*2;
    }
    ARRAY.ALL_start_pointers[V] = pointer;

    size_t E_in = ARRAY.INs_Edges.size();
    size_t E_out = ARRAY.OUTs_Edges.size();
    size_t E_all = E_in + E_out;
    ARRAY.E_all = E_all;

    cudaMallocManaged((void **)&ARRAY.in_pointer, (V + 1) * sizeof(int));
    cudaMallocManaged((void **)&ARRAY.out_pointer, (V + 1) * sizeof(int));
    cudaMallocManaged((void **)&ARRAY.all_pointer, (V + 1) * sizeof(int));
    cudaMallocManaged((void **)&ARRAY.in_edge, E_in * sizeof(int));
    cudaMallocManaged((void **)&ARRAY.out_edge, E_out * sizeof(int));
    cudaMallocManaged((void **)&ARRAY.all_edge, E_all * sizeof(int));
    cudaMallocManaged((void **)&ARRAY.in_edge_weight, E_in * sizeof(int));
    cudaMallocManaged((void **)&ARRAY.out_edge_weight, E_out * sizeof(int));

    cudaDeviceSynchronize();
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
    }

    cudaMemcpy(ARRAY.in_pointer, ARRAY.INs_Neighbor_start_pointers.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ARRAY.out_pointer, ARRAY.OUTs_Neighbor_start_pointers.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ARRAY.all_pointer, ARRAY.ALL_start_pointers.data(), (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ARRAY.in_edge, ARRAY.INs_Edges.data(), E_in * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ARRAY.out_edge, ARRAY.OUTs_Edges.data(), E_out * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ARRAY.all_edge, ARRAY.all_Edges.data(), E_all * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ARRAY.in_edge_weight, ARRAY.INs_Edge_weights.data(), E_in * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(ARRAY.out_edge_weight, ARRAY.OUTs_Edge_weights.data(), E_out * sizeof(int), cudaMemcpyHostToDevice);

    return ARRAY;
}

#endif