#include "HBPLL/test.h"
#include "HBPLL/gpu_clean.cuh"

int main(int argc,char **argv) {
    int ec_min = 1, ec_max = 10;
    int V = 10, E = 10;
    std::cout << "Input the number of vertices: " << std::endl;
    std::cin >> V;
    std::cout << "Input the number of edges: " << std::endl;
    std::cin >> E;

    graph_v_of_v<int> instance_graph;
    instance_graph = graph_v_of_v_generate_random_graph<int>(V, E, ec_min, ec_max, 1, boost_random_time_seed);
    instance_graph = graph_v_of_v_update_vertexIDs_by_degrees_large_to_small(instance_graph); // sort vertices
    instance_graph.txt_save("simple_iterative_tests.txt");

    hop_constrained_case_info mm;
	mm.upper_k = 5;
	mm.use_rank_prune = 1;
	mm.use_2023WWW_generation = 0;
	mm.use_canonical_repair = 1;
	mm.max_run_time_seconds = 10;
	mm.thread_num = 10;

    test_HSDL(instance_graph);

    vector<vector<hop_constrained_two_hop_label>> uncleaned_L;

    hop_constrained_two_hop_labels_generation(instance_graph, mm, uncleaned_L);
    hop_constrained_check_correctness(mm, instance_graph, 10, 10, 5);

    vector<vector<label>> L;

    int L_size = uncleaned_L.size();
    L.resize(L_size);

    for (int i = 0; i < L_size; i++) {
        L[i].resize(uncleaned_L[i].size());
        int _size = uncleaned_L[i].size();
        for (int j = 0; j < _size; j++) {
            L[i][j].v = uncleaned_L[i][j].hub_vertex;
            L[i][j].h = uncleaned_L[i][j].hop;
            L[i][j].d = uncleaned_L[i][j].distance;
        }
    }

    cuda_vector<label>** Lc = gpu_clean(instance_graph, L, 10, 5);

    vector<vector<hop_constrained_two_hop_label>> L_gpu;
    L_gpu.resize(L_size);

    for (int i = 0; i < L_size; i++) {
        L_gpu[i].resize(Lc[i]->current_size);
        for (int j = 0; j < Lc[i]->current_size; j++) {
            L_gpu[i][j].hub_vertex = (*Lc[i])[j].v;
            L_gpu[i][j].hop = (*Lc[i])[j].h;
            L_gpu[i][j].distance = (*Lc[i])[j].d;
        }
    }
    mm.L = L_gpu;
    hop_constrained_check_correctness(mm, instance_graph, 10, 10, 5);

    Lc[0]->pool->~base_memory();
    cudaFree(Lc[0]->pool);
    // cuda free
    for (int i = 0; i < L_size; i++) {
        Lc[i]->~cuda_vector();
        cudaFree(Lc[i]);
    }
    cudaFree(Lc);

    return 0;
}