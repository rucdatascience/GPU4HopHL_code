#include <graph_v_of_v/ldbc.hpp>
#include <cuda_runtime.h>
#include "HBPLL/test.h"
#include "HBPLL/gpu_clean.cuh"

int main(int argc,char **argv) {
    int ec_min = 1, ec_max = 10;
    int V = 10, E = 10;
    int tc = 10;
    std::cout << "Input the number of vertices: " << std::endl;
    std::cin >> V;
    std::cout << "Input the number of edges: " << std::endl;
    std::cin >> E;
    std::cout << "Input the number of threads: " << std::endl;
    std::cin >> tc;

    std::ios::sync_with_stdio(0);
    std::cin.tie(0);
    std::cout.tie(0);

    //freopen("../input.txt", "r", stdin);

    /*std::string directory;
    std::cout << "Please input the data directory: " << std::endl;
    std::cin >> directory;

    if (directory.back() != '/')
        directory += "/";

    std::string graph_name;
    std::cout << "Please input the graph name: " << std::endl;
    std::cin >> graph_name;

    std::string config_file_path = directory + graph_name + ".properties";*/

    graph_v_of_v<int> instance_graph;
    instance_graph = graph_v_of_v_generate_random_graph<int>(V, E, ec_min, ec_max, 1, boost_random_time_seed);
    //instance_graph.txt_read("simple_iterative_tests.txt");
    //LDBC<int> ldbc_graph(directory, graph_name);
    //ldbc_graph.read_config(config_file_path);

    //ldbc_graph.load_graph();

    instance_graph = graph_v_of_v_update_vertexIDs_by_degrees_large_to_small(instance_graph); // sort vertices
    instance_graph.txt_save("simple_iterative_tests.txt");

    hop_constrained_case_info mm;
	mm.upper_k = 5;
	mm.use_rank_prune = 1;
	mm.use_2023WWW_generation = 0;
	mm.use_canonical_repair = 1;
	mm.max_run_time_seconds = 10;
	mm.thread_num = 100;

    //test_HSDL(instance_graph);

    vector<vector<hop_constrained_two_hop_label>> uncleaned_L;

    hop_constrained_two_hop_labels_generation(instance_graph, mm, uncleaned_L);
    hop_constrained_check_correctness(mm, instance_graph, 10, 10, 5);
    std::cout<<"CPU Clean Time: "<<mm.time_canonical_repair<<" s"<<endl;

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
    vector<vector<hop_constrained_two_hop_label>> L_gpu;
    L_gpu.resize(L_size);

    gpu_clean(instance_graph, L, L_gpu,tc, 5);
    std::cout << "GPU clean finished" << std::endl;
    // if (Lc == nullptr) {
    //     std::cout << "GPU clean failed" << std::endl;
    //     return 0;
    // }

    // for (int i = 0; i < L_size; i++) {
    //     L_gpu[i].resize(Lc[i]->current_size);
    //     for (int j = 0; j < Lc[i]->current_size; j++) {
    //         L_gpu[i][j].hub_vertex = (*Lc[i])[j].v;
    //         L_gpu[i][j].hop = (*Lc[i])[j].h;
    //         L_gpu[i][j].distance = (*Lc[i])[j].d;
    //     }
    // }

    if (mm.L.size() != L_gpu.size())
        std::cout << "Total GPU clean size mismatch" << std::endl;
    else {
        int l_size = mm.L.size();
        for (int i = 0; i < l_size; i++) {
            if (mm.L[i].size() != L_gpu[i].size()) {
                std::cout << "GPU clean size mismatch at " << i << std::endl;
                std::cout << "Baseline label size: " << mm.L[i].size() << std::endl;
                for (int j = 0; j < mm.L[i].size(); j++) {
                    std::cout << mm.L[i][j].hub_vertex << " " << mm.L[i][j].hop << " " << mm.L[i][j].distance << std::endl;
                }
                std::cout << "GPU label size: " << L_gpu[i].size() << std::endl;
                for (int j = 0; j < L_gpu[i].size(); j++) {
                    std::cout << L_gpu[i][j].hub_vertex << " " << L_gpu[i][j].hop << " " << L_gpu[i][j].distance << std::endl;
                }
                continue;
            }
            else {
                int _size = mm.L[i].size();
                for (int j = 0; j < _size; j++) {
                    if (mm.L[i][j].hub_vertex != L_gpu[i][j].hub_vertex || mm.L[i][j].hop != L_gpu[i][j].hop || mm.L[i][j].distance != L_gpu[i][j].distance) {
                        std::cout << "GPU clean mismatch at " << i << " " << j << std::endl;
                        break;
                    }
                }
            }
        }
    }

    mm.L = L_gpu;
    hop_constrained_check_correctness(mm, instance_graph, 10, 10, 5);

    //Lc[0]->pool->~base_memory();
    // cudaFree(Lc[0]->pool);
    // // cuda free
    // for (int i = 0; i < L_size; i++) {
    //     Lc[i]->~cuda_vector();
    //     cudaFree(Lc[i]);
    // }
    // cudaFree(Lc);

    return 0;
}