#include <graph_v_of_v/ldbc.hpp>
#include <cuda_runtime.h>
#include <iomanip>
#include "HBPLL/test.h"
#include "HBPLL/gpu_query.cuh"
#include <boost/random.hpp>

int main (int argc, char **argv) {

    int ec_min = 1, ec_max = 10, upper_k = 5;
    int V = 10000, E = 50000, tc = 2000;
    // int query_num = 500000;
    int query_num = 1000;

    std::ios::sync_with_stdio(0);
    std::cin.tie(0);
    std::cout.tie(0);

    boost::random::uniform_int_distribution<> rnd_s {static_cast<int>(V / 3 * 2), static_cast<int>(V - 1)};
    boost::random::uniform_int_distribution<> rnd_t {static_cast<int>(V / 3 * 2), static_cast<int>(V - 1)};
    boost::random::uniform_int_distribution<> rnd_h {static_cast<int>(0), static_cast<int>(upper_k)};

    query_info *que;
    int *ans_cpu, *ans_gpu;
    
    ans_cpu = (int *)malloc(query_num * sizeof(int));
    ans_gpu = (int *)malloc(query_num * sizeof(int));
    que = (query_info *)malloc(query_num * sizeof(query_info));

    for (int i = 0; i < query_num; i++) {
        que[i] = query_info(rnd_s(boost_random_time_seed), rnd_t(boost_random_time_seed), rnd_h(boost_random_time_seed));
        ans_cpu[i] = ans_gpu[i] = INT_MAX;
    }

    graph_v_of_v<int> instance_graph;
    instance_graph = graph_v_of_v_generate_random_graph <int> (V, E, ec_min, ec_max, 1, boost_random_time_seed);

    instance_graph = graph_v_of_v_update_vertexIDs_by_degrees_large_to_small(instance_graph); // sort vertices
    instance_graph.txt_save("../data/simple_iterative_tests.txt");

    hop_constrained_case_info mm;
	mm.upper_k = upper_k;
	mm.use_rank_prune = 1;
	mm.use_2023WWW_generation = 0;
	mm.use_canonical_repair = 1;
	mm.max_run_time_seconds = 100;
	mm.thread_num = 100;

    vector<vector<hop_constrained_two_hop_label>> uncleaned_L;
    printf("Generation Start !\n");
    hop_constrained_two_hop_labels_generation(instance_graph, mm, uncleaned_L);
    printf("Generation End !\n");
    // hop_constrained_check_correctness(mm, instance_graph, 10, 10, 5);
    
    clock_t start = clock();
    for (int i = 0; i < query_num; i++) {
        ans_cpu[i] = hop_constrained_extract_distance(mm.L, que[i].s, que[i].t, que[i].h);
    }
    clock_t end = clock();
    std::cout << fixed << setprecision(10) << "CPU query Time: " << (double)(end - start) / CLOCKS_PER_SEC << "s" << endl;

    vector<vector<label>> L;
    int L_size = mm.L.size();
    L.resize(L_size);
    for (int i = 0; i < L_size; i++) {
        L[i].resize(mm.L[i].size());
        int _size = mm.L[i].size();
        for (int j = 0; j < _size; j++) {
            L[i][j].v = mm.L[i][j].hub_vertex;
            L[i][j].h = mm.L[i][j].hop;
            L[i][j].d = mm.L[i][j].distance;
        }
    }

    double GPU_clean_time = gpu_query(instance_graph, L, query_num, que, ans_gpu, mm.upper_k);
    std::cout << fixed << setprecision(10) << "GPU query Time: " << GPU_clean_time << "s"  << endl;

    int check = 1;
    for (int i = 0; i < query_num; ++i) {
        // std::cout << ans_cpu[i] << ", " << ans_gpu[i] << endl;
        if (ans_cpu[i] != ans_gpu[i]) {
            std::cout << i << ": " << ans_cpu[i] << ", " << ans_gpu[i] << endl;
            check = 0;
            break;
        }
    }
    if (check) {
        printf("check pass !\n");
    }else{
        printf("check not pass !\n");
    }

    return 0;
}