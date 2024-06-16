#pragma once
#include <boost/heap/fibonacci_heap.hpp>
#include <Hop/graph_v_of_v_idealID_HB_canonical_repair.h>

using namespace std;

struct HBPLL_v1_node {
public:
    int vertex, parent_vertex, hop;
    double priority_value;
};

bool operator<(HBPLL_v1_node const &x, HBPLL_v1_node const &y) {
    return x.priority_value > y.priority_value;
}
typedef typename boost::heap::fibonacci_heap<HBPLL_v1_node>::handle_type graph_v_of_v_idealID_HL_PLL_v1_handle_t_for_sp;

void graph_v_of_v_idealID_HL_HB_v2_thread_function_HBDIJ(int v_k, int N, int upper_k, bool use_rank_pruning) {

    /**
     *
     *  Attention:
     *  Code needs to be completed
     *
     *
     */
}

void graph_v_of_v_idealID_HB_v2_transfer_thread(vector<vector<two_hop_label_v2>> *output_L, int v_k, int value_M) {
    sort(L_temp_599[v_k].begin(), L_temp_599[v_k].end(), compare_two_hop_label_small_to_large);

    int size_vk = L_temp_599[v_k].size();
    if (size_vk == 0)
        return;

    if (value_M != 0) {
        for (int i = 0; i < size_vk; i++) {
            L_temp_599[v_k][i].distance += L_temp_599[v_k][i].hop * value_M;
        }
    }

    vector<tuple<double, int, int>> dist_info;
    dist_info.push_back({L_temp_599[v_k][0].distance, L_temp_599[v_k][0].parent_vertex, L_temp_599[v_k][0].hop});

    vector<two_hop_label_v2> L2_vk;
    for (int i = 1; i < size_vk; i++) {
        if (L_temp_599[v_k][i].vertex == L_temp_599[v_k][i-1].vertex) {
            dist_info.push_back({L_temp_599[v_k][i].distance, L_temp_599[v_k][i].parent_vertex, L_temp_599[v_k][i].hop});
        } else {
            two_hop_label_v2 xx;
            xx.dist_info.swap(dist_info);
            xx.vertex = L_temp_599[v_k][i-1].vertex;
            L2_vk.push_back(xx);
            vector<tuple<double, int, int>>().swap(dist_info);
            dist_info.push_back({L_temp_599[v_k][i].distance, L_temp_599[v_k][i].parent_vertex, L_temp_599[v_k][i].hop});
        }
    }
    two_hop_label_v2 xx;
    xx.dist_info.swap(dist_info);
    xx.vertex = L_temp_599[v_k][size_vk-1].vertex;
    L2_vk.push_back(xx);

    (*output_L)[v_k] = L2_vk;
    vector<two_hop_label_v1>().swap(L_temp_599[v_k]);
}

vector<vector<two_hop_label_v2>> graph_v_of_v_idealID_HB_v2_transfer_labels(int N, int max_N_ID, int num_of_threads, int value_M = 0) {
    vector<vector<two_hop_label_v2>> output_L(max_N_ID);
    vector<vector<two_hop_label_v2>> *p = &output_L;

    ThreadPool pool(num_of_threads);
    std::vector<std::future<int>> results;
    for (int v_k = 0; v_k < N; v_k++) {
        results.emplace_back(pool.enqueue([p, v_k, value_M] {
            graph_v_of_v_idealID_HB_v2_transfer_thread(p, v_k, value_M);
            return 1;
        }));
    }
    for (auto &&result : results)
        result.get();

    return output_L;
}

void graph_v_of_v_idealID_HB_v2(graph_v_of_v_idealID &input_graph, int max_N_ID , int num_of_threads, graph_v_of_v_idealID_two_hop_case_info_v1 &case_info) {
    //----------------------------------- step 1: initialization -----------------------------------
    cout << "step 1: initialization" << endl;

    auto begin = std::chrono::high_resolution_clock::now();
    /*information prepare*/
    labal_size_599 = 0;
    begin_time_599 = std::chrono::high_resolution_clock::now();
    max_run_time_nanoseconds_599 = case_info.max_run_time_seconds * 1e9;
    max_labal_size_599 = case_info.max_labal_size;

    if (max_N_ID > max_N_ID_for_mtx_599) {
        cout << "max_N_ID > max_N_ID_for_mtx_599; max_N_ID_for_mtx_599 is too small!" << endl;
        exit(1);
    }

    L_temp_599.resize(max_N_ID);
    int N = input_graph.size();

    /* thread info */
    ThreadPool pool(num_of_threads);
    std::vector<std::future<int>> results;
    int num_of_threads_per_push = num_of_threads * 100;

    ideal_graph_599 = input_graph;

    auto end = std::chrono::high_resolution_clock::now();
    case_info.time_initialization = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;

    //----------------------------------------------- step 2: generate labels ---------------------------------------------------------------
    cout << "step 2: generate labels" << endl;
    begin = std::chrono::high_resolution_clock::now();

    /*searching shortest paths*/
    int upper_k = case_info.upper_k == 0 ? std::numeric_limits<int>::max() : case_info.upper_k;
    bool use_rank_pruning = case_info.use_rank_pruning;

    int push_num = 0;
    for (int v_k = 0; v_k < N; v_k++) {
        if (ideal_graph_599[v_k].size() > 0) {
            results.emplace_back(
                    pool.enqueue([v_k, N, upper_k, use_rank_pruning] {
                        graph_v_of_v_idealID_HL_HB_v2_thread_function_HBDIJ(v_k, N, upper_k, use_rank_pruning);
                        return 1;
                    }));
            push_num++;
        }
        if (push_num % num_of_threads_per_push == 0) {
            for (auto &&result : results)
                result.get();
            results.clear();
        }
    }

    for (auto &&result : results)
        result.get();

    end = std::chrono::high_resolution_clock::now();
    case_info.time_generate_labels = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;  // s

    //----------------------------------------------- step 3: transfer_labels---------------------------------------------------------------
    cout << "step 3: transfer_labels" << endl;

    L2_temp_599 = graph_v_of_v_idealID_HB_v2_transfer_labels(N, max_N_ID, num_of_threads, case_info.value_M);
    case_info.L2 = L2_temp_599;

    graph_v_of_v_idealID_two_hop_clear_global_values();
}