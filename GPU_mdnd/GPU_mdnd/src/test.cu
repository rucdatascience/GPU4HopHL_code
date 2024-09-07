#include <bits/stdc++.h>
#include <boost/random.hpp>

#include <label/gen_label.cuh>
#include <label/global_labels_v2.cuh>

#include <graph/ldbc.hpp>
#include <graph/csr_graph.hpp>
#include <graph_v_of_v/graph_v_of_v.h>
#include <graph_v_of_v/graph_v_of_v_shortest_paths.h>
#include <graph_v_of_v/graph_v_of_v_generate_random_graph.h>
#include <graph_v_of_v/graph_v_of_v_hop_constrained_shortest_distance.h>
#include <graph_v_of_v/graph_v_of_v_update_vertexIDs_by_degrees_large_to_small.h>

#include <HBPLL/hop_constrained_two_hop_labels_generation.h>

boost::random::mt19937 boost_random_time_seed{static_cast<std::uint32_t>(std::time(0))}; // 随机种子 

void graph_v_of_v_to_LDBC (LDBC<weight_type> &graph, graph_v_of_v<int> &input_graph) {
    int N = input_graph.size();
    for (int i = 0; i < N; i++) {
        int v_adj_size = input_graph[i].size();
        for (int j = 0; j < v_adj_size; j++) {
            int adj_v = input_graph[i][j].first;
            int ec = (int)input_graph[i][j].second;
            graph.add_edge(i, adj_v, ec);
        }
    }
}

void query_mindis_with_hub_host (int V, int x, int y, int hop_cst,
                vector<vector<hub_type> >&L, weight_type *distance) {
    (*distance) = 1e9;

    // label 还没有 sort 过，暂且这样查询
    for (int i = 0; i < L[x].size(); i++){
        for (int j = 0; j < L[y].size(); j++) {
            if (L[x][i].hub_vertex == L[y][j].hub_vertex) {
                if (L[x][i].hop + L[y][j].hop <= hop_cst) {
                    (*distance) = min((*distance), L[x][i].distance + L[y][j].distance);
                }
            }
        }
    }
}

void GPU_HSDL_checker (hop_constrained_case_info_v2 *info,  vector<vector<hub_type> >&L, graph_v_of_v<int> &instance_graph,
                            int iteration_source_times, int iteration_terminal_times, int hop_bounded) {

    boost::random::uniform_int_distribution<> vertex_range{ static_cast<int>(0), static_cast<int>(instance_graph.size() - 1) };
    boost::random::uniform_int_distribution<> hop_range{ static_cast<int>(1), static_cast<int>(hop_bounded) };

    printf("checker start.\n");

    for (int yy = 0; yy < iteration_source_times; yy++) {
        int source = vertex_range(boost_random_time_seed);
        std::vector<weight_type> distances; // record shortest path
        distances.resize(instance_graph.size());

        int hop_cst = hop_range(boost_random_time_seed);

        graph_v_of_v_hop_constrained_shortest_distance(instance_graph, source, hop_cst, distances);

        for (int xx = 0; xx < iteration_terminal_times; xx++) {
                int terminal = vertex_range(boost_random_time_seed);

                weight_type q_dis = 0;
                query_mindis_with_hub_host(instance_graph.size(), source, terminal, hop_cst, L, &q_dis);
                if (abs(q_dis - distances[terminal]) > 1e-2 ) {
                    cout << "source, terminal, hopcst = " << source << ", "<< terminal << ", " << hop_cst << endl;
                    cout << fixed << setprecision(5) << "dis = " << q_dis << endl;
                    cout << fixed << setprecision(5) << "distances[terminal] = " << distances[terminal] << endl;
                    cout << endl;
                    return;
                }else{
                    // cout << "correct !!!" << endl;
                    // cout << "source, terminal, hopcst = " << source << ", "<< terminal << ", " << hop_cst << endl;
                    // cout << fixed << setprecision(5) << "dis = " << q_dis << endl;
                    // cout << fixed << setprecision(5) << "distances[terminal] = " << distances[terminal] << endl;
                    // cout << endl;
                }
        }
    }
    printf("checker end.\n");
    return;
}

int main () {
    
    // 测试次数参数
    int iteration_graph_times = 1;
    int iteration_source_times = 1000, iteration_terminal_times = 1000;

    // 样例图参数
    int V = 10000, E = 50000;
    // scanf("%d %d", &V, &E);
    
    int upper_k = 5;
    double ec_min = 1, ec_max = 10;

    hop_constrained_case_info info_cpu;
    info_cpu.upper_k = upper_k;
	info_cpu.use_rank_prune = 1;
	info_cpu.use_2023WWW_generation = 0;
	info_cpu.use_canonical_repair = 1;
	info_cpu.max_run_time_seconds = 10;
    info_cpu.thread_num = 100;
    
    hop_constrained_case_info_v2 *info_gpu = new hop_constrained_case_info_v2();
    info_gpu->use_d_optimization = 1;

    // printf("yes1\n");

    /* test parameters */
    int generate_new_graph = 1;
    int print_details = 1;
    int check_correctness = 0;
    int print_L = 0;
    
    vector<vector<hub_type> > L;
    L.resize(V);

    // printf("yes2\n");

    /* iteration */
    for (int i = 0; i < iteration_graph_times; i++) {

        // printf("yes3\n");

        // 生成图
        graph_v_of_v<int> instance_graph;
        if (generate_new_graph) {
            // printf("yes3\n");
            instance_graph = graph_v_of_v_generate_random_graph<int>(V, E, ec_min, ec_max, 1, boost_random_time_seed);
            // printf("yes3\n");
            instance_graph = graph_v_of_v_update_vertexIDs_by_degrees_large_to_small(instance_graph); // sort vertices
            // printf("yes3\n");
            instance_graph.txt_save("../data/simple_iterative_tests.txt");
        }else{
            instance_graph.txt_read("../data/simple_iterative_tests.txt");
        }

        // printf("yes4\n");
    
        // 通过 instance_graph 生成 CSR_graph
        LDBC<weight_type> graph(V);
        graph_v_of_v_to_LDBC(graph, instance_graph);
        CSR_graph<weight_type> csr_graph = toCSR(graph);
        
        // label generation CPU 和 GPU 区别
        label_gen(csr_graph, info_gpu, upper_k, L);
        hop_constrained_two_hop_labels_generation(instance_graph, info_cpu);

        // 检验正确性
        if (check_correctness) {
            GPU_HSDL_checker(info_gpu, L, instance_graph, iteration_source_times, iteration_terminal_times, upper_k);
        }

        // 输出标签
        if (print_L) {

        }

    }
    
    // info_cpu.print_L();
    // 比较cpu和gpu的label看看哪里有问题
    for (int i = 0; i < V; ++i) {
        // printf("%d: %d %d", i, L[i].size(), info_cpu.L[i].size());
        // if (L[i].size() != info_cpu.L[i].size()){
        //     printf(" dd");
        // }
        // puts("");
        for (int j = 0; j < info_cpu.L[i].size(); ++j) {
            int tag = 0;
            for (int k = 0; k < L[i].size(); ++k) {
                // printf("%d %d %d\n", info_cpu.L[i][j].hub_vertex, info_cpu.L[i][j].hop, info_cpu.L[i][j].distance);
                if (info_cpu.L[i][j].hub_vertex == L[i][k].hub_vertex && 
                    info_cpu.L[i][j].hop == L[i][k].hop && 
                    info_cpu.L[i][j].distance == L[i][k].distance) {
                        tag = 1;
                        break;
                }
            }
            if (tag == 0) {
                // printf("%d %d %d\n", info_cpu.L[i][j].hub_vertex, info_cpu.L[i][j].hop, info_cpu.L[i][j].distance);
                // for (int k = 0; k < L[i].size(); ++k) {
                //     if (info_cpu.L[i][j].hub_vertex == L[i][k].hub_vertex && 
                //         info_cpu.L[i][j].hop == L[i][k].hop) {
                //             printf("%d %d %d %d %d\n", i, info_cpu.L[i][j].hub_vertex, 
                //             info_cpu.L[i][j].hop, info_cpu.L[i][j].distance
                //             , L[i][k].distance);
                //             break;
                //     }
                // }
            }
        }
    }

    // 输出详细记录
    if (print_details) {
        printf("CPU Lable Size: %.5lf\n", info_cpu.label_size);
        printf("GPU Lable Size: %.5lf\n", info_gpu->label_size);
        printf("CPU Time Generation: %.5lf\n", info_cpu.time_generate_labels + info_cpu.time_canonical_repair);
        printf("GPU Time Generation: %.5lf\n", info_gpu->time_generate_labels);
    }

    return 0;
}