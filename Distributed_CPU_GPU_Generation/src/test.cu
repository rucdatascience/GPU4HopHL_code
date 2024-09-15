#include <bits/stdc++.h>
#include <boost/random.hpp>
#include <boost/signals2/signal.hpp>

#include <label/gen_label.cuh>
#include <label/global_labels_v2.cuh>

#include <memoryManagement/graph_pool.hpp>

#include <graph/ldbc.hpp>
#include <graph/csr_graph.hpp>
#include <graph_v_of_v/graph_v_of_v.h>
#include <graph_v_of_v/graph_v_of_v_shortest_paths.h>
#include <graph_v_of_v/graph_v_of_v_generate_random_graph.h>
#include <graph_v_of_v/graph_v_of_v_hop_constrained_shortest_distance.h>
#include <graph_v_of_v/graph_v_of_v_update_vertexIDs_by_degrees_large_to_small.h>

#include <HBPLL/hop_constrained_two_hop_labels_generation.h>

boost::random::mt19937 boost_random_time_seed{static_cast<std::uint32_t>(std::time(0))}; // ������� 

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

    // label ��û�� sort ��������������ѯ
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
                    cout << "correct !!!" << endl;
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
    
    // ���Դ�������
    int iteration_graph_times = 1;
    int iteration_source_times = 1000, iteration_terminal_times = 1000;

    // ����ͼ����
    int V = 10000, E = 50000, Distributed_Graph_Num = 10;
    
    int hop_cst = 5, thread_num = 1000;
    double ec_min = 1, ec_max = 10;

    // cpu info
    hop_constrained_case_info info_cpu;
    info_cpu.upper_k = hop_cst;
	info_cpu.use_rank_prune = 1;
	info_cpu.use_2023WWW_generation = 0;
	info_cpu.use_canonical_repair = 1;
	info_cpu.max_run_time_seconds = 100;
    info_cpu.thread_num = 100;
    
    // gpu info
    hop_constrained_case_info_v2 *info_gpu = new hop_constrained_case_info_v2();
    info_gpu->init(V, V * V * (hop_cst + 1), hop_cst, thread_num);
    printf("init case_info success\n");
    info_gpu->hop_cst = hop_cst;
    info_gpu->thread_num = thread_num;
    info_gpu->use_d_optimization = 1;

    // �ֲ�ʽͼ
    vector<vector<int> > Distributed_Graph;
    Distributed_Graph.resize(Distributed_Graph_Num);
    int Nodes_Per_Graph = (V - 1) / Distributed_Graph_Num + 1;
    for (int i = 0; i < Distributed_Graph_Num; ++i) {
        for (int j = Nodes_Per_Graph * i; j < Nodes_Per_Graph * (i + 1); ++j) {
            if (j >= V) break;
            Distributed_Graph[i].push_back(j);
        }
    }
    Graph_pool<int> graph_pool(Distributed_Graph_Num);
    graph_pool.graph_group = Distributed_Graph;

    /* test parameters */
    int generate_new_graph = 1;
    int print_details = 1;
    int check_correctness = 1;
    int print_L = 0;
    
    vector<vector<hub_type> > L;
    L.resize(V);
    for (int i = 0; i < V; ++i) L.clear();

    /* iteration */
    for (int i = 0; i < iteration_graph_times; i++) {

        // ����ͼ
        graph_v_of_v<int> instance_graph;
        if (generate_new_graph) {
            instance_graph = graph_v_of_v_generate_random_graph<int>(V, E, ec_min, ec_max, 1, boost_random_time_seed);
            instance_graph = graph_v_of_v_update_vertexIDs_by_degrees_large_to_small(instance_graph); // sort vertices
            instance_graph.txt_save("../data/simple_iterative_tests.txt");
        }else{
            instance_graph.txt_read("../data/simple_iterative_tests.txt");
        }
    
        // ͨ�� instance_graph ���� CSR_graph
        LDBC<weight_type> graph(V);
        graph_v_of_v_to_LDBC(graph, instance_graph);
        CSR_graph<weight_type> csr_graph = toCSR(graph);
        
        // label generation CPU �� GPU ����
        for (int j = 0; j < Distributed_Graph_Num; ++j) {
            // printf
            label_gen(csr_graph, info_gpu, L, graph_pool.graph_group[j]);
        }
        hop_constrained_two_hop_labels_generation(instance_graph, info_cpu);

        // ������ȷ��
        if (check_correctness) {
            GPU_HSDL_checker(info_gpu, L, instance_graph, iteration_source_times, iteration_terminal_times, hop_cst);
        }

        // �����ǩ
        if (print_L) {

        }

    }
    
    int label_size = 0;
    for (int i = 0; i < V; ++i) {
        label_size += L[i].size();
    }
    
    // �����ϸ��¼
    if (print_details) {
        printf("CPU Lable Size: %.6lf\n", info_cpu.label_size);
        printf("GPU Lable Size: %.6lf\n", (double)label_size / V);
        printf("CPU Time Generation: %.6lf\n", info_cpu.time_generate_labels);
        printf("GPU Time Generation: %.6lf\n", info_gpu->time_generate_labels);
    }

    return 0;
}