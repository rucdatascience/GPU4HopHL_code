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

vector<vector<hub_type> > L;
vector<vector<hub_type> > L_gpu, L_cpu;

hop_constrained_case_info info_cpu;
hop_constrained_case_info_v2 *info_gpu;

graph_v_of_v<int> instance_graph;
CSR_graph<weight_type> csr_graph;
Graph_pool<int> graph_pool;

boost::random::mt19937 boost_random_time_seed{static_cast<std::uint32_t>(std::time(0))}; // 随机种子 

int hop_constrained_extract_distance(vector<vector<hub_type>> &L, int source, int terminal, int hop_cst) {
	/*return std::numeric_limits<int>::max() is not connected*/
	if (hop_cst < 0) {
		return 1e9;
	}
	if (source == terminal) {
		return 0;
	} else if (hop_cst == 0) {
		return 1e9;
	}
	long long int distance = 1e9;
	auto vector1_check_pointer = L[source].begin();
	auto vector2_check_pointer = L[terminal].begin();
	auto pointer_L_s_end = L[source].end(), pointer_L_t_end = L[terminal].end();

	while (vector1_check_pointer != pointer_L_s_end && vector2_check_pointer != pointer_L_t_end) {
		if (vector1_check_pointer->hub_vertex == vector2_check_pointer->hub_vertex) {
			auto vector1_end = vector1_check_pointer;
			while (vector1_check_pointer->hub_vertex == vector1_end->hub_vertex && vector1_end != pointer_L_s_end) {
				vector1_end++;
			}
			auto vector2_end = vector2_check_pointer;
			while (vector2_check_pointer->hub_vertex == vector2_end->hub_vertex && vector2_end != pointer_L_t_end) {
				vector2_end++;
			}

			for (auto vector1_begin = vector1_check_pointer; vector1_begin != vector1_end; vector1_begin++) {
				for (auto vector2_begin = vector2_check_pointer; vector2_begin != vector2_end; vector2_begin++) {
					if (vector1_begin->hop + vector2_begin->hop <= hop_cst) {
						long long int dis = (long long int)vector1_begin->distance + vector2_begin->distance;
						if (distance > dis) {
							distance = dis;
						}
					} else {
						break;
					}
				}
			}
			vector1_check_pointer = vector1_end;
			vector2_check_pointer = vector2_end;
		} else if (vector1_check_pointer->hub_vertex > vector2_check_pointer->hub_vertex) {
			vector2_check_pointer++;
		} else {
			vector1_check_pointer++;
		}
	}
	return distance;
}

bool compare_hop_constrained_two_hop_label_v2 (hub_type &i, hub_type &j) {
	if (i.hub_vertex != j.hub_vertex) {
		return i.hub_vertex < j.hub_vertex;
	} else if (i.hop != j.hop) {
		return i.hop < j.hop;
	} else {
		return i.distance < j.distance;
	}
}

vector<vector<hub_type> > hop_constrained_sort_L (int num_of_threads) {
	/*time complexity: O(V*L*logL), where L is average number of labels per
	 * vertex*/

	int N = L_gpu.size();
	vector<vector<hub_type> > output_L(N);

	/*time complexity: O(V*L*logL), where L is average number of labels per
	 * vertex*/
	ThreadPool pool(num_of_threads);
	std::vector<std::future<int>> results; // return typename: xxx
	for (int v_k = 0; v_k < N; v_k++)
	{
		results.emplace_back(pool.enqueue(
			[&output_L, v_k] { // pass const type value j to thread; [] can be empty
				sort(L_gpu[v_k].begin(), L_gpu[v_k].end(), compare_hop_constrained_two_hop_label_v2);
				vector<hub_type>(L_gpu[v_k]).swap(L_gpu[v_k]); // swap释放vector中多余空间
				output_L[v_k] = L_gpu[v_k];
				vector<hub_type>().swap(L_gpu[v_k]); // clear new labels for RAM efficiency

				return 1; // return to results; the return type must be the same with
						  // results
			}));
	}
	for (auto &&result : results)
		result.get(); // all threads finish here

	return output_L;
}

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

void GPU_HSDL_checker (hop_constrained_case_info_v2 *info,  vector<vector<hub_type> >&LL, graph_v_of_v<int> &instance_graph,
                            int iteration_source_times, int iteration_terminal_times, int hop_bounded) {

    boost::random::uniform_int_distribution<> vertex_range{ static_cast<int>(0), static_cast<int>(instance_graph.size() - 1) };
    boost::random::uniform_int_distribution<> hop_range{ static_cast<int>(1), static_cast<int>(hop_bounded) };

    printf("checker start.\n");

    for (int yy = 0; yy < iteration_source_times; yy++) {
        // printf("checker iteration %d !\n", yy);

        int source = vertex_range(boost_random_time_seed);
        std::vector<weight_type> distances; // record shortest path
        distances.resize(instance_graph.size());

        int hop_cst = hop_range(boost_random_time_seed);

        graph_v_of_v_hop_constrained_shortest_distance(instance_graph, source, hop_cst, distances);

        for (int xx = 0; xx < iteration_terminal_times; xx++) {
                int terminal = vertex_range(boost_random_time_seed);

                weight_type q_dis = 0;
                q_dis = hop_constrained_extract_distance(LL, source, terminal, hop_cst);
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

void cosumer_gpu () {
    int x;
    while ((x = graph_pool.get_next_graph()) != -1) {
        printf("gpu x: %d\n", x);
        label_gen(csr_graph, info_gpu, L_gpu, graph_pool.graph_group[x]);
    }
    printf("generation gpu complete !\n");
}

void cosumer_cpu () {
    int x;
    while ((x = graph_pool.get_next_graph()) != -1) {
        printf("cpu x: %d\n", x);
        hop_constrained_two_hop_labels_generation(instance_graph, info_cpu, L_cpu, graph_pool.graph_group[x]);
    }
    printf("generation cpu complete !\n");
}

int main () {
    
    // 测试次数参数
    int iteration_graph_times = 1;
    int iteration_source_times = 100, iteration_terminal_times = 5000;

    // 样例图参数
    int V = 50000, E = 250000, Distributed_Graph_Num = 100;
    int G_max = V / Distributed_Graph_Num + 1;
    
    int hop_cst = 5, thread_num = 1000;
    double ec_min = 1, ec_max = 10;
    double time_generate_labels_total = 0.0;

    // cpu info
    info_cpu.upper_k = hop_cst;
	info_cpu.use_rank_prune = 1;
	info_cpu.use_2023WWW_generation = 0;
	info_cpu.use_canonical_repair = 1;
	info_cpu.max_run_time_seconds = 100;
    info_cpu.thread_num = 100;
    
    // gpu info
    info_gpu = new hop_constrained_case_info_v2();
    info_gpu->init(V, (long long)V * V * (hop_cst + 1), hop_cst, G_max, thread_num);
    printf("init case_info success\n");
    info_gpu->hop_cst = hop_cst;
    info_gpu->thread_num = thread_num;
    info_gpu->use_d_optimization = 1;

    // 分布式图
    vector<vector<int> > Distributed_Graph;
    Distributed_Graph.resize(Distributed_Graph_Num);
    int Nodes_Per_Graph = (V - 1) / Distributed_Graph_Num + 1;
    for (int i = 0; i < Distributed_Graph_Num; ++i) {
        for (int j = Nodes_Per_Graph * i; j < Nodes_Per_Graph * (i + 1); ++j) {
            if (j >= V) break;
            Distributed_Graph[i].push_back(j);
        }
    }
    graph_pool.graph_group.resize(Distributed_Graph_Num);
    graph_pool.graph_group = Distributed_Graph;

    /* test parameters */
    int generate_new_graph = 0;
    int print_details = 1;
    int check_correctness_gpu = 1;
    int check_correctness_cpu = 1;
    int print_L = 0;
    
    // init label
    L_gpu.resize(V);
    for (int i = 0; i < V; ++i) L_gpu.clear();
    L_cpu.resize(V);
    for (int i = 0; i < V; ++i) L_cpu.clear();
    L.resize(V);
    for (int i = 0; i < V; ++i) L.clear();

    /* iteration */
    for (int i = 0; i < iteration_graph_times; i++) {

        // 生成图
        if (generate_new_graph) {
            instance_graph = graph_v_of_v_generate_random_graph<int>(V, E, ec_min, ec_max, 1, boost_random_time_seed);
            instance_graph = graph_v_of_v_update_vertexIDs_by_degrees_large_to_small(instance_graph); // sort vertices
            instance_graph.txt_save("../data/simple_iterative_tests.txt");
        }else{
            instance_graph.txt_read("../data/simple_iterative_tests.txt");
        }

        // 通过 instance_graph 生成 CSR_graph
        LDBC<weight_type> graph(V);
        graph_v_of_v_to_LDBC(graph, instance_graph);
        csr_graph = toCSR(graph);

        // distributed cpu gpu generation
        // auto begin = std::chrono::high_resolution_clock::now();
        // std::thread thread_cpu (cosumer_cpu);
        // std::thread thread_gpu (cosumer_gpu);
        // thread_gpu.join();
        // thread_cpu.join();
        // auto end = std::chrono::high_resolution_clock::now();
        
        // for (int i = 0; i < V; ++i){
        //     for (int j = 0; j < L_gpu[i].size(); ++j) {
        //         hub_type x = L_gpu[i][j];
        //         L[i].push_back({x.hub_vertex, x.hop, x.distance});
        //     }
        //     for (int j = 0; j < L_cpu[i].size(); ++j) {
        //         hub_type x = L_cpu[i][j];
        //         L[i].push_back({x.hub_vertex, x.hop, x.distance});
        //     }
        // }
        // time_generate_labels_total += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
        // printf("generation complete !\n");
        
        // for (int v_k = 0; v_k < V; ++ v_k) {
        //     sort(L[v_k].begin(), L[v_k].end(), compare_hop_constrained_two_hop_label_v2);
        // }

        // if (check_correctness_cpu) {
        //     // printf("check distributed cpu gpu !\n");
        //     GPU_HSDL_checker(info_gpu, L, instance_graph, iteration_source_times, iteration_terminal_times, hop_cst);
        // }

        // label generation GPU
        // for (int j = 0; j < Distributed_Graph_Num; ++j) {
        //     label_gen(csr_graph, info_gpu, L_gpu, graph_pool.graph_group[j]);
        // }
        // for (int v_k = 0; v_k < V; ++ v_k) {
        //     sort(L_gpu[v_k].begin(), L_gpu[v_k].end(), compare_hop_constrained_two_hop_label_v2);
        // }
        // // 检验 GPU 正确性
        // if (check_correctness_gpu) {
        //     printf("check gpu !\n");
        //     GPU_HSDL_checker(info_gpu, L_gpu, instance_graph, iteration_source_times, iteration_terminal_times, hop_cst);
        // }

        // label generation CPU
        for (int j = 0; j < Distributed_Graph_Num; ++j) {
            hop_constrained_two_hop_labels_generation(instance_graph, info_cpu, L_cpu, graph_pool.graph_group[j]);
        }
        for (int v_k = 0; v_k < V; ++ v_k) {
            sort(L_cpu[v_k].begin(), L_cpu[v_k].end(), compare_hop_constrained_two_hop_label_v2);
        }
        // hop_constrained_two_hop_labels_generation(instance_graph, info_cpu);
        // 检验 CPU 正确性
        if (check_correctness_cpu) {
            printf("check cpu !\n");
            GPU_HSDL_checker(info_gpu, L_cpu, instance_graph, iteration_source_times, iteration_terminal_times, hop_cst);
        }

        // 输出标签
        if (print_L) {

        }

    }
    
    int label_size_gpu = 0;
    for (int i = 0; i < V; ++i) {
        label_size_gpu += L_gpu[i].size();
    }
    
    int label_size_cpu = 0;
    for (int i = 0; i < V; ++i) {
        label_size_cpu += L_cpu[i].size();
    }

    int label_size_total = 0;
    for (int i = 0; i < V; ++i) {
        label_size_total += L[i].size();
    }

    // 输出详细记录
    if (print_details) {
        printf("CPU Lable Size: %.6lf\n", (double)label_size_cpu / V);
        printf("GPU Lable Size: %.6lf\n", (double)label_size_gpu / V);
        printf("Total Lable Size: %.6lf\n", (double)label_size_total / V);
        printf("CPU Time Generation: %.6lf\n", info_cpu.time_generate_labels);
        printf("CPU Time Tranverse: %.6lf\n", info_cpu.time_traverse);
        printf("CPU Time Init: %.6lf\n", info_cpu.time_initialization);
        printf("CPU Time Clear: %.6lf\n", info_cpu.time_clear);
        printf("GPU Time Generation: %.6lf\n", info_gpu->time_generate_labels);
        printf("GPU Time Tranverse: %.6lf\n", info_gpu->time_traverse_labels);
        printf("Total Time Generation: %.6lf\n", time_generate_labels_total);
    }

    return 0;
}