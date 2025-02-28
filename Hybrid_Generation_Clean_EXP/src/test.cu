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
#include <HBPLL/gpu_clean.cuh>

#include <vgroup/CDLP_group.cuh>

vector<vector<hop_constrained_two_hop_label> > L_hybrid;
vector<vector<hop_constrained_two_hop_label> > L_hybrid_after_clean;

hop_constrained_case_info info_cpu;
hop_constrained_case_info_v2 *info_gpu;

graph_v_of_v<int> instance_graph;
CSR_graph<weight_type> csr_graph;
Graph_pool<int> graph_pool;

boost::random::mt19937 boost_random_time_seed { static_cast<std::uint32_t>(std::time(0)) }; // Random seed 

struct Executive_Core {
    int id;
    double time_use;
    int core_type; // 0: cpu, 1: gpu
    Executive_Core (int x, double y, int z) : id(x), time_use(y), core_type(z) {}
};
inline bool operator < (Executive_Core a, Executive_Core b) {
    if (a.time_use == b.time_use) return a.id > b.id;
    return a.time_use > b.time_use;
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

void graph_v_of_v_to_LDBC (LDBC<weight_type> &graph, graph_v_of_v<int> &input_graph) {
    int N = input_graph.size();
    int EE = 0;
    for (int i = 0; i < N; i++) {
        int v_adj_size = input_graph[i].size();
        for (int j = 0; j < v_adj_size; j++) {
            int adj_v = input_graph[i][j].first;
            int ec = input_graph[i][j].second;
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

void GPU_HSDL_checker (vector<vector<hub_type_v2> >&LL, graph_v_of_v<int> &instance_graph,
                        int iteration_source_times, int iteration_terminal_times, int hop_bounded, int check_path) {

    boost::random::uniform_int_distribution<> vertex_range{ static_cast<int>(0), static_cast<int>(instance_graph.size() - 1) };
    boost::random::uniform_int_distribution<> hop_range{ static_cast<int>(1), static_cast<int>(hop_bounded) };

    printf("Checker Start.\n");

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
            // hop_constrained_extract_shortest_path;
            if (abs(q_dis - distances[terminal]) > 1e-2 ) {
                cout << "source, terminal, hopcst = " << source << ", "<< terminal << ", " << hop_cst << endl;
                cout << fixed << setprecision(5) << "dis = " << q_dis << endl;
                cout << fixed << setprecision(5) << "distances[terminal] = " << distances[terminal] << endl;
                cout << endl;
                return;
                //exit(0);
            }else if (distances[terminal] != std::numeric_limits<int>::max()) {
                // cout << "correct !!!" << endl;
                // cout << "source, terminal, hopcst = " << source << ", "<< terminal << ", " << hop_cst << endl;
                // cout << fixed << setprecision(5) << "dis = " << q_dis << endl;
                // cout << fixed << setprecision(5) << "distances[terminal] = " << distances[terminal] << endl;
                // cout << endl;
            }
            if (check_path) {
                vector<pair<int, int>> path = hop_constrained_extract_shortest_path(LL, source, terminal, hop_cst);
                int path_dis = 0;
                if (path.size() == 0 && source != terminal) {
                    path_dis = std::numeric_limits<int>::max();
                }
                for (auto xx : path) {
                    path_dis += instance_graph.edge_weight(xx.first, xx.second);
                }
                if (abs(q_dis - path_dis) > 1e-2) {
                    // instance_graph.print();
                    cout << "source = " << source << endl;
                    cout << "terminal = " << terminal << endl;
                    cout << "hop_cst = " << hop_cst << endl;
                    std::cout << "print_vector_pair_int:" << std::endl;
                    for (int i = 0; i < path.size(); i++) {
                        std::cout << "item: |" << path[i].first << "," << path[i].second << "|" << std::endl;
                    }
                    cout << "query_dis = " << q_dis << endl;
                    cout << "path_dis = " << path_dis << endl;
                    cout << "abs(dis - path_dis) > 1e-2!" << endl;
                    getchar();
                }
            }
        }
    }
    printf("Checker End.\n");
    return;
}

int max_N_ID_for_mtx_group_599 = 1e7;
// vector<std::shared_timed_mutex> mtx_group_599(max_N_ID_for_mtx_group_599);
// queue<int> Qid_group_599;
queue<pair<int, int> > que_get_group_bfs[100];

static void get_bfs_group_vertices_thread_function (int group_id, int hop_cst) {

    // vertex, hop
    queue<pair<int, int> > q;
    set<int> s;

    for (int i = 0; i < graph_pool.graph_group[group_id].size(); ++i) {
        q.push(make_pair(graph_pool.graph_group[group_id][i], 0));
        graph_pool.graph_group_bfs[group_id].push_back(graph_pool.graph_group[group_id][i]);
        s.insert(graph_pool.graph_group[group_id][i]);
        // s.insert(graph_pool.graph_group[group_id][i]);
    }
    // printf("shit!!!!\n");

    while (!q.empty()) {
        pair<int, int> x = q.front();
        q.pop();

        // if (s.find(x.first) == s.end()) {
        //     s.insert(x.first);
        //     graph_pool.graph_group_bfs[group_id].push_back(x.first);
        // }

        if (x.second >= hop_cst) continue;

        int v_adj_size = instance_graph[x.first].size();

        for (int i = 0; i < v_adj_size; i++) {
            int adj_v = instance_graph[x.first][i].first;

            if (s.find(adj_v) == s.end()) {
                q.push(make_pair(adj_v, x.second + 1));
                graph_pool.graph_group_bfs[group_id].push_back(adj_v);
                s.insert(adj_v);
            }
        }
    }
}

void get_bfs_group_vertices (int hop_cst) {
    std::vector<std::future<int>> results;
    ThreadPool pool(100);

    for (int group_id = 0; group_id < graph_pool.graph_group.size(); ++ group_id) {
        results.emplace_back(pool.enqueue([group_id, hop_cst] {
            get_bfs_group_vertices_thread_function(group_id, hop_cst);
            return 1;
        }));
    }
    for (auto &&result : results) {
        result.get();
    }
    results.clear();
    
    for (int group_id = 0; group_id < graph_pool.graph_group.size(); ++group_id) {
        printf("graph_pool, graph_pool_bfs: %d, %d\n", graph_pool.graph_group[group_id].size(), graph_pool.graph_group_bfs[group_id].size());
    }
}

int main (int argc, char** argv) {
    srand(time(0));

    size_t free_byte, total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    printf("Device memory start: total %ld, free %ld\n", total_byte, free_byte);

    // Test frequency parameter
    int iteration_graph_times = 1;
    int iteration_source_times = 50000, iteration_terminal_times = 50000;
    
    // graph parameters
    int V = 62586, E = 147892;
    double ec_min = 1, ec_max = 100;

    // CD parameters
    int G_max = 10000;
    // int Distributed_Graph_Num = (V + G_max - 1) / G_max;
    int Distributed_Graph_Num = 10;

    // algo parameters
    // std::string dataset = "/home/mdnd/dataset/data_exp_1w/as-caida20071105/as-caida20071105.e";
    std::string dataset = "/home/mdnd/dataset/data_exp_1w/p2p-Gnutella31/p2p-Gnutella31.e";
    // std::string dataset = "/home/mdnd/dataset/data_exp_1w/twitch/twitch.e";
    // std::string dataset = "/home/mdnd/dataset/data_exp_1w/git_web_ml/git_web_ml.e";
    // std::string dataset = "/home/mdnd/dataset/data_exp_1w/Brightkite_edges/Brightkite_edges.e";
    // std::string dataset = "../data/simple_iterative_tests.txt";
    int hop_cst = 5;
    int algo = 3;
    std::string output = "/home/mdnd/Hybrid_Generation_Clean_EXP/exp_record/test.csv";

    // G_max = 1;
    int CPU_Gen_Num = 0, GPU_Gen_Num = 0;
    int CPU_Clean_Num = 1, GPU_Clean_Num = 4;
    int thread_num = 1000, thread_num_clean = 1000;
    
    // dataset = argv[1];
    // hop_cst = std::stoi(argv[2]);
    // algo = std::stoi(argv[3]);
    // output = argv[4];
    // G_max = std::stoi(argv[5]);
    // thread_num = std::stoi(argv[6]);
    
    double time_generate_labels_total = 0.0;
    double time_clean_labels_total = 0.0;
    
    // test parameters
    int generate_new_graph = 0;
    int print_details = 1;
    int check_correctness = 1;
    int use_cd = 0;
    int use_clean = 1;
    string data_path = dataset;
    //string data_path = "../data/simple_iterative_tests_100w.txt";

    if (generate_new_graph) {
        instance_graph = graph_v_of_v_generate_random_graph<int> (V, E, ec_min, ec_max, 1, boost_random_time_seed);
        instance_graph = graph_v_of_v_update_vertexIDs_by_degrees_large_to_small(instance_graph); // sort vertices
        instance_graph.txt_save("../data/simple_iterative_tests.txt");
    } else {
        V = 0, E = 0;
        instance_graph.txt_read(data_path);
        // instance_graph.txt_read_v2(data_path);
        instance_graph = graph_v_of_v_update_vertexIDs_by_degrees_large_to_small(instance_graph);
        V = instance_graph.size();
        for (int i = 0; i < V; ++i) {
            E += instance_graph[i].size();
        }
    }

    if (use_cd == 0) {
        graph_pool.graph_group.resize(Distributed_Graph_Num);
        int Nodes_Per_Graph = (V - 1) / Distributed_Graph_Num + 1;
        for (int i = 0; i < Distributed_Graph_Num; ++ i) {
            for (int j = Nodes_Per_Graph * i; j < Nodes_Per_Graph * (i + 1); ++j) {
                if (j >= V) break;
                graph_pool.graph_group[i].push_back(j);
            }
        }
        G_max = V / Distributed_Graph_Num + 1;
        G_max = V;
    } else {
        generate_Group_CDLP(instance_graph, graph_pool.graph_group, G_max);
        Distributed_Graph_Num = graph_pool.graph_group.size();
    }

    // cpu info
    info_cpu.upper_k = hop_cst;
	info_cpu.use_rank_prune = 1;
	info_cpu.use_2023WWW_generation = 0;
    info_cpu.use_2023WWW_generation_optimized = 0;
    info_cpu.use_GPU_version_generation = 0;
    info_cpu.use_GPU_version_generation_optimized = 0;
	info_cpu.use_canonical_repair = 0;
	info_cpu.max_run_time_seconds = 100;
    info_cpu.thread_num = 50; // 要和hop_constrained_two_hop_labels_generation.h里面的 #define num_of_threads_cpu 100 保持一致
    printf("Init CPU_Info Successful!\n");

    // gpu info
    info_gpu = new hop_constrained_case_info_v2();
    info_gpu->hop_cst = hop_cst;
    info_gpu->thread_num = thread_num;
    info_gpu->use_2023WWW_GPU_version = 0;
    info_gpu->use_new_algo = 0;
    printf("Init GPU_Info Successful!\n");
    
    // set algo type
    printf("algo: %d\n", algo);
    if (algo == 1) {info_cpu.use_2023WWW_generation = 1, CPU_Gen_Num = 1, GPU_Gen_Num = 0;}
    else if (algo == 2) {info_cpu.use_2023WWW_generation_optimized = 1, CPU_Gen_Num = 1, GPU_Gen_Num = 0;}
    else if (algo == 3) {info_gpu->use_new_algo = 1, CPU_Gen_Num = 0, GPU_Gen_Num = 1;}
    else if (algo == 4) {info_gpu->use_new_algo = 1, CPU_Gen_Num = 0, GPU_Gen_Num = 4;}
    else if (algo == 5) {
        info_cpu.use_2023WWW_generation_optimized = 1, info_gpu->use_new_algo = 1;
        CPU_Gen_Num = 1, GPU_Gen_Num = 4;
    } else if (algo == 6) {
        info_gpu->use_2023WWW_GPU_version = 1, CPU_Gen_Num = 0, GPU_Gen_Num = 1;
        // if (hop_cst == 4) return 0;
    }

    if (GPU_Gen_Num) {
        info_gpu->init(V, hop_cst, G_max, thread_num, graph_pool.graph_group);
    }

    // init label
    L_hybrid.resize(V);
    for (int i = 0; i < V; ++i) L_hybrid.clear();

    // Generate CSR_graph from instance_graph
    LDBC<weight_type> graph(V);
    graph_v_of_v_to_LDBC(graph, instance_graph);
    csr_graph = toCSR(graph);
    printf("Generation Graph Successful!\n");

    // init cpu_generation
    if (CPU_Gen_Num) {
        hop_constrained_two_hop_labels_generation_init(instance_graph, info_cpu);
    }

    // get graph_pool, use_cd 0/1
    if (GPU_Gen_Num) {
        info_gpu->set_nid(Distributed_Graph_Num, graph_pool.graph_group);
    }
    
    printf("G_max: %d\n",G_max);

    cudaMemGetInfo(&free_byte, &total_byte);
    printf("Device memory initial success!: total %ld, free %ld\n", total_byte, free_byte);

    priority_queue<Executive_Core> pq_gen;
    for (int i = 0; i < CPU_Gen_Num; ++i) pq_gen.push(Executive_Core(i, 0, 0)); // id, time, cpu/gpu
    for (int i = 0; i < GPU_Gen_Num; ++i) pq_gen.push(Executive_Core(CPU_Gen_Num + i, 0, 1)); // id, time, cpu/gpu
    for (int i = 0; i < Distributed_Graph_Num; ++i) {
        Executive_Core x = pq_gen.top();
        pq_gen.pop();

        auto begin = std::chrono::high_resolution_clock::now();
        if (x.core_type == 0) { // core type is cpu
            hop_constrained_two_hop_labels_generation(instance_graph, info_cpu, L_hybrid, graph_pool.graph_group[i]);
        } else {
            label_gen(csr_graph, info_gpu, L_hybrid, graph_pool.graph_group[i], i);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
        
        x.time_use += duration;
        pq_gen.push(x);
    }
    while (!pq_gen.empty()) {
        Executive_Core x = pq_gen.top();
        pq_gen.pop();
        time_generate_labels_total = max(time_generate_labels_total, x.time_use);
        printf("Time_Generate_Labels_Total: %.6lf\n", time_generate_labels_total);
    }
    
    // Clear video memory
    cudaMemGetInfo(&free_byte, &total_byte);
    printf("Device memory before: total %ld, free %ld\n", total_byte, free_byte);
    if (GPU_Gen_Num) {
        info_gpu->destroy_L_cuda(G_max);
    }
    csr_graph.destroy_csr_graph();
    // free(info_gpu);
    cudaMemGetInfo(&free_byte, &total_byte);
    printf("Device memory after: total %ld, free %ld\n", total_byte, free_byte);

    // sort the label.
    if (use_clean || check_correctness) {
        for (int v_k = 0; v_k < V; ++ v_k) {
            sort(L_hybrid[v_k].begin(), L_hybrid[v_k].end(), compare_hop_constrained_two_hop_label);
        }
    }

    // numbering for labels
    long long total_label_num = 0;
    vector<long long> label_num;
    vector<int> label_id;
    label_num.resize(V + 1);
    vector<vector<long long>> label_id_2to1;
    label_id_2to1.resize(V);
    vector<pair<int,int>> label_id_1to2;
    for (int i = 0; i < V; i ++) {
        for (int j = 0; j < L_hybrid[i].size(); j ++) {
            label_num[i] ++;
            label_id.push_back(i);
            label_id_1to2.push_back(make_pair(i, j));
            label_id_2to1[i].push_back(total_label_num);
            total_label_num ++;
        }
        label_num[i + 1] = label_num[i];
    }
    
    vector<vector<hop_constrained_two_hop_label> > L_vertex_hub_vertex;
    L_vertex_hub_vertex.resize(V);
    for (int i = 0; i < V; i ++) {
        for (int j = 0; j < L_hybrid[i].size(); j ++) {
            hop_constrained_two_hop_label xxx_tenaction = L_hybrid[i][j];
            xxx_tenaction.hub_vertex = i;
            L_vertex_hub_vertex[L_hybrid[i][j].hub_vertex].push_back(xxx_tenaction);
        }
    }

    int L_use_num = 5;
    vector<vector<vector<hop_constrained_two_hop_label> > > L_use;
    L_use.resize(L_use_num);

    for (int i = 0; i < L_use_num; i ++) L_use[i].resize(V);
    for (int i = 0; i < V; i ++) {
        int rnd = rand() % L_use_num;
        for (int j = 0; j < L_vertex_hub_vertex[i].size(); j ++) {
            if (i % 2 == 1){
                L_use[0][i].push_back(L_vertex_hub_vertex[i][j]);
            } else {
                L_use[1][i].push_back(L_vertex_hub_vertex[i][j]);
            }
        }
    }

    int L_clean_num = 5;
    vector<vector<vector<hop_constrained_two_hop_label> > > L_clean;
    vector<vector<vector<long long> > > L_clean_id;
    L_clean.resize(L_clean_num);
    L_clean_id.resize(L_clean_num);
    for (int i = 0; i < L_clean_num; i ++) L_clean[i].resize(V);
    for (int i = 0; i < L_clean_num; i ++) L_clean_id[i].resize(V);
    for (int i = 0; i < V; i ++) {
        for (int j = 0; j < L_hybrid[i].size(); j ++) {
            int rnd = rand() % L_clean_num;
            L_clean[rnd][i].push_back(L_hybrid[i][j]);
            L_clean_id[rnd][i].push_back(label_id_2to1[i][j]);
        }
    }

    vector<int> check_tot(total_label_num, 1);

    // for (int i = 0; i < L_use_num; i ++) {
    //     for (int j = 0; j < L_clean_num; j ++) {
    //         gpu_clean_init_v3 (instance_graph, L_use[i], L_clean[j], L_clean_id[j], info_gpu, graph_pool, thread_num, hop_cst);
    //         gpu_clean_v3 (instance_graph, info_gpu, L_clean[j], check_tot, thread_num_clean);
    //     }
    // }
    
    // gpu_clean_init_v3 (instance_graph, L_hybrid, L_hybrid, label_id_2to1, info_gpu, graph_pool, thread_num, hop_cst);
    // auto begin = std::chrono::high_resolution_clock::now();
    // gpu_clean_v3 (instance_graph, info_gpu, L_hybrid, check_tot, thread_num_clean); 

    // L_hybrid_after_clean.resize(V);
    // for (long long i = 0; i < total_label_num; i ++) {
    //     if (check_tot[i]) {
    //         L_hybrid_after_clean[label_id_1to2[i].first].push_back(L_hybrid[label_id_1to2[i].first][label_id_1to2[i].second]);
    //     }
    // }
    // std::swap(L_hybrid, L_hybrid_after_clean);

    // auto end = std::chrono::high_resolution_clock::now();
    // auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
    // printf("!!!! Total Time Clean_v2: %.6lf\n", duration);

    long long clean_use_label_num = 50000000;
    long long clean_cleaned_label_num = 50000000;

    if (use_clean) {
        long long label_size_total = 0;
        for (int i = 0; i < V; ++i) {
            label_size_total += L_hybrid[i].size();
        }

        // Hybrid Clean
        priority_queue<Executive_Core> pq_clean;
        for (int i = 0; i < CPU_Clean_Num; ++i) pq_clean.push(Executive_Core(i, 0, 0)); // id, time, cpu/gpu
        for (int i = 0; i < GPU_Clean_Num; ++i) pq_clean.push(Executive_Core(CPU_Clean_Num + i, 0, 1)); // id, time, cpu/gpu
        if (GPU_Clean_Num) {
            gpu_clean_init(instance_graph, L_hybrid, info_gpu, graph_pool, thread_num, hop_cst);
        }
        cudaMemGetInfo(&free_byte, &total_byte);
        printf("Device memory after clean: total %ld, free %ld\n", total_byte, free_byte);
        
        // 4_GPU
        int cnt_label_use = 0, last_label_use = 0;
        
        for (int i = 0; i <= V; i ++) {
            if (i == V || cnt_label_use + L_vertex_hub_vertex[i].size() > clean_use_label_num) {
                for (int j = last_label_use; j < i; j ++) {
                    for (int k = 0; k < L_vertex_hub_vertex[j].size(); k ++) {
                        L_use[0][j].push_back(L_vertex_hub_vertex[j][k]);
                    }
                }

                int cnt_label_cleaned = 0, last_label_cleaned = 0;
                for (int j = 0; j <= V; j++) {
                    if (j == V || cnt_label_cleaned + L_hybrid[j].size() > clean_cleaned_label_num) {
                        for (int k = last_label_cleaned; k < j; k ++) {
                            for (int l = 0; l < L_hybrid[k].size(); l ++) {
                                L_clean[0][k].push_back(L_hybrid[k][l]);
                                L_clean_id[0][k].push_back(label_id_2to1[k][l]);
                            }
                        }
                        printf("shit !!!!!!! : %d, %d\n", i, j);
                        Executive_Core x = pq_clean.top();
                        pq_clean.pop();
                        gpu_clean_init_v3 (instance_graph, L_use[0], L_clean[0], L_clean_id[0], info_gpu, graph_pool, thread_num, hop_cst);
                        auto begin = std::chrono::high_resolution_clock::now();
                        gpu_clean_v3 (instance_graph, info_gpu, L_clean[0], check_tot, thread_num_clean);
                        auto end = std::chrono::high_resolution_clock::now();
                        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
                        
                        for (int k = last_label_cleaned; k < j; k ++) {
                            L_clean[0][k].clear();
                            L_clean_id[0][k].clear();
                        }
                        last_label_cleaned = j;
                        cnt_label_cleaned = 0;
                    }else{
                        cnt_label_cleaned += L_hybrid[j].size();
                    }
                }
                
                for (int j = last_label_use; j < i; j ++) {
                    L_use[0][j].clear();
                }
                last_label_use = i;
                cnt_label_use = 0;
            } else {
                cnt_label_use += L_vertex_hub_vertex[i].size();
            }
        }
        // for (int i = 0; i < Distributed_Graph_Num; ++i) {
        //     // printf("shi!!!!!!!!\n");
        //     Executive_Core x = pq_clean.top();
        //     pq_clean.pop();

        //     auto begin = std::chrono::high_resolution_clock::now();
        //     if (x.core_type == 0) { // core type is cpu
        //         printf("shit11111111111111\n");
        //         hop_constrained_clean_L_distributed (info_cpu, L_hybrid, graph_pool.graph_group[i], info_cpu.thread_num);
        //         printf("shit2222222222222222\n");
        //     } else {
        //         gpu_clean(instance_graph, info_gpu, L_hybrid, thread_num_clean, i);
        //     }
        //     auto end = std::chrono::high_resolution_clock::now();
        //     auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
            
        //     x.time_use += duration;
        //     pq_clean.push(x);
        // }
        
        while (!pq_clean.empty()) {
            Executive_Core x = pq_clean.top();
            pq_clean.pop();
            time_clean_labels_total = max(time_clean_labels_total, x.time_use);
            printf("Time_Clean_Labels_Total: %.6lf\n", time_clean_labels_total);
        }
    }

    // printf("shit11111111111111\n");
    // hop_constrained_clean_L(info_cpu, L_hybrid, info_cpu.thread_num, V);

    L_hybrid_after_clean.resize(V);
    for (long long i = 0; i < total_label_num; i ++) {
        if (check_tot[i]) {
            L_hybrid_after_clean[label_id_1to2[i].first].push_back(L_hybrid[label_id_1to2[i].first][label_id_1to2[i].second]);
        }
    }
    std::swap(L_hybrid, L_hybrid_after_clean);

    long long label_size_total_1 = 0;
    for (int i = 0; i < V; ++i) {
        label_size_total_1 += L_hybrid[i].size();
    }

    if (use_clean || check_correctness) {
        for (int v_k = 0; v_k < V; ++ v_k) {
            sort(L_hybrid[v_k].begin(), L_hybrid[v_k].end(), compare_hop_constrained_two_hop_label);
        }
    }

    if (check_correctness) {
        printf("Check Union !\n");
        GPU_HSDL_checker(L_hybrid, instance_graph, iteration_source_times, iteration_terminal_times, hop_cst, 0);
    }

    long long label_size_total = 0;
    for (int i = 0; i < V; ++i) {
        label_size_total += L_hybrid[i].size();
    }

    // 输出详细记录
    if (print_details) {
        // printf("Total Lable Size Before Clean: %.6lf\n", (double)label_size_total_before_clean / V);
        printf("Total Lable Size: %.6lf\n", (double)label_size_total_1 / V);
        printf("Total Lable Size: %.6lf\n", (double)label_size_total / V);
        printf("CPU Time Generation: %.6lf\n", info_cpu.time_generate_labels);
        printf("CPU Time Tranverse: %.6lf\n", info_cpu.time_traverse);
        printf("CPU Time Init: %.6lf\n", info_cpu.time_initialization);
        printf("CPU Time Clear: %.6lf\n", info_cpu.time_clear);
        printf("GPU Time Generation: %.6lf\n", info_gpu->time_generate_labels);
        printf("GPU Time Tranverse: %.6lf\n", info_gpu->time_traverse_labels);
        printf("Total Time Generation: %.6lf\n", time_generate_labels_total);
        printf("Total Time Clean: %.6lf\n", time_clean_labels_total);
        printf("\n");
    }

    std::ofstream out(output, std::ios::app); // 以追加模式打开文件
    std::string algoname;
    
    if (algo == 1) {algoname = "use_2023WWW_generation";}
    else if (algo == 2) {algoname = "use_2023WWW_generation_optimized";}
    else if (algo == 3) {algoname = "use_new_algo";}
    else if (algo == 4) {algoname = "use_Hybrid_4GPU";}
    else if (algo == 5) {algoname = "use_Hybrid_1CPU_4GPU";}
    else if (algo == 6) {algoname = "use_2023WWW_GPU_version";}
    
    out << algoname << "," << dataset << "," << hop_cst << "," << time_generate_labels_total << "," 
    << label_size_total <<std::endl;
    out.close();

    return 0;
}
/*

Total Lable Size: 928.125500
CPU Time Generation: 0.583115
CPU Time Tranverse: 0.028347
CPU Time Init: 0.000000
CPU Time Clear: 0.000000
GPU Time Generation: 0.401870
GPU Time Tranverse: 0.338247
Total Time Generation: 0.611483
Total Time Clean: 0.250251

Total Lable Size: 928.898800
CPU Time Generation: 2.103966
CPU Time Tranverse: 0.085331
CPU Time Init: 0.000000
CPU Time Clear: 0.000000
GPU Time Generation: 0.000000
GPU Time Tranverse: 0.000000
Total Time Generation: 2.189312
Total Time Clean: 0.613922

*/