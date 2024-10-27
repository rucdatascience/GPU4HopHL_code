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

hop_constrained_case_info info_cpu;
hop_constrained_case_info_v2 *info_gpu;

graph_v_of_v<int> instance_graph;
CSR_graph<weight_type> csr_graph;
Graph_pool<int> graph_pool;

boost::random::mt19937 boost_random_time_seed { static_cast<std::uint32_t>(std::time(0)) }; // Random seed 
struct Res {
  double index_time =0;
  long long before_clean_size=0; // MB
  long long size=0;// MB
  double query_time=0; // average query time
  double before_clean_query_time = 0;
  double clean_time=0;
};
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
                        int iteration_source_times, int iteration_terminal_times, int hop_bounded) {

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
                    exit(0);
                }else if (distances[terminal] != std::numeric_limits<int>::max()) {
                    // cout << "correct !!!" << endl;
                    // cout << "source, terminal, hopcst = " << source << ", "<< terminal << ", " << hop_cst << endl;
                    // cout << fixed << setprecision(5) << "dis = " << q_dis << endl;
                    // cout << fixed << setprecision(5) << "distances[terminal] = " << distances[terminal] << endl;
                    // cout << endl;
                }
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
    printf("Checker End.\n");
    return;
}

void query_vertex_pair(std::string query_path, vector<vector<hop_constrained_two_hop_label> >&LL, graph_v_of_v<int> &instance_graph, int upper_k, Res& result, int before_clean) {
    
    const int ITERATIONS = 10;  // 进行100次完整的查询操作

    long long total_time = 0;  // 累计所有查询的时间

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        std::ifstream in(query_path);  // 每次循环重新打开文件
        if (!in) {
            std::cerr << "Cannot open input file: " << query_path << "\n";
            return;
        }

        std::string header;
        std::getline(in, header); // 跳过标题行

        int source = 0, terminal = 0;
        long long time = 0;
        int lines = 0;
        long long match_count = 0;//实际计算次数
        volatile long long dis=0;
        // 执行一次完整的文件查询操作
        //auto begin = std::chrono::steady_clock::now();
        while (in >> source >> terminal) {
            lines++;
            
            auto begin = std::chrono::steady_clock::now();
            // 每对 source 和 terminal 执行一次查询
            dis+= hop_constrained_extract_distance(LL, source, terminal, upper_k);
            auto end = std::chrono::steady_clock::now();
            time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
            
            if(lines%10000==0)
            {
                //printf("size1: %d,size2: %d, total time now: %lld,match count: %lld\n\n",LL[source].size(),LL[terminal].size(),time,match_count);
            }
        }
        // auto end = std::chrono::steady_clock::now();
        // time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

        // 验证查询行数是否符合预期
        if (lines != 100000) {
            std::cerr << "Query error: Expected 100000 lines, but got " << lines << "\n";
        }

        total_time += time;  // 将每次的查询时间累加
        
    }

    // 计算平均查询时间
    if (before_clean == 1) {
       
        result.before_clean_query_time = total_time / ITERATIONS / 1e6;  // 总时间除以ITERATIONS，转换为ms
    } else {
        printf("total time %ld\n",total_time);
        result.query_time = total_time / ITERATIONS / 1e6;  // 总时间除以ITERATIONS，转换为ms
    }
}


int main (int argc,char** argv) {

    Res result;

    // 样例图参数
    string data_path = "/home/pengchang/GPU4HSDL_EXP/new-data/git_web_ml/git_web_ml.e";
    int V = 308550, E = 577873;
    // int Distributed_Graph_Num = 30;
    // int G_max = V / Distributed_Graph_Num + 1;
    int G_max = 1000;
    int Distributed_Graph_Num = (V + G_max - 1) / G_max;

    // G_max = 1;
    int CPU_Gen_Num = 1, GPU_Gen_Num = 4;
    int CPU_Clean_Num = 1, GPU_Clean_Num = 4;

    int hop_cst = 4, thread_num = 1000,thread_num_clean = 1000;
    double ec_min = 1, ec_max = 100;

    double time_generate_labels_total = 0.0;
    double time_clean_labels_total = 0.0;

    std::string dataset = argv[1];
    int upper_k = std::stoi(argv[2]);
    int algo = std::stoi(argv[3]);
    std::string query_path = argv[4];
    std::string output = argv[5];
    int is_clean = std::stoi(argv[6]);
    std::string dataset_name = std::filesystem::path(dataset).stem().string();

    // Remove trailing ".e" if present
    if (dataset_name.size() > 2 && dataset_name.substr(dataset_name.size() - 2) == ".e") {
        dataset_name = dataset_name.substr(0, dataset_name.size() - 2);
    }
    hop_cst = upper_k;
    data_path = dataset;

    size_t free_byte, total_byte;
    cudaMemGetInfo(&free_byte, &total_byte);
    printf("Device memory start: total %ld, free %ld\n", total_byte, free_byte);

    // Test frequency parameter
    int iteration_graph_times = 1;
    int iteration_source_times = 1000, iteration_terminal_times = 1000;

    
    
    // test parameters
    int generate_new_graph = 0;
    int print_details = 1;
    int check_correctness = 0;
    int use_cd = 1;
    int use_clean = 1;
    //string data_path = "../data/simple_iterative_tests_100w.txt";

     if (generate_new_graph) {
        instance_graph = graph_v_of_v_generate_random_graph<int> (V, E, ec_min, ec_max, 1, boost_random_time_seed);
        instance_graph = graph_v_of_v_update_vertexIDs_by_degrees_large_to_small(instance_graph); // sort vertices
        instance_graph.txt_save("../data/simple_iterative_tests.txt");
    }else{
        instance_graph.txt_read(data_path);
        instance_graph = graph_v_of_v_update_vertexIDs_by_degrees_large_to_small(instance_graph);
        V = instance_graph.size();
        for (int i = 0; i < V; ++i) {E += instance_graph[i].size();}
        Distributed_Graph_Num = (V + G_max - 1) / G_max;
    }
    
    // cpu info
    info_cpu.upper_k = hop_cst;
	info_cpu.use_rank_prune = 1;
	info_cpu.use_2023WWW_generation = 0;
	info_cpu.use_canonical_repair = 1;
	info_cpu.max_run_time_seconds = 100;
    info_cpu.thread_num = 145; //要和hop_constrained_two_hop_labels_generation.h里面的 #define num_of_threads_cpu 100 保持一致
    printf("Init CPU_Info Successful!\n");

    // gpu info
    info_gpu = new hop_constrained_case_info_v2();
    info_gpu->init(V, hop_cst, G_max, thread_num, graph_pool.graph_group);
    cudaMemGetInfo(&free_byte, &total_byte);
    printf("Device memory after init: total %ld, free %ld\n", total_byte, free_byte);
    info_gpu->hop_cst = hop_cst;
    info_gpu->thread_num = thread_num;
    printf("Init GPU_Info Successful!\n");
    
    // init label
    L_hybrid.resize(V);
    for (int i = 0; i < V; ++i) L_hybrid.clear();



    // Generate CSR_graph from instance_graph
    LDBC<weight_type> graph(V);
    graph_v_of_v_to_LDBC(graph, instance_graph);
    csr_graph = toCSR(graph);
    printf("Generation Graph Successful!\n");

    // init cpu_generation
    hop_constrained_two_hop_labels_generation_init(instance_graph, info_cpu);

    // get graph_pool, use_cd 0/1
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
    } else {
        generate_Group_CDLP(instance_graph, graph_pool.graph_group, G_max);
        Distributed_Graph_Num = graph_pool.graph_group.size();
    }
    info_gpu->set_nid(Distributed_Graph_Num, graph_pool.graph_group);
    
    printf("G_max: %d\n",G_max);


    priority_queue<Executive_Core> pq_gen;
    for (int i = 0; i < CPU_Gen_Num; ++i) pq_gen.push(Executive_Core(i, 0, 0)); // id, time, cpu/gpu
    for (int i = 0; i < GPU_Gen_Num; ++i) pq_gen.push(Executive_Core(CPU_Gen_Num + i, 0, 1)); // id, time, cpu/gpu
    for (int i = 0; i < Distributed_Graph_Num; ++i) {
        Executive_Core x = pq_gen.top();
        pq_gen.pop();
        auto begin = std::chrono::high_resolution_clock::now();
        printf("Gen Core Information: %lf, %d, %d\n", x.time_use, x.id, x.core_type);
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
    info_gpu->destroy_L_cuda();
    csr_graph.destroy_csr_graph();
    // free(info_gpu);
    cudaMemGetInfo(&free_byte, &total_byte);
    printf("Device memory after: total %ld, free %ld\n", total_byte, free_byte);

    // sort the label.
    for (int v_k = 0; v_k < V; ++ v_k) {
        sort(L_hybrid[v_k].begin(), L_hybrid[v_k].end(), compare_hop_constrained_two_hop_label);
    }

    if (use_clean) {
        query_vertex_pair(query_path, L_hybrid, instance_graph, upper_k,result,1);
        long long label_size_total = 0;
        for (int i = 0; i < V; ++i) {
            label_size_total += L_hybrid[i].size();
        }
        result.before_clean_size = label_size_total;

        // Hybrid Clean
        priority_queue<Executive_Core> pq_clean;
        for (int i = 0; i < CPU_Clean_Num; ++i) pq_clean.push(Executive_Core(i, 0, 0)); // id, time, cpu/gpu
        for (int i = 0; i < GPU_Clean_Num; ++i) pq_clean.push(Executive_Core(CPU_Clean_Num + i, 0, 1)); // id, time, cpu/gpu
        if (GPU_Clean_Num) {
            gpu_clean_init(instance_graph, L_hybrid, info_gpu, graph_pool, thread_num, hop_cst);
        }
        cudaMemGetInfo(&free_byte, &total_byte);
        printf("Device memory after clean: total %ld, free %ld\n", total_byte, free_byte);
        for (int i = 0; i < Distributed_Graph_Num; ++i) {
            Executive_Core x = pq_clean.top();
            pq_clean.pop();
            auto begin = std::chrono::high_resolution_clock::now();
            printf("Clean Core Information: %lf, %d, %d\n", x.time_use, x.id, x.core_type);
            if (x.core_type == 0) { // core type is cpu
                hop_constrained_clean_L_distributed(info_cpu, L_hybrid, graph_pool.graph_group[i], info_cpu.thread_num);
                // hop_constrained_clean_L (info_cpu, L_hybrid, info_cpu.thread_num, V);
            } else {
                gpu_clean(instance_graph, info_gpu, L_hybrid, thread_num_clean, i);
            }
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
            x.time_use += duration;
            pq_clean.push(x);
        }
        while (!pq_clean.empty()) {
            Executive_Core x = pq_clean.top();
            pq_clean.pop();
            time_clean_labels_total = max(time_clean_labels_total, x.time_use);
            printf("Time_Clean_Labels_Total: %.6lf\n", time_clean_labels_total);
        }
    }

    if (check_correctness) {
        printf("Check Union !\n");
        GPU_HSDL_checker(L_hybrid, instance_graph, iteration_source_times, iteration_terminal_times, hop_cst);
    }

    long long label_size_total = 0;
    for (int i = 0; i < V; ++i) {
        label_size_total += L_hybrid[i].size();
    }
    query_vertex_pair(query_path, L_hybrid, instance_graph, upper_k,result,0);
    // 输出详细记录
    if (print_details) {
        printf("Total Lable Size: %.6lf\n", (double)label_size_total / V);
        printf("CPU Time Generation: %.6lf\n", info_cpu.time_generate_labels);
        printf("CPU Time Tranverse: %.6lf\n", info_cpu.time_traverse);
        printf("CPU Time Init: %.6lf\n", info_cpu.time_initialization);
        printf("CPU Time Clear: %.6lf\n", info_cpu.time_clear);
        printf("GPU Time Generation: %.6lf\n", info_gpu->time_generate_labels);
        printf("GPU Time Tranverse: %.6lf\n", info_gpu->time_traverse_labels);
        printf("Total Time Generation: %.6lf\n", time_generate_labels_total);
        printf("Total Time Clean: %.6lf\n", time_clean_labels_total);
    }
    result.index_time = time_generate_labels_total;
    result.clean_time = time_clean_labels_total;

    result.size = label_size_total;


    std::ofstream out(output, std::ios::app); // 以追加模式打开文件
    // 追加写入结果到文件
    std::string algoname;
    algoname = (algo==0)?"Hybrid":"GPU";
    out << algoname<< "," << "GPU,"<<dataset_name<<","<<upper_k<<","<<result.index_time << "," 
        << result.size << "," << result.before_clean_size<<","<<result.query_time << "," <<result.before_clean_query_time<<","
        << result.clean_time << std::endl;

    out.close();

    return 0;
}