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

#include <vgroup/CDLP_group.cuh>

vector<vector<hub_type_v2> > L;
vector<vector<hub_type> > L_gpu, L_cpu;

hop_constrained_case_info info_cpu;
hop_constrained_case_info_v2 *info_gpu;

graph_v_of_v<int> instance_graph;
CSR_graph<weight_type> csr_graph;
Graph_pool<int> graph_pool;

boost::random::mt19937 boost_random_time_seed { static_cast<std::uint32_t>(std::time(0)) }; // 随机种子 
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
    double time_generation;
    int core_type; // 0: cpu, 1: gpu
    Executive_Core (int x, double y, int z) : id(x), time_generation(y), core_type(z) {}
};
inline bool operator < (Executive_Core a, Executive_Core b) {
    if (a.time_generation == b.time_generation) return a.id > b.id;
    return a.time_generation > b.time_generation;
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
                    exit(0);
                }else if (distances[terminal] != std::numeric_limits<int>::max()) {
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

// void query_vertex_pair(std::string query_path, vector<vector<hub_type_v2> >&LL, graph_v_of_v<int> &instance_graph, int upper_k,Res& result,int before_clean) {
//   std::ifstream in(query_path);
//   if (!in) {
// 	std::cout << "Cannot open input file.\n";
// 	return;
//   }
//   std::string line;
//   int source=0, terminal=0;
//   long long time = 0;
//   std::getline(in, line); // skip the first line
//   int lines = 0;
//   while (std::getline(in, line)) {
// 	std::istringstream iss(line);
// 	if (!(iss >> source >> terminal)) {
//     continue;
// }

//     printf("source %d,terminal: %d\n\n");
//     lines++;
// 	std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
// 	 for (int i = 0; i < 10; ++i) {
//             hop_constrained_extract_distance(LL, source, terminal, upper_k);
//      }
// 	std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
//     time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count(); // s
//   //printf("time: %lf\n",time);

//   }
//   if(lines!=100000)
//   {
//     printf("query error\n");
//   }
//   if(before_clean) 
//   	result.before_clean_query_time = time/1e7;
//   else{
// 	result.query_time = time/1e7;
//   }
// }

void query_vertex_pair(std::string query_path, vector<vector<hub_type_v2> >&LL, graph_v_of_v<int> &instance_graph, int upper_k, Res& result, int before_clean) {
    std::ifstream in(query_path);
    if (!in) {
        std::cout << "Cannot open input file.\n";
        return;
    }
    std::string header;
    std::getline(in, header); // 跳过标题行
    int source = 0, terminal = 0;
    long long time = 0;
    int lines = 0;
    while (in >> source >> terminal) {
        //printf("source %d, terminal: %d\n\n", source, terminal);
        lines++;
        std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
        for (int i = 0; i < 10; ++i) {
            hop_constrained_extract_distance(LL, source, terminal, upper_k);
        }
        std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
        time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    }
    if (lines != 100000) {
        printf("query error\n");
    }
    if (before_clean) {
        result.before_clean_query_time = time / 1e7;
    } else {
        result.query_time = time / 1e7;
    }
}

int main (int argc, char **argv) {
    Res result;

    // 样例图参数
    string data_path = "/home/pengchang/GPU4HSDL_EXP/new-data/git_web_ml/git_web_ml.e";
    int V = 30855, E = 577873;
    // int Distributed_Graph_Num = 30;
    // int G_max = V / Distributed_Graph_Num + 1;
    int G_max = 1000;
    int Distributed_Graph_Num = (V + G_max - 1) / G_max;

    // G_max = 1;
    int CPU_Num = 1, GPU_Num = 4;

    int hop_cst = 4, thread_num = 1000;
    double ec_min = 1, ec_max = 100;
    double time_generate_labels_total = 0.0;

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
    
    // // 测试次数参数
    // int iteration_graph_times = 1;
    int iteration_source_times = 1000, iteration_terminal_times = 1000;

        
    // test parameters
    int generate_new_graph = 0;
    int print_details = 1;
    int check_correctness_gpu = 0;
    int check_correctness_cpu = 0;
    int check_correctness = 0;
    int use_cd = 0;

    // 生成图
    if (generate_new_graph) {
        instance_graph = graph_v_of_v_generate_random_graph<int> (V, E, ec_min, ec_max, 1, boost_random_time_seed);
        instance_graph = graph_v_of_v_update_vertexIDs_by_degrees_large_to_small(instance_graph); // sort vertices
        instance_graph.txt_save("../data/simple_iterative_tests.txt");
    }else{
        instance_graph.txt_read(data_path);
        V = instance_graph.size();
        for (int i = 0; i < V; ++i) {E += instance_graph[i].size();}
        Distributed_Graph_Num = (V + G_max - 1) / G_max;
    }
    
    
    // init label
    L_gpu.resize(V);
    //for (int i = 0; i < V; ++i) L_gpu.clear();
    L_cpu.resize(V);
    //for (int i = 0; i < V; ++i) L_cpu.clear();
    L.resize(V);
    //for (int i = 0; i < V; ++i) L.clear();

        // cpu info
    info_cpu.upper_k = hop_cst;
	info_cpu.use_rank_prune = 1;
	info_cpu.use_2023WWW_generation = 0;
	info_cpu.use_canonical_repair = 1;
	info_cpu.max_run_time_seconds = 3600;
    info_cpu.thread_num = 100;
    printf("init cpu_info successful!\n");

    // gpu info
    info_gpu = new hop_constrained_case_info_v2();
    // printf("G_max init: %d\n", G_max);
    info_gpu->init(V, hop_cst, G_max, thread_num, graph_pool.graph_group);
    info_gpu->hop_cst = hop_cst;
    info_gpu->thread_num = thread_num;
    info_gpu->use_d_optimization = 1;
    printf("init gpu_info successful!\n");

    
    // 通过 instance_graph 生成 CSR_graph
    LDBC<weight_type> graph(V);
    graph_v_of_v_to_LDBC(graph, instance_graph);
    csr_graph = toCSR(graph);
    printf("generation graph successful!\n");

    // init cpu_generation
    hop_constrained_two_hop_labels_generation_init(instance_graph, info_cpu);

    // get graph_pool
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

    cudaMallocManaged(&info_gpu->nid, sizeof(int*) * Distributed_Graph_Num);
    cudaMallocManaged(&info_gpu->nid_size, sizeof(int) * Distributed_Graph_Num);
    for (int j = 0; j < Distributed_Graph_Num; ++ j) {
        cudaMallocManaged(&info_gpu->nid[j], sizeof(int) * graph_pool.graph_group[j].size());
        info_gpu->nid_size[j] = graph_pool.graph_group[j].size();
        for (int k = 0; k < graph_pool.graph_group[j].size(); ++k) {
            info_gpu->nid[j][k] = graph_pool.graph_group[j][k];
        }
    }
    priority_queue<Executive_Core> pq;
    while (!pq.empty()) pq.pop();
    for (int i = 0; i < CPU_Num; ++i) {
        pq.push(Executive_Core(i, 0, 0)); // id, time, cpu/gpu
    }
    for (int i = 0; i < GPU_Num; ++i) {
        pq.push(Executive_Core(CPU_Num + i, 0, 1)); // id, time, cpu/gpu
    }
    for (int i = 0; i < Distributed_Graph_Num; ++i) {
        
        Executive_Core x = pq.top();
        pq.pop();
        auto begin = std::chrono::high_resolution_clock::now();
        //printf("xxxxxxxxxxxxxx: %lf, %d, %d\n", x.time_generation, x.id, x.core_type);
        if (x.core_type == 0) { // core type is cpu
            hop_constrained_two_hop_labels_generation(instance_graph, info_cpu, L, graph_pool.graph_group[i]);
        }else{
            if(label_gen(csr_graph, info_gpu, L, graph_pool.graph_group[i], i)==-1)
            {
                printf("falied\n\n");
                return;
            }
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
        
        x.time_generation += duration;
        pq.push(x);

    }

    for (int v_k = 0; v_k < V; ++ v_k) {
        sort(L[v_k].begin(), L[v_k].end(), compare_hop_constrained_two_hop_label);
    }

    if (check_correctness) {
        printf("check union !\n");
        GPU_HSDL_checker(L, instance_graph, iteration_source_times, iteration_terminal_times, hop_cst);
    }
    query_vertex_pair(query_path, L, instance_graph, upper_k,result,0);
    
    while (!pq.empty()) {
        Executive_Core x = pq.top();
        pq.pop();
        time_generate_labels_total = max(time_generate_labels_total, x.time_generation);
        printf("time_generate_labels_total: %.6lf\n", time_generate_labels_total);
    }

    double label_size_gpu = 0;
    for (int i = 0; i < V; ++i) {
        label_size_gpu += L_gpu[i].size();
    }
    
    double label_size_cpu = info_cpu.label_size * V;
    for (int i = 0; i < V; ++i) {
        label_size_cpu += L_cpu[i].size();
    }

    long long label_size_total = 0;
    for (int i = 0; i < V; ++i) {
        label_size_total += L[i].size();
    }

//    输出详细记录
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
    result.index_time = time_generate_labels_total;
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