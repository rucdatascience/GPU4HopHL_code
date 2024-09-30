#include <graph_v_of_v/ldbc.hpp>

#include <memoryManagement/graph_pool.hpp>
#include <cuda_runtime.h>

#include "HBPLL/test.h"
#include "HBPLL/gpu_clean.cuh"

#include <vgroup/CDLP_group.cuh>

void save_labels(const std::vector<std::vector<hop_constrained_two_hop_label>>& labels, const std::string& filename) {
    std::ofstream ofs(filename, std::ios::binary);

    // 写入外部向量的大小
    size_t outer_size = labels.size();
    ofs.write(reinterpret_cast<const char*>(&outer_size), sizeof(size_t));

    for (const auto& inner_vector : labels) {
        // 写入内部向量的大小
        size_t inner_size = inner_vector.size();
        ofs.write(reinterpret_cast<const char*>(&inner_size), sizeof(size_t));

        // 写入每个 hop_constrained_two_hop_label 对象的数据
        for (const auto& label : inner_vector) {
            ofs.write(reinterpret_cast<const char*>(&label.hub_vertex), sizeof(int));
            ofs.write(reinterpret_cast<const char*>(&label.parent_vertex), sizeof(int));
            ofs.write(reinterpret_cast<const char*>(&label.hop), sizeof(int));
            ofs.write(reinterpret_cast<const char*>(&label.distance), sizeof(int));
        }
    }

    ofs.close();
}

void load_labels(std::vector<std::vector<hop_constrained_two_hop_label>>& labels, const std::string& filename) {
    std::ifstream ifs(filename, std::ios::binary);

    // 读取外部向量的大小
    size_t outer_size;
    ifs.read(reinterpret_cast<char*>(&outer_size), sizeof(size_t));
    labels.resize(outer_size);

    for (size_t i = 0; i < outer_size; ++i) {
        // 读取内部向量的大小
        size_t inner_size;
        ifs.read(reinterpret_cast<char*>(&inner_size), sizeof(size_t));
        labels[i].resize(inner_size);

        // 读取每个 hop_constrained_two_hop_label 对象的数据
        for (size_t j = 0; j < inner_size; ++j) {
            hop_constrained_two_hop_label label;
            ifs.read(reinterpret_cast<char*>(&label.hub_vertex), sizeof(int));
            ifs.read(reinterpret_cast<char*>(&label.parent_vertex), sizeof(int));
            ifs.read(reinterpret_cast<char*>(&label.hop), sizeof(int));
            ifs.read(reinterpret_cast<char*>(&label.distance), sizeof(int));
            labels[i][j] = label;
        }
    }

    ifs.close();
}

struct Executive_Core {
    int id;
    double time_clear;
    int core_type; // 0: cpu, 1: gpu
    Executive_Core (int x, double y, int z) : id(x), time_clear(y), core_type(z) {}
};
inline bool operator < (Executive_Core a, Executive_Core b) {
    if (a.time_clear == b.time_clear) return a.id > b.id;
    return a.time_clear > b.time_clear;
}

class record_info {
public:
    double CPU_Clean_Time, GPU_Clean_Time, Hybrid_Clean_Time;
};

vector<vector<hop_constrained_two_hop_label>> uncleaned_L;

record_info main_element () {

    record_info record_info_case;
    Graph_pool<int> graph_pool;
    graph_v_of_v<int> instance_graph;

    // iteration times
    int iteration_source_times = 1000, iteration_terminal_times = 1000;

    // parameters
    int ec_min = 1, ec_max = 100, V = 20000, E = 100000, hop_cst = 5;
    int thread_num = 1000;
    int Distributed_Graph_Num = 40;
    int G_max = 1000;
    int CPU_Num = 1, GPU_Num = 4;

    int use_cd = 0; // 0: according to rank 1: according to cd
    int check_correctness = 1;
    int label_type = 0; // 0: random graph, 1: real graph data, 2: from label file.

    // gpu case_info
    gpu_clean_info info_gpu;
    info_gpu.hop_cst = hop_cst;

    // cpu case_info
    hop_constrained_case_info info_cpu;
	info_cpu.upper_k = hop_cst;
	info_cpu.use_rank_prune = 1;
	info_cpu.use_2023WWW_generation = 0;
	info_cpu.use_canonical_repair = 1;
	info_cpu.max_run_time_seconds = 1000;
	info_cpu.thread_num = 100;

    // generate the graph and get the label
    if (label_type == 0) {
        instance_graph = graph_v_of_v_generate_random_graph<int>(V, E, ec_min, ec_max, 1, boost_random_time_seed);
        instance_graph = graph_v_of_v_update_vertexIDs_by_degrees_large_to_small(instance_graph); // sort vertices
        instance_graph.txt_save("../data/simple_iterative_tests.txt");
        
        printf("label type 0!\n");
        printf("Generation Start !\n");
        hop_constrained_two_hop_labels_generation(instance_graph, info_cpu, uncleaned_L);
        printf("Generation End !\n");
    } else if (label_type == 1) {

    } else {

    }

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

    // hop_constrained_check_correctness(mm, instance_graph, 10, 10, 5);
    std::cout << "CPU Clean Time: " << info_cpu.time_canonical_repair << "s" << endl;
    record_info_case.CPU_Clean_Time = info_cpu.time_canonical_repair;

    vector<vector<label>> L;
    int L_size = uncleaned_L.size();
    L.resize(L_size);
    for (int i = 0; i < L_size; i++) {
        L[i].resize(uncleaned_L[i].size());
        int _size = uncleaned_L[i].size();
        for (int j = 0; j < _size; j++) {
            L[i][j].v = uncleaned_L[i][j].hub_vertex, L[i][j].h = uncleaned_L[i][j].hop, L[i][j].d = uncleaned_L[i][j].distance;
        }
    }

    // vector<vector<hop_constrained_two_hop_label>> L_cpu; L_cpu.resize(L_size);
    // double CPU_clean_time;
    // auto begin = std::chrono::high_resolution_clock::now();
    // for (int i = 0; i < Distributed_Graph_Num; ++i) {
    //     hop_constrained_clean_L_distributed(info_cpu, uncleaned_L, L_cpu, graph_pool.graph_group[i], info_cpu.thread_num);
    // }
    // auto end = std::chrono::high_resolution_clock::now();
    // CPU_clean_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
    // printf("Time cpu distributed: %.6lfs\n", CPU_clean_time);

    // vector<vector<hop_constrained_two_hop_label>> L_gpu; L_gpu.resize(L_size);
    // double GPU_clean_time;
    // begin = std::chrono::high_resolution_clock::now();
    // gpu_clean_init(instance_graph, L, info_gpu, graph_pool, thread_num, hop_cst);
    // for (int i = 0; i < Distributed_Graph_Num; ++i) {
    //     gpu_clean(instance_graph, info_gpu, L_gpu, thread_num, i);
    // }
    // end = std::chrono::high_resolution_clock::now();
    // GPU_clean_time = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
    // printf("Time gpu distributed: %.6lfs\n", GPU_clean_time);

    vector<vector<hop_constrained_two_hop_label> > L_cpu_gpu_hybrid;    
    L_cpu_gpu_hybrid.resize(L_size);
    double Hybrid_clean_time = 0.0;

    priority_queue<Executive_Core> pq;
    while (!pq.empty()) pq.pop();
    for (int i = 0; i < CPU_Num; ++i) {
        pq.push(Executive_Core(i, 0.0, 0)); // id, time, cpu/gpu
    }
    for (int i = 0; i < GPU_Num; ++i) {
        pq.push(Executive_Core(CPU_Num + i, 0.0, 1)); // id, time, cpu/gpu
    }

    if (GPU_Num) {
        gpu_clean_init(instance_graph, L, info_gpu, graph_pool, thread_num, hop_cst);
    }
    for (int i = 0; i < Distributed_Graph_Num; ++i) {
        Executive_Core x = pq.top();
        pq.pop();
        
        auto begin = std::chrono::high_resolution_clock::now();
        printf("xxxxxxxxxxxxxx: %lf, %d, %d\n", x.time_clear, x.id, x.core_type);
        if (x.core_type == 0) { // core type is cpu
            hop_constrained_clean_L_distributed(info_cpu, uncleaned_L, L_cpu_gpu_hybrid, graph_pool.graph_group[i], info_cpu.thread_num);
        }else{
            gpu_clean(instance_graph, info_gpu, L_cpu_gpu_hybrid, thread_num, i);
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
        x.time_clear += duration;

        pq.push(x);
    }
    while (!pq.empty()) {
        Executive_Core x = pq.top();
        pq.pop();
        printf("xxxxxxxxxxxxxx: %lf, %d, %d\n", x.time_clear, x.id, x.core_type);
        Hybrid_clean_time = max(Hybrid_clean_time, x.time_clear);
    }
    printf("Time Hybrid: %.6lfs\n", Hybrid_clean_time);
    // record_info_case.GPU_Clean_Time = GPU_clean_time;

    auto& L_CPUclean = info_cpu.L;
    int uncleaned_L_num = 0, cleaned_L_num = 0, L_gpu_num = 0, L_cpu_num = 0, L_Hybrid_num = 0;
    for (int i = 0; i < L_size; i++) {
        uncleaned_L_num += L[i].size();
        cleaned_L_num += L_CPUclean[i].size();
        // L_gpu_num += L_gpu[i].size();
        // L_cpu_num += L_cpu[i].size();
        L_Hybrid_num += L_cpu_gpu_hybrid[i].size();
    }
    cout << "uncleaned_L_num: " << uncleaned_L_num << endl;
    cout << "cleaned_L_num: " << cleaned_L_num << endl;
    cout << "L_Hybrid_num: " << L_Hybrid_num << endl;

    if (check_correctness) {
        info_cpu.L = L_cpu_gpu_hybrid;
        cout << "check start !" << endl;
        hop_constrained_check_correctness(info_cpu, instance_graph, iteration_source_times, iteration_terminal_times, hop_cst);
        cout << "check end !" << endl;
    }

    return record_info_case;
}

int main () {

    int iteration_times = 1;

    double CPU_Clean_Time_avg = 0, GPU_Clean_Time_avg, Hybrid_Clean_Time_avg = 0;

    for (int i = 0; i < iteration_times; i++) {
        record_info x = main_element();
        CPU_Clean_Time_avg += x.CPU_Clean_Time / iteration_times;
        GPU_Clean_Time_avg += x.GPU_Clean_Time / iteration_times;
        Hybrid_Clean_Time_avg += x.Hybrid_Clean_Time / iteration_times;
    }

    cout << "CPU_Clean_Time_Avg: " << CPU_Clean_Time_avg << "s" <<endl;
    cout << "GPU_Clean_Time_Avg: " << GPU_Clean_Time_avg << "s" <<endl;
    cout << "Hybrid_Clean_Time_avg: " << Hybrid_Clean_Time_avg << "s" <<endl;
    cout << "Clean Finished !" << endl;
    return 0;
}