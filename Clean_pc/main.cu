//#include <graph_v_of_v/ldbc.hpp>
#include <cuda_runtime.h>
#include "HBPLL/test.h"
#include "HBPLL/gpu_clean.cuh"


#include <fstream>
#include <vector>
#include <string>

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


class record_info {
public:
    double CPU_Clean_Time, GPU_Clean_Time;
};

record_info main_element () {


    record_info record_info_case;

    int ec_min = 1, ec_max = 10;
    int V = 10000, E = 100000, tc = 8192;

    std::ios::sync_with_stdio(0);
    std::cin.tie(0);
    std::cout.tie(0);

    graph_v_of_v<int> instance_graph;
    int READ = 0;
    std::cin>>READ;
    if(READ)
    {
        instance_graph.txt_read("./simple_iterative_tests.txt");
    }
    else{
        instance_graph = graph_v_of_v_generate_random_graph<int>(V, E, ec_min, ec_max, 1, boost_random_time_seed);
        instance_graph = graph_v_of_v_update_vertexIDs_by_degrees_large_to_small(instance_graph); // sort vertices
        instance_graph.txt_save("./simple_iterative_tests.txt");

    }
    //instance_graph.txt_read("../data/simple_iterative_tests.txt");

    hop_constrained_case_info mm;
	mm.upper_k = 5;
	mm.use_rank_prune = 1;
	mm.use_2023WWW_generation = 0;
	mm.use_canonical_repair = 1;
	mm.max_run_time_seconds = 1000;
	mm.thread_num = 144;
    //mm.print_L = false;

    vector<vector<hop_constrained_two_hop_label>> uncleaned_L;
    printf("Generation Start !\n");
    if(!READ)
        hop_constrained_two_hop_labels_generation(instance_graph, mm, uncleaned_L);
    else{
        load_labels(uncleaned_L,"./label");
    }
    printf("Generation End !\n");
    // hop_constrained_check_correctness(mm, instance_graph, 10, 10, 5);
    std::cout << "CPU Clean Time: " << mm.time_canonical_repair << "s" << endl;
    record_info_case.CPU_Clean_Time = mm.time_canonical_repair;

    vector<vector<label>> L;

    vector<vector<hop_constrained_two_hop_label>> L_gpu;
    L_gpu.resize(uncleaned_L.size());
    if(!READ)
        save_labels(uncleaned_L,"./label");


    double GPU_clean_time =
        gpu_clean(instance_graph, uncleaned_L, L_gpu, tc, mm.upper_k);
    
    std::cout << "GPU Clean Finished" << std::endl;
    std::cout << "GPU Clean Time: " << GPU_clean_time << " s" << std::endl;
    record_info_case.GPU_Clean_Time = GPU_clean_time;

    auto& L_CPUclean = mm.L;
    int uncleaned_L_num = 0, L_gpu_num = 0, L_CPUclean_num = 0;
    for (int i = 0; i < uncleaned_L.size(); i++) {
        L_CPUclean_num += L_CPUclean[i].size();
        uncleaned_L_num += uncleaned_L[i].size();
        L_gpu_num += L_gpu[i].size();
    }
    cout << "L_CPU_clean_num: " << L_CPUclean_num << endl;    
    cout << "uncleaned_L_num: " << uncleaned_L_num << endl;
    cout << "L_GPU_num: " << L_gpu_num << endl;


    mm.L = L_gpu;
    hop_constrained_check_correctness(mm, instance_graph, 10, 10, 5);


    return record_info_case;
}

int main (int argc,char **argv) {

    int iteration_times = 1;

    double CPU_Clean_Time_avg = 0, GPU_Clean_Time_avg = 0;

    for (int i = 0; i < iteration_times; i++) {
        record_info x = main_element();
        CPU_Clean_Time_avg += x.CPU_Clean_Time / iteration_times;
        GPU_Clean_Time_avg += x.GPU_Clean_Time / iteration_times;
    }

    cout << "CPU_Clean_Time_Avg: " << CPU_Clean_Time_avg << "s" <<endl;
    cout << "GPU_Clean_Time_Avg: " << GPU_Clean_Time_avg << "s" <<endl;

    return 0;
}