#pragma once
#include "definition/hub_def.h"
#include "graph_v_of_v/graph_v_of_v.h"
#include "vgroup/CDLP/GPU_Community_Detection.cuh"
#include <unordered_map>
#include <vector>
#include "chrono"

static void generate_Group_CDLP(graph_v_of_v<int> &instance_graph, std::vector<std::vector<int>> &groups, int MAX_GROUP_SIZE) {
    printf("debug CDLP !!! \n");
    auto start = std::chrono::high_resolution_clock::now();
    // ��ͼת��Ϊ CSR ��ʽ
    CSR_graph<int> csr = graph_v_of_v_to_CSR<int>(instance_graph);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "transform graph took " << duration.count() << " seconds." << std::endl;

    // ��ʼ����ǩ����
    std::vector<int> labels(instance_graph.size(), 0);

    // ִ�� CDLP �㷨
    start = std::chrono::high_resolution_clock::now();
    CDLP_GPU(instance_graph.size(), csr, labels, MAX_GROUP_SIZE, 1000);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "CDLP took " << duration.count() << " seconds." << std::endl;

    // ȷ�� groups �Ĵ�С�㹻
    groups.resize(*max_element(labels.begin(), labels.end()) + 1);

    start = std::chrono::high_resolution_clock::now();
    // ���ݱ�ǩ���ڵ����
    for (int node_id = 0; node_id < labels.size(); node_id++) {
        groups[labels[node_id]].push_back(node_id);
    }
    //ȥ����СΪ 0 �� group
    groups.erase(std::remove_if(groups.begin(), groups.end(), [](const std::vector<int>& group) { return group.size() == 0; }), groups.end());
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "gen groups took " << duration.count() << " seconds." << std::endl;

}