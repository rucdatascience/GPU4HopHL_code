#pragma once
#include "definition/hub_def.h"
#include "graph/graph_v_of_v.h"
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <random>


static void
generate_Group_Random(graph_v_of_v<disType> &instance_graph, int hop_cst, std::vector<std::vector<int>> &groups) {
    //根据 MAX_GROUP_SIZE 限制每个group的大小，随机的将点分配到group中
    int N = instance_graph.size();

    // 创建一个包含所有节点的列表
    std::vector<int> nodes(N);
    for (int i = 0; i < N; ++i) {
        nodes[i] = i;
    }

    // 随机打乱节点列表
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(nodes.begin(), nodes.end(), g);
    groups.resize(N);

    // 将节点分配到组中
    int group_id = 0;
    for (int i = 0; i < N; ++i) {
        if (groups[group_id].size() >= MAX_GROUP_SIZE) {
            ++group_id;
        }
        groups[group_id].push_back(nodes[i]);
    }
}