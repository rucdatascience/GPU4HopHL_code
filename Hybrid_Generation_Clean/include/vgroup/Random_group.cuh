#pragma once
#include "definition/hub_def.h"
#include "graph/graph_v_of_v.h"
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <random>


static void
generate_Group_Random(graph_v_of_v<disType> &instance_graph, int hop_cst, std::vector<std::vector<int>> &groups) {
    //���� MAX_GROUP_SIZE ����ÿ��group�Ĵ�С������Ľ�����䵽group��
    int N = instance_graph.size();

    // ����һ���������нڵ���б�
    std::vector<int> nodes(N);
    for (int i = 0; i < N; ++i) {
        nodes[i] = i;
    }

    // ������ҽڵ��б�
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(nodes.begin(), nodes.end(), g);
    groups.resize(N);

    // ���ڵ���䵽����
    int group_id = 0;
    for (int i = 0; i < N; ++i) {
        if (groups[group_id].size() >= MAX_GROUP_SIZE) {
            ++group_id;
        }
        groups[group_id].push_back(nodes[i]);
    }
}