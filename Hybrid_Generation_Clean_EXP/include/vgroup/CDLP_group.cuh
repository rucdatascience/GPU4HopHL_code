#pragma once
#include "definition/hub_def.h"
#include "graph_v_of_v/graph_v_of_v.h"
#include "vgroup/CDLP/GPU_Community_Detection.cuh"
#include <unordered_map>
#include <vector>
#include "chrono"

struct group_union_unit {
    int id;
    long long group_size;
    std::vector<int> group_node;
    inline bool operator < (const group_union_unit b) const { // a < b
        if (group_size == b.group_size) return id < b.id;
        else return group_size < b.group_size;
	}
};
std::set<group_union_unit> se;

static void generate_Group_CDLP(graph_v_of_v<int> &instance_graph, std::vector<std::vector<int>> &groups, int MAX_GROUP_SIZE) {
    printf("debug CDLP !!! \n");
    auto start = std::chrono::high_resolution_clock::now();
    // 将图转换为 CSR 格式
    CSR_graph<int> csr = graph_v_of_v_to_CSR<int>(instance_graph);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;
    std::cout << "transform graph took " << duration.count() << " seconds." << std::endl;

    // 初始化标签向量
    std::vector<int> labels(instance_graph.size(), 0);

    // 执行 CDLP 算法
    start = std::chrono::high_resolution_clock::now();
    CDLP_GPU(instance_graph.size(), csr, labels, MAX_GROUP_SIZE, 10000);
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "CDLP took " << duration.count() << " seconds." << std::endl;

    std::vector<std::vector<int>> groups_after;

    // 确保 groups 的大小足够
    groups.resize(*max_element(labels.begin(), labels.end()) + 1);

    start = std::chrono::high_resolution_clock::now();
    // 根据标签将节点分组
    for (int node_id = 0; node_id < labels.size(); node_id++) {
        groups[labels[node_id]].push_back(node_id);
    }
    //去除大小为 0 的 group
    groups.erase(std::remove_if(groups.begin(), groups.end(), [](const std::vector<int>& group) { return group.size() == 0; }), groups.end());
    
    for (int group_id = 0; group_id < groups.size(); group_id++) {
        printf("groups size : %d\n", groups[group_id].size());
        se.insert((group_union_unit){group_id, groups[group_id].size(), groups[group_id]});
    }

    // merge the small subsets
    std::set<group_union_unit>::iterator it1 = se.end(), it2;
    it1--;
    printf("max groups size: %d\n", (*it1).group_size);
    
    while (!se.empty()) {
        it1 = --se.end();
        int ret_group_size = MAX_GROUP_SIZE - (*it1).group_size;
        it2 = --se.upper_bound((group_union_unit){groups.size(), ret_group_size, groups[0]});
        if ((*it1).id == (*it2).id) {
            -- it2;
        }
        if (it2 != se.end() && (*it1).id != (*it2).id && (*it1).group_size + (*it2).group_size <= MAX_GROUP_SIZE) {
            group_union_unit a = (*it1), b = (*it2);
            a.group_size += b.group_size;
            a.group_node.insert(a.group_node.end(), b.group_node.begin(), b.group_node.end());
            se.insert(a);
            se.erase(it1), se.erase(it2);
        } else {
            groups_after.push_back((*it1).group_node);
            se.erase(it1);
        }
    }
    
    // sort as rank
    std::vector<group_union_unit> group_sort;
    for (int group_id = 0; group_id < groups_after.size(); ++group_id) {
        long long tot_degree = 0;
        for (int j = 0; j < groups_after[group_id].size(); ++j) {
            tot_degree -= instance_graph[groups_after[group_id][j]].size();
        }
        group_sort.push_back((group_union_unit){group_id, tot_degree, groups_after[group_id]});
    }
    sort(group_sort.begin(), group_sort.end());

    groups.resize(group_sort.size());
    for (int group_id = 0; group_id < groups_after.size(); ++group_id) {
        groups[group_id] = group_sort[group_id].group_node;
    }
    // groups = groups_after;
    int group_tot_node = 0;
    for (int group_id = 0; group_id < groups.size(); group_id++) {
        printf("groups after size : %d\n", groups[group_id].size());
        group_tot_node += groups[group_id].size();
    }
    printf("group_tot_node: %d\n", group_tot_node);
    
    end = std::chrono::high_resolution_clock::now();
    duration = end - start;
    std::cout << "gen groups took " << duration.count() << " seconds." << std::endl;

}