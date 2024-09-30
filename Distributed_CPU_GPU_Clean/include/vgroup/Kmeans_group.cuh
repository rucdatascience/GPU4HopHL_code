#ifndef KMEANS_GROUP_H
#define KMEANS_GROUP_H
#include "utilities/dijkstra.cuh"
#include <cstdio>
#include <graph/graph_v_of_v.h>
#include <shared_mutex>
#include <stdio.h>
#include <tool_functions/ThreadPool.h>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>




// algo
void generate_Group_kmeans(graph_v_of_v<disType> &instance_graph, int hop_cst, std::vector<std::vector<int>> &groups);

int find_new_center(vector<int> &cluster, graph_v_of_v<disType> &graph, dijkstra_table &dt);

vector<int> get_centers(vector<pair<int, int>> &graph, int &start, vector<bool> &chosen, int nums);

// func
void print_groups(std::vector<int, std::vector<int>> &groups);

#endif