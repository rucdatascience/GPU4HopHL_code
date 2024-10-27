#pragma once
#include "definition/hub_def.h"
#include "vgroup/CT/CT_labels.hpp"
#include "PLL.hpp"
#include <climits>
#include <cmath>
#include <graph/graph_v_of_v.h>
#include <graph/graph_v_of_v_update_vertexIDs_by_degrees_large_to_small.h>
#include <queue>
#include <unordered_set>
#include <vector>




/*global values*/
static graph_v_of_v<disType> global_ideal_graph_CT,
    global_eliminated_edge_predecessors; // weight of
                                         // global_eliminated_edge_predecessors
                                         // is predecessor of a merged edge; (it
                                         // is slightly faster to use
                                         // graph_v_of_v<int> than to use map)

static void clear_gloval_values_CT() {
  PLL_clear_global_values();
  global_ideal_graph_CT.clear();
  global_eliminated_edge_predecessors.clear();
}


static void CT_cores(graph_v_of_v<disType> &input_graph, CT_case_info &case_info) {
  auto begin1 = std::chrono::high_resolution_clock::now();

  //auto &Bags = case_info.Bags;
  auto &isIntree = case_info.isIntree;
  auto &root = case_info.root;
  auto &tree_st = case_info.tree_st;
  auto &tree_st_r = case_info.tree_st_r;
  auto &first_pos = case_info.first_pos;
  auto &lg = case_info.lg;
  auto &dep = case_info.dep;

  global_ideal_graph_CT = input_graph;

  int N = input_graph.size();
  isIntree.resize(N, 0); // whether it is in the CT-tree
  //vector<vector<int>> bag_predecessors(N);
  global_eliminated_edge_predecessors.ADJs.resize(N);
  
  /*priority_queue for maintaining the degrees of vertices (we do not update
   * degrees in q, so everytime you pop out a degree in q, you check whether it
   * is the right one, ignore it if wrong)*/
  priority_queue<node_degree> q;
  for (int i = 0; i < N; i++) {
    node_degree nd;
    nd.degree = global_ideal_graph_CT[i].size();
    nd.vertex = i;
    q.push(nd);
  }
  //------------------------------------------------------------------------------------------------------------------------------------

  //-------------------------------------------------- step 2: MDE-based tree
  // decomposition ------------------------------------------------------------

  /*MDE-based tree decomposition; generating bags*/
  int bound_lambda = N;
  //Bags.resize(N);
  vector<int> node_order(N + 1); // merging ID to original ID

  //ThreadPool pool(case_info.thread_num);
  //std::vector<std::future<int>> results; // return typename: xxx

  int last = -1;
  for (int i = 1; i <= N; i++) {
    node_degree nd;
    while (1) {
      nd = q.top();
      q.pop();
      if (!isIntree[nd.vertex] &&
          global_ideal_graph_CT[nd.vertex].size() == nd.degree)
        break; // nd.vertex is the lowest degree vertex not in tree
    }
    last = nd.vertex;
    int v_x = nd.vertex;          // the node with the minimum degree in G
    if (nd.degree >= case_info.d) // reach the boudary
    {
      bound_lambda = i - 1;
      q.push(nd);
      case_info.tree_vertex_num = i - 1;
      break; // until |Ni| >= d
    }

    isIntree[v_x] = 1; // add to CT-tree
    node_order[i] = v_x;

    auto &adj_temp =
        global_ideal_graph_CT[v_x]; // global_ideal_graph_CT is G_i-1
    int v_adj_size = adj_temp.size();
    // for (int j = 0; j < v_adj_size; j++) {
     
    //   Bags[v_x].push_back(
    //       {adj_temp[j].first,
    //        adj_temp[j]
    //            .second}); // Bags[v_x] stores adj vertices and weights of v_x
    //   //int pred = adj_temp[j].first;
    //   // bag_predecessors[v_x].push_back(
    //   //     adj_temp[j].first); // bag_predecessors[v_x] stores predecessors (in
    //   //                         // merged graphs) for vertices in Bags[v_x]
    // }

    /*add new edge*/
    for (int j = 0; j < v_adj_size; j++) {
      int adj_j = adj_temp[j].first;
      for (int k = j + 1; k < v_adj_size; k++) {
        int adj_k = adj_temp[k].first;
        int new_ec = adj_temp[j].second + adj_temp[k].second;

        int pos = sorted_vector_binary_operations_search_position(
            global_ideal_graph_CT[adj_j], adj_k);
        if (pos == -1 || new_ec < global_ideal_graph_CT[adj_j][pos].second) {
          global_ideal_graph_CT.add_edge(adj_j, adj_k, new_ec);
          global_eliminated_edge_predecessors.add_edge(adj_j, adj_k, v_x);
        }
      }
    }

    // delete edge of v_x and update degree (due to added edges above and
    // deleted edge below)
    for (int j = 0; j < v_adj_size; j++) {
      int m = adj_temp[j].first;
      // update degree
      nd.vertex = m;
      nd.degree =
          global_ideal_graph_CT[m].size() -
          1; // v_x will be removed from global_ideal_graph_CT[nd.vertex] below
      q.push(nd);
      // remove v_x
      int pos = sorted_vector_binary_operations_search_position(
          global_ideal_graph_CT[m], v_x);
      global_ideal_graph_CT[m].erase(global_ideal_graph_CT[m].begin() + pos);
    }
    // delete v_x from ideal graph directly
    vector<pair<int, disType>>().swap(global_ideal_graph_CT[v_x]);
  }

  
}