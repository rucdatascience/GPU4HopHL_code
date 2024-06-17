#pragma once
#include <boost/heap/fibonacci_heap.hpp>
//#include <cpp_redis/cpp_redis>
#include <graph_v_of_v/graph_v_of_v.h>
#include <hop_constrained_two_hop_labels.h>
#include <shared_mutex>
#include <tool_functions/ThreadPool.h>
#include <two_hop_labels.h>
#include <unordered_map>
#include <unordered_set>

using LabelSet = std::unordered_set<hop_constrained_two_hop_label>;
vector<LabelSet> unionedLabels; //每个group结束后，将所有group的label合并到这里

/*unique code for this file: 599*/
long long int max_labal_size_599;
long long int labal_size_599;
int max_N_ID_for_mtx_599 = 1e7;
double max_run_time_nanoseconds_599;
int global_upper_k;
long long int label_size_before_canonical_repair_599,
    label_size_after_canonical_repair_599;
auto begin_time_599 = std::chrono::high_resolution_clock::now();
vector<std::shared_timed_mutex> mtx_599(max_N_ID_for_mtx_599);
graph_v_of_v<int> ideal_graph_599;
vector<vector<hop_constrained_two_hop_label>> L_temp_599;
vector<vector<vector<pair<int, int>>>> Temp_L_vk_599;
vector<vector<pair<int, int>>> dist_hop_599;
vector<vector<vector<int>>> Vh_599;
queue<int> Qid_599;

typedef typename boost::heap::fibonacci_heap<
    hop_constrained_two_hop_label>::handle_type hop_constrained_node_handle;
vector<vector<vector<pair<hop_constrained_node_handle, int>>>>
    Q_handle_priorities_599;

void HSDL_thread_function_insert_to_union(int v_k) {

  if (labal_size_599 > max_labal_size_599) {
    throw reach_limit_error_string_MB;
  }
  if (std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::high_resolution_clock::now() - begin_time_599)
          .count() > max_run_time_nanoseconds_599) {
    throw reach_limit_error_string_time;
  }

  /* get unique thread id */
  mtx_599[max_N_ID_for_mtx_599 - 1].lock();
  int used_id = Qid_599.front();
  Qid_599.pop();
  mtx_599[max_N_ID_for_mtx_599 - 1].unlock();

  vector<int> Temp_L_vk_changes, dist_hop_changes;
  auto &Temp_L_vk = Temp_L_vk_599[used_id];
  auto &dist_hop =
      dist_hop_599[used_id]; // record the minimum distance (and the
                             // corresponding hop) of a searched vertex in Q
  vector<pair<int, int>> Q_handle_priorities_changes;
  auto &Q_handle_priorities = Q_handle_priorities_599[used_id];

  long long int new_label_num = 0;

  boost::heap::fibonacci_heap<hop_constrained_two_hop_label> Q;

  hop_constrained_two_hop_label node;
  node.hub_vertex = v_k;
  node.parent_vertex = v_k;
  node.hop = 0;
  node.distance = 0;
  Q_handle_priorities[v_k][0] = {Q.push({node}), node.distance};
  Q_handle_priorities_changes.push_back({v_k, 0});

  /* Temp_L_vk_599 stores the label (dist and hop) of vertex v_k */
  mtx_599[v_k].lock();
  L_temp_599[v_k].push_back(node);
  // insert to unionedLabels
  unionedLabels[v_k].insert(node);

  new_label_num++;
  for (auto &xx : L_temp_599[v_k]) {
    int L_vk_vertex = xx.hub_vertex;
    Temp_L_vk[L_vk_vertex].push_back({xx.distance, xx.hop});
    Temp_L_vk_changes.push_back(L_vk_vertex);
  }
  mtx_599[v_k].unlock();

  /*  dist_hop_599 stores the shortest distance from vk to any other vertices
     with its hop_cst, note that the hop_cst is determined by the shortest
     distance */
  dist_hop[v_k] = {0, 0};
  dist_hop_changes.push_back(v_k);

  while (Q.size() > 0) {

    node = Q.top();
    Q.pop();
    int u = node.hub_vertex;

    if (v_k <=
        u) { // = is to start the while loop by searching neighbors of v_k

      int u_parent = node.parent_vertex;
      int u_hop = node.hop;
      int P_u = node.distance;

      int query_v_k_u = std::numeric_limits<int>::max();
      mtx_599[u].lock();
      for (auto &xx : L_temp_599[u]) {
        int common_v = xx.hub_vertex;
        for (auto &yy : Temp_L_vk[common_v]) {
          if (xx.hop + yy.second <= u_hop) {
            long long int dis = (long long int)xx.distance + yy.first;
            if (query_v_k_u > dis) {
              query_v_k_u = dis;
            }
          }
        }
      }
      mtx_599[u].unlock();

      // if (u == 4 && v_k == 0 && u_hop == 3 && P_u == 8) {
      //	cout << "xx: " << P_u << " " << query_v_k_u << endl;
      // }

      if (P_u < query_v_k_u ||
          query_v_k_u == 0) { // query_v_k_u == 0 is to start the while loop by
                              // searching neighbors of v_k

        if (P_u < query_v_k_u) {
          node.hub_vertex = v_k;
          node.hop = u_hop;
          node.distance = P_u;
          node.parent_vertex = u_parent;
          mtx_599[u].lock();
          L_temp_599[u].push_back(node);
          // insert into unionedLabels
          unionedLabels[u].insert(node);
          mtx_599[u].unlock();
          new_label_num++;
        }

        if (u_hop + 1 > global_upper_k) {
          continue;
        }

        /* update adj */
        for (auto &xx : ideal_graph_599[u]) {
          int adj_v = xx.first, ec = xx.second;

          if (global_use_2M_prune && P_u + ec >= TwoM_value) {
            continue;
          }

          /* update node info */
          node.hub_vertex = adj_v;
          node.parent_vertex = u;
          node.distance = P_u + ec;
          node.hop = u_hop + 1;

          auto &yy = Q_handle_priorities[adj_v][node.hop];

          /*directly using the following codes without dist_hop is OK, but is
           * slower; dist_hop is a pruning technique without increasing the time
           * complexity*/
          // if (yy.second != std::numeric_limits<int>::max()) {
          //	if (yy.second > node.distance) {
          //		Q.update(yy.first, node);
          //		yy.second = node.distance;
          //	}
          // }
          // else {
          //	yy = { Q.push(node), node.distance };
          //	Q_handle_priorities_changes.push_back({ adj_v, node.hop });
          // }

          if (yy.second <= node.distance) { // adj_v has been reached with a
                                            // smaller distance and the same hop
            continue;
          }
          if (dist_hop[adj_v].first ==
              std::numeric_limits<int>::max()) { // adj_v has not been reached
            yy = {Q.push({node}), node.distance};
            Q_handle_priorities_changes.push_back({adj_v, node.hop});
            dist_hop[adj_v].first = node.distance;
            dist_hop[adj_v].second = node.hop;
            dist_hop_changes.push_back(adj_v);
          } else {
            if (node.distance <
                dist_hop[adj_v]
                    .first) { // adj_v has been reached with a larger distance
              if (yy.second != std::numeric_limits<int>::max()) {
                Q.update(yy.first, node);
                yy.second = node.distance;
              } else {
                yy = {Q.push(node), node.distance};
                Q_handle_priorities_changes.push_back({adj_v, node.hop});
              }
              dist_hop[adj_v].first = node.distance;
              dist_hop[adj_v].second = node.hop;
            } else if (node.hop <
                       dist_hop[adj_v]
                           .second) { // adj_v has been reached with a larger
                                      // distance and a larger hop
              if (yy.second != std::numeric_limits<int>::max()) {
                Q.update(yy.first, node);
                yy.second = node.distance;
              } else {
                yy = {Q.push(node), node.distance};
                Q_handle_priorities_changes.push_back({adj_v, node.hop});
              }
            }
          }
        }
      }
    }
  }

  for (auto &xx : Temp_L_vk_changes) {
    vector<pair<int, int>>().swap(Temp_L_vk[xx]);
  }
  for (auto &xx : dist_hop_changes) {
    dist_hop[xx] = {std::numeric_limits<int>::max(), 0};
  }
  hop_constrained_node_handle handle_x;
  for (auto &xx : Q_handle_priorities_changes) {
    Q_handle_priorities[xx.first][xx.second] = {
        handle_x, std::numeric_limits<int>::max()};
  }

  mtx_599[v_k].lock();
  vector<hop_constrained_two_hop_label>(L_temp_599[v_k]).swap(L_temp_599[v_k]);
  mtx_599[v_k].unlock();

  mtx_599[max_N_ID_for_mtx_599 - 1].lock();
  Qid_599.push(used_id);
  labal_size_599 = labal_size_599 + new_label_num;
  mtx_599[max_N_ID_for_mtx_599 - 1].unlock();
}

void clear_unionedLabels() { vector<LabelSet>().swap(unionedLabels); }

void hop_constrained_clear_global_values() {
  vector<vector<hop_constrained_two_hop_label>>().swap(L_temp_599);
  ideal_graph_599.clear();
  vector<vector<vector<pair<int, int>>>>().swap(Temp_L_vk_599);
  vector<vector<pair<int, int>>>().swap(dist_hop_599);
  vector<vector<vector<pair<hop_constrained_node_handle, int>>>>().swap(
      Q_handle_priorities_599);
  vector<vector<vector<int>>>().swap(Vh_599);
  queue<int>().swap(Qid_599);
}

void HSDL_thread_function(int v_k) {

  if (labal_size_599 > max_labal_size_599) {
    throw reach_limit_error_string_MB;
  }
  if (std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::high_resolution_clock::now() - begin_time_599)
          .count() > max_run_time_nanoseconds_599) {
    throw reach_limit_error_string_time;
  }

  /* get unique thread id */
  mtx_599[max_N_ID_for_mtx_599 - 1].lock();
  int used_id = Qid_599.front();
  Qid_599.pop();
  mtx_599[max_N_ID_for_mtx_599 - 1].unlock();

  vector<int> Temp_L_vk_changes, dist_hop_changes;
  auto &Temp_L_vk = Temp_L_vk_599[used_id];
  auto &dist_hop =
      dist_hop_599[used_id]; // record the minimum distance (and the
                             // corresponding hop) of a searched vertex in Q
  vector<pair<int, int>> Q_handle_priorities_changes;
  auto &Q_handle_priorities = Q_handle_priorities_599[used_id];

  long long int new_label_num = 0;

  boost::heap::fibonacci_heap<hop_constrained_two_hop_label> Q;

  hop_constrained_two_hop_label node;
  node.hub_vertex = v_k;
  node.parent_vertex = v_k;
  node.hop = 0;
  node.distance = 0;
  Q_handle_priorities[v_k][0] = {Q.push({node}), node.distance};
  Q_handle_priorities_changes.push_back({v_k, 0});

  /* Temp_L_vk_599 stores the label (dist and hop) of vertex v_k */
  mtx_599[v_k].lock();
  L_temp_599[v_k].push_back(node);
  new_label_num++;
  for (auto &xx : L_temp_599[v_k]) {
    int L_vk_vertex = xx.hub_vertex;
    Temp_L_vk[L_vk_vertex].push_back({xx.distance, xx.hop});
    Temp_L_vk_changes.push_back(L_vk_vertex);
  }
  mtx_599[v_k].unlock();

  /*  dist_hop_599 stores the shortest distance from vk to any other vertices
     with its hop_cst, note that the hop_cst is determined by the shortest
     distance */
  dist_hop[v_k] = {0, 0};
  dist_hop_changes.push_back(v_k);

  while (Q.size() > 0) {

    node = Q.top();
    Q.pop();
    int u = node.hub_vertex;

    if (v_k <=
        u) { // = is to start the while loop by searching neighbors of v_k

      int u_parent = node.parent_vertex;
      int u_hop = node.hop;
      int P_u = node.distance;

      int query_v_k_u = std::numeric_limits<int>::max();
      mtx_599[u].lock();
      for (auto &xx : L_temp_599[u]) {
        int common_v = xx.hub_vertex;
        for (auto &yy : Temp_L_vk[common_v]) {
          if (xx.hop + yy.second <= u_hop) {
            long long int dis = (long long int)xx.distance + yy.first;
            if (query_v_k_u > dis) {
              query_v_k_u = dis;
            }
          }
        }
      }
      mtx_599[u].unlock();

      // if (u == 4 && v_k == 0 && u_hop == 3 && P_u == 8) {
      //	cout << "xx: " << P_u << " " << query_v_k_u << endl;
      // }

      if (P_u < query_v_k_u ||
          query_v_k_u == 0) { // query_v_k_u == 0 is to start the while loop by
                              // searching neighbors of v_k

        if (P_u < query_v_k_u) {
          node.hub_vertex = v_k;
          node.hop = u_hop;
          node.distance = P_u;
          node.parent_vertex = u_parent;
          mtx_599[u].lock();
          L_temp_599[u].push_back(node);
          mtx_599[u].unlock();
          new_label_num++;
        }

        if (u_hop + 1 > global_upper_k) {
          continue;
        }

        /* update adj */
        for (auto &xx : ideal_graph_599[u]) {
          int adj_v = xx.first, ec = xx.second;

          if (global_use_2M_prune && P_u + ec >= TwoM_value) {
            continue;
          }

          /* update node info */
          node.hub_vertex = adj_v;
          node.parent_vertex = u;
          node.distance = P_u + ec;
          node.hop = u_hop + 1;

          auto &yy = Q_handle_priorities[adj_v][node.hop];

          /*directly using the following codes without dist_hop is OK, but is
           * slower; dist_hop is a pruning technique without increasing the time
           * complexity*/
          // if (yy.second != std::numeric_limits<int>::max()) {
          //	if (yy.second > node.distance) {
          //		Q.update(yy.first, node);
          //		yy.second = node.distance;
          //	}
          // }
          // else {
          //	yy = { Q.push(node), node.distance };
          //	Q_handle_priorities_changes.push_back({ adj_v, node.hop });
          // }

          if (yy.second <= node.distance) { // adj_v has been reached with a
                                            // smaller distance and the same hop
            continue;
          }
          if (dist_hop[adj_v].first ==
              std::numeric_limits<int>::max()) { // adj_v has not been reached
            yy = {Q.push({node}), node.distance};
            Q_handle_priorities_changes.push_back({adj_v, node.hop});
            dist_hop[adj_v].first = node.distance;
            dist_hop[adj_v].second = node.hop;
            dist_hop_changes.push_back(adj_v);
          } else {
            if (node.distance <
                dist_hop[adj_v]
                    .first) { // adj_v has been reached with a larger distance
              if (yy.second != std::numeric_limits<int>::max()) {
                Q.update(yy.first, node);
                yy.second = node.distance;
              } else {
                yy = {Q.push(node), node.distance};
                Q_handle_priorities_changes.push_back({adj_v, node.hop});
              }
              dist_hop[adj_v].first = node.distance;
              dist_hop[adj_v].second = node.hop;
            } else if (node.hop <
                       dist_hop[adj_v]
                           .second) { // adj_v has been reached with a larger
                                      // distance and a larger hop
              if (yy.second != std::numeric_limits<int>::max()) {
                Q.update(yy.first, node);
                yy.second = node.distance;
              } else {
                yy = {Q.push(node), node.distance};
                Q_handle_priorities_changes.push_back({adj_v, node.hop});
              }
            }
          }
        }
      }
    }
  }

  for (auto &xx : Temp_L_vk_changes) {
    vector<pair<int, int>>().swap(Temp_L_vk[xx]);
  }
  for (auto &xx : dist_hop_changes) {
    dist_hop[xx] = {std::numeric_limits<int>::max(), 0};
  }
  hop_constrained_node_handle handle_x;
  for (auto &xx : Q_handle_priorities_changes) {
    Q_handle_priorities[xx.first][xx.second] = {
        handle_x, std::numeric_limits<int>::max()};
  }

  mtx_599[v_k].lock();
  vector<hop_constrained_two_hop_label>(L_temp_599[v_k]).swap(L_temp_599[v_k]);
  mtx_599[v_k].unlock();

  mtx_599[max_N_ID_for_mtx_599 - 1].lock();
  Qid_599.push(used_id);
  labal_size_599 = labal_size_599 + new_label_num;
  mtx_599[max_N_ID_for_mtx_599 - 1].unlock();
}

void _2023WWW_thread_function(int v_k) {

  if (labal_size_599 > max_labal_size_599) {
    throw reach_limit_error_string_MB;
  }
  if (std::chrono::duration_cast<std::chrono::nanoseconds>(
          std::chrono::high_resolution_clock::now() - begin_time_599)
          .count() > max_run_time_nanoseconds_599) {
    throw reach_limit_error_string_time;
  }

  /* get unique thread id */
  mtx_599[max_N_ID_for_mtx_599 - 1].lock();
  int used_id = Qid_599.front();
  Qid_599.pop();
  mtx_599[max_N_ID_for_mtx_599 - 1].unlock();

  vector<int> Temp_L_vk_changes, dist_hop_changes;
  auto &Temp_L_vk = Temp_L_vk_599[used_id];
  auto &dist_hop = dist_hop_599[used_id]; // record {dis, predecessor}
  auto &Vh = Vh_599[used_id];

  long long int new_label_num = 0;

  hop_constrained_two_hop_label node;
  node.hub_vertex = v_k;
  node.parent_vertex = v_k;
  node.hop = 0;
  node.distance = 0;

  /* Temp_L_vk_599 stores the label (dist and hop) of vertex v_k */
  mtx_599[v_k].lock();
  // L_temp_599[v_k].push_back(node); new_label_num++;
  for (auto &xx : L_temp_599[v_k]) {
    int L_vk_vertex = xx.hub_vertex;
    Temp_L_vk[L_vk_vertex].push_back({xx.distance, xx.hop});
    Temp_L_vk_changes.push_back(L_vk_vertex);
  }
  mtx_599[v_k].unlock();

  Vh[0].push_back(v_k);

  dist_hop[v_k] = {0, v_k};
  dist_hop_changes.push_back(v_k);

  vector<tuple<int, int, int>> dh_updates;

  for (int h = 0; h <= global_upper_k; h++) {

    for (auto &xx : dh_updates) {
      if (dist_hop[get<0>(xx)].first > get<1>(xx)) {
        dist_hop[get<0>(xx)] = {get<1>(xx), get<2>(xx)};
        dist_hop_changes.push_back(get<0>(xx));
      }
    }
    vector<tuple<int, int, int>>().swap(dh_updates);

    for (auto u : Vh[h]) {
      int P_u = dist_hop[u].first;

      if (std::chrono::duration_cast<std::chrono::nanoseconds>(
              std::chrono::high_resolution_clock::now() - begin_time_599)
              .count() > max_run_time_nanoseconds_599) {
        throw reach_limit_error_string_time;
      }

      // if (u < v_k) { // rank pruning
      // 貌似并不能明显加速_2023WWW_thread_function，_2023WWW_thread_function之所以慢是因为BFS生成了过多的冗余的需要被清洗的label（冗余的比非冗余的多几乎一个数量级）
      //	continue;
      // }

      int query_v_k_u = std::numeric_limits<int>::max();
      mtx_599[u].lock();
      for (auto &xx : L_temp_599[u]) {
        int common_v = xx.hub_vertex;
        for (auto &yy : Temp_L_vk[common_v]) {
          if (xx.hop + yy.second <= h) {
            long long int dis = (long long int)xx.distance + yy.first;
            if (query_v_k_u > dis) {
              query_v_k_u = dis;
            }
          }
        }
      }
      mtx_599[u].unlock();

      if (P_u < query_v_k_u) {

        node.hub_vertex = v_k;
        node.hop = h;
        node.distance = P_u;
        node.parent_vertex = dist_hop[u].second;
        mtx_599[u].lock();
        L_temp_599[u].push_back(node);
        mtx_599[u].unlock();
        new_label_num++;

        /* update adj */
        for (auto &xx : ideal_graph_599[u]) {
          int adj_v = xx.first, ec = xx.second;
          if (P_u + ec < dist_hop[adj_v].first) {
            Vh[h + 1].push_back(adj_v);
            dh_updates.push_back({adj_v, P_u + ec, u});
          }
        }
      }
    }
  }

  for (auto &xx : Temp_L_vk_changes) {
    vector<pair<int, int>>().swap(Temp_L_vk[xx]);
  }
  for (int i = 0; i <= global_upper_k; i++) {
    vector<int>().swap(Vh[i]);
  }
  for (auto &xx : dist_hop_changes) {
    dist_hop[xx] = {std::numeric_limits<int>::max(), 0};
  }

  mtx_599[v_k].lock();
  vector<hop_constrained_two_hop_label>(L_temp_599[v_k]).swap(L_temp_599[v_k]);
  mtx_599[v_k].unlock();

  mtx_599[max_N_ID_for_mtx_599 - 1].lock();
  Qid_599.push(used_id);
  labal_size_599 = labal_size_599 + new_label_num;
  mtx_599[max_N_ID_for_mtx_599 - 1].unlock();
}

/*sortL*/
bool compare_hop_constrained_two_hop_label(hop_constrained_two_hop_label &i,
                                           hop_constrained_two_hop_label &j) {
  if (i.hub_vertex != j.hub_vertex) {
    return i.hub_vertex < j.hub_vertex;
  } else if (i.hop != j.hop) {
    return i.hop < j.hop;
  } else {
    return i.distance < j.distance;
  }
}
vector<vector<hop_constrained_two_hop_label>>
hop_constrained_sortL(int num_of_threads) {

  /*time complexity: O(V*L*logL), where L is average number of labels per
   * vertex*/

  int N = L_temp_599.size();
  vector<vector<hop_constrained_two_hop_label>> output_L(N);

  /*time complexity: O(V*L*logL), where L is average number of labels per
   * vertex*/
  ThreadPool pool(num_of_threads);
  std::vector<std::future<int>> results; // return typename: xxx
  for (int v_k = 0; v_k < N; v_k++) {
    results.emplace_back(pool.enqueue(
        [&output_L, v_k] { // pass const type value j to thread; [] can be empty
          sort(L_temp_599[v_k].begin(), L_temp_599[v_k].end(),
               compare_hop_constrained_two_hop_label);
          vector<hop_constrained_two_hop_label>(L_temp_599[v_k])
              .swap(L_temp_599[v_k]); // swap释放vector中多余空间
          output_L[v_k] = L_temp_599[v_k];
          vector<hop_constrained_two_hop_label>().swap(
              L_temp_599[v_k]); // clear new labels for RAM efficiency

          return 1; // return to results; the return type must be the same with
                    // results
        }));
  }
  for (auto &&result : results)
    result.get(); // all threads finish here

  return output_L;
}

/*canonical_repair*/
void hop_constrained_clean_L(hop_constrained_case_info &case_info,
                             int thread_num) {

  auto &L = case_info.L;
  int N = L.size();
  label_size_before_canonical_repair_599 = 0;
  label_size_after_canonical_repair_599 = 0;

  ThreadPool pool(thread_num);
  std::vector<std::future<int>> results;

  for (int v = 0; v < N; v++) {
    results.emplace_back(pool.enqueue(
        [v, &L] { // pass const type value j to thread; [] can be empty
          mtx_599[max_N_ID_for_mtx_599 - 1].lock();
          int used_id = Qid_599.front();
          Qid_599.pop();
          mtx_599[max_N_ID_for_mtx_599 - 1].unlock();

          vector<hop_constrained_two_hop_label> Lv_final;

          mtx_599[v].lock_shared();
          vector<hop_constrained_two_hop_label> Lv = L[v];
          mtx_599[v].unlock_shared();
          label_size_before_canonical_repair_599 += Lv.size();

          auto &T = Temp_L_vk_599[used_id];

          for (auto Lvi : Lv) {
            int u = Lvi.hub_vertex;
            int u_hop = Lvi.hop;

            mtx_599[u].lock_shared();
            auto Lu = L[u];
            mtx_599[u].unlock_shared();

            int min_dis = std::numeric_limits<int>::max();
            for (auto &label1 : Lu) {
              for (auto &label2 : T[label1.hub_vertex]) {
                if (label1.hop + label2.second <= u_hop) {
                  long long int query_dis =
                      label1.distance + (long long int)label2.first;
                  if (query_dis < min_dis) {
                    min_dis = query_dis;
                  }
                }
              }
            }

            if (min_dis > Lvi.distance) {
              Lv_final.push_back(Lvi);
              T[u].push_back({Lvi.distance, Lvi.hop});
            }
          }

          for (auto label : Lv_final) {
            vector<pair<int, int>>().swap(T[label.hub_vertex]);
          }

          mtx_599[v].lock();
          L[v] = Lv_final;
          L[v].shrink_to_fit();
          mtx_599[v].unlock();
          label_size_after_canonical_repair_599 += Lv_final.size();

          mtx_599[max_N_ID_for_mtx_599 - 1].lock();
          Qid_599.push(used_id);
          mtx_599[max_N_ID_for_mtx_599 - 1].unlock();

          return 1; // return to results; the return type must be the same with
                    // results
        }));
  }

  for (auto &&result : results)
    result.get(); // all threads finish here
  results.clear();

  case_info.label_size_before_canonical_repair =
      label_size_before_canonical_repair_599;
  case_info.label_size_after_canonical_repair =
      label_size_after_canonical_repair_599;
  case_info.canonical_repair_remove_label_ratio =
      (double)(label_size_before_canonical_repair_599 -
               label_size_after_canonical_repair_599) /
      label_size_before_canonical_repair_599;
}

void hop_constrained_two_hop_labels_generation(
    graph_v_of_v<int> &input_graph, hop_constrained_case_info &case_info) {

  //----------------------------------- step 1: initialization
  //-----------------------------------
  auto begin = std::chrono::high_resolution_clock::now();

  labal_size_599 = 0;
  begin_time_599 = std::chrono::high_resolution_clock::now();
  max_run_time_nanoseconds_599 = case_info.max_run_time_seconds * 1e9;
  max_labal_size_599 =
      case_info.max_bit_size / sizeof(hop_constrained_two_hop_label);

  int N = input_graph.size();
  L_temp_599.resize(N);
  if (N > max_N_ID_for_mtx_599) {
    cout << "N > max_N_ID_for_mtx_599!" << endl;
    exit(1);
  }

  int num_of_threads = case_info.thread_num;
  ThreadPool pool(num_of_threads);
  std::vector<std::future<int>> results;

  ideal_graph_599 = input_graph;
  global_use_2M_prune = case_info.use_2M_prune;

  auto end = std::chrono::high_resolution_clock::now();
  case_info.time_initialization =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
          .count() /
      1e9;

  //----------------------------------------------- step 2: generate labels
  //---------------------------------------------------------------
  begin = std::chrono::high_resolution_clock::now();

  global_upper_k = case_info.upper_k == 0 ? std::numeric_limits<int>::max()
                                          : case_info.upper_k;

  Temp_L_vk_599.resize(num_of_threads);
  dist_hop_599.resize(num_of_threads);
  Q_handle_priorities_599.resize(num_of_threads);
  Vh_599.resize(num_of_threads);
  hop_constrained_node_handle handle_x;
  for (int i = 0; i < num_of_threads; i++) {
    Temp_L_vk_599[i].resize(N);
    dist_hop_599[i].resize(N, {std::numeric_limits<int>::max(), 0});
    Q_handle_priorities_599[i].resize(N);
    for (int j = 0; j < N; j++) {
      Q_handle_priorities_599[i][j].resize(
          global_upper_k + 1, {handle_x, std::numeric_limits<int>::max()});
    }
    Vh_599[i].resize(global_upper_k + 2);
    Qid_599.push(i);
  }
  if (case_info.use_2023WWW_generation) {
    for (int v_k = 0; v_k < N; v_k++) {
      results.emplace_back(pool.enqueue([v_k] {
        _2023WWW_thread_function(v_k);
        return 1;
      }));
    }
  } else {
    for (int v_k = 0; v_k < N; v_k++) {
      results.emplace_back(pool.enqueue([v_k] {
        HSDL_thread_function(v_k);
        return 1;
      }));
    }
  }
  for (auto &&result : results)
    result.get();

  end = std::chrono::high_resolution_clock::now();
  case_info.time_generate_labels =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
          .count() /
      1e9; // s

  //----------------------------------------------- step 3:
  // sortL---------------------------------------------------------------
  begin = std::chrono::high_resolution_clock::now();

  case_info.L = hop_constrained_sortL(num_of_threads);

  end = std::chrono::high_resolution_clock::now();
  case_info.time_sortL =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
          .count() /
      1e9; // s

  //----------------------------------------------- step 4:
  // canonical_repair---------------------------------------------------------------
  begin = std::chrono::high_resolution_clock::now();

  // case_info.print_L();

  if (case_info.use_canonical_repair) {
    hop_constrained_clean_L(case_info, num_of_threads);
  }

  // case_info.print_L();

  end = std::chrono::high_resolution_clock::now();
  case_info.time_canonical_repair =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
          .count() /
      1e9; // s
  //---------------------------------------------------------------------------------------------------------------------------------------

  case_info.time_total = case_info.time_initialization +
                         case_info.time_generate_labels + case_info.time_sortL +
                         case_info.time_canonical_repair;

  hop_constrained_clear_global_values();
}

void generate_Group_kmeans(graph_v_of_v<int> &instance_graph,
                           hop_constrained_case_info &case_info, int group_num,
                           int hop_cst,
                           std::unordered_map<int, std::vector<int>> &groups) {
  int n = instance_graph.size();
  vector<int> centers(group_num);
  vector<int> labels(n, -1);

  // 1. 无重复的随机选择group_num个点作为初始的聚类中心
  for (int i = 0; i < group_num; ++i) {
    do {
      centers[i] = rand() % n;
    } while (find(centers.begin(), centers.end(), centers[i]) !=
             centers.begin() + i);
  }

  bool changed;
  while (changed) {
    changed = false;
    // 2.
    // 对于每个点，计算它到每个聚类中心的距离，将它划分到距离最近的聚类中心所在的类中
    for (int i = 0; i < n; ++i) {
      int nearest_center = -1;
      int min_distance = numeric_limits<int>::max();
      for (int j = 0; j < group_num; ++j) {
        int distance = hop_constrained_extract_distance(case_info.L, i,
                                                        centers[j], hop_cst);
        if (distance < min_distance) {
          nearest_center = j;
          min_distance = distance;
        }
      }
      if (labels[i] != nearest_center) {
        labels[i] = nearest_center;
        changed = true;
      }
    }

    // 3.
    // 重新计算聚类中心（这个步骤在图聚类中略有不同，因为我们不能简单地取平均值）
    // 这里我们留空，因为重新计算聚类中心在图中意义不大，通常保持选择的中心不变

  } // 如果在一轮迭代中聚类结果没有变化，则算法结束

  // 根据最终的labels数组构建groups输出
  for (int i = 0; i < n; ++i) {
    groups[labels[i]].push_back(i);
  }

  //对groups中的每个group进行排序，按点的度数从大到小排序
  for (auto it : groups) {
    sort(it.second.begin(), it.second.end(), [&](int a, int b) {
      return instance_graph[a].size() > instance_graph[b].size();
    });
  }
}

/*
void generate_Group_louvain(std::unordered_map<int, std::vector<int>> &groups) {
  //运行python脚本，生成社区划分结果
  std::string cmd = "python3 louvain_clustering.py";
  int result = std::system(cmd.data());

  cpp_redis::client client;

  // Connect to Redis
  if (!client.is_connected()) {
    client.connect("127.0.0.1", 6379,
                   [](const std::string &host, std::size_t port,
                      cpp_redis::client::connect_state status) {
                     if (status == cpp_redis::client::connect_state::ok) {
                       std::cout << "Connected to redis" << std::endl;
                     } else {
                       // std::cout << "Failed to connect to redis" <<
                       // std::endl;
                     }
                   });
  }
  if (!client.is_connected()) {
    std::cerr << "Cannot connect to redis" << std::endl;
    return;
  }

  // Read community detection results from Redis
  client.hgetall("partition_result", [&groups](const cpp_redis::reply &reply) {
    if (reply.is_array()) {
      const auto &replies = reply.as_array();
      for (size_t i = 0; i < replies.size(); i += 2) {
        int node = std::stoi(replies[i].as_string());
        int community = std::stoi(replies[i + 1].as_string());
        groups[community].push_back(node);
      }
    }
  });

  // Synchronously commit the outstanding operations and wait for them to
  // complete
  client.sync_commit();

  printf("Redis operation completed\n");
}
*/

void generate_Group_CT_cores(
    graph_v_of_v<int> &instance_graph,
    hop_constrained_case_info &case_info, //暂时用作最短距离查询
    int hop_cst, int group_num,
    std::unordered_map<int, std::vector<int>> &groups) {

  int N = instance_graph.size();
  vector<int> labels(N, -1);
  unordered_set<int> centers;
  // generate CT cores
  CT_case_info mm;
  mm.d = 10;
  mm.use_P2H_pruning = 1;
  mm.two_hop_info.use_2M_prune = 1;
  mm.two_hop_info.use_canonical_repair = 1;
  mm.thread_num = 10;

  CT_cores(instance_graph, mm);
  printf("CT_cores finished\n");
  for (int i = 0; i < N; i++) {
    if (mm.isIntree[i] == 0) {
      centers.insert(i);
    }
  }

  bool changed;
  while (changed) {
    changed = false;
    // 2.
    // 对于每个点，计算它到每个聚类中心的距离，将它划分到距离最近的聚类中心所在的类中
    for (int i = 0; i < N; ++i) {
      int nearest_center = -1;
      int min_distance = numeric_limits<int>::max();
      for (auto j : centers) {
        int distance =
            hop_constrained_extract_distance(case_info.L, i, j, hop_cst);
        if (distance < min_distance) {
          nearest_center = j;
          min_distance = distance;
        }
      }
      if (labels[i] != nearest_center) {
        labels[i] = nearest_center;
        changed = true;
      }
    }
  }

  // 根据最终的labels数组构建groups输出
  for (int i = 0; i < N; ++i) {
    groups[labels[i]].push_back(i);
  }
}

void unionLabel2L_temp_599() {
  for (int i = 0; i < unionedLabels.size(); i++) {
    for (auto &xx : unionedLabels[i]) {
      L_temp_599[i].push_back(xx);
    }
  }
}

void union_groupLabels_with_unionedLabels() {
  for (int i = 0; i < L_temp_599.size(); i++) {
    for (int j = 0; j < L_temp_599[i].size(); j++) {
      unionedLabels[i].insert(L_temp_599[i][j]);
    }
  }
}

void generate_one_group(graph_v_of_v<int> &input_graph,
                        hop_constrained_case_info &case_info,
                        vector<int> &group) {
  // 1. initialize
  labal_size_599 = 0;

  int N = input_graph.size();
  L_temp_599.resize(N);
  int num_of_threads = case_info.thread_num;
  ThreadPool pool(num_of_threads);
  std::vector<std::future<int>> results;
  global_use_2M_prune = case_info.use_2M_prune;

  // 2. generate labels

  global_upper_k = case_info.upper_k == 0 ? std::numeric_limits<int>::max()
                                          : case_info.upper_k;

  Temp_L_vk_599.resize(num_of_threads);
  dist_hop_599.resize(num_of_threads);
  Q_handle_priorities_599.resize(num_of_threads);
  Vh_599.resize(num_of_threads);
  hop_constrained_node_handle handle_x;

  for (int i = 0; i < num_of_threads; i++) {
    Temp_L_vk_599[i].resize(N);
    dist_hop_599[i].resize(N, {std::numeric_limits<int>::max(), 0});
    Q_handle_priorities_599[i].resize(N);
    for (int j = 0; j < N; j++) {
      Q_handle_priorities_599[i][j].resize(
          global_upper_k + 1, {handle_x, std::numeric_limits<int>::max()});
    }
    Vh_599[i].resize(global_upper_k + 2);
    Qid_599.push(i);
  }

  if (case_info.use_2023WWW_generation) {
    for (auto it : group) {
      results.emplace_back(pool.enqueue([it] {
        _2023WWW_thread_function(it);
        return 1;
      }));
    }
  } else {
    for (auto it : group) {
      results.emplace_back(pool.enqueue([it] {
        HSDL_thread_function_insert_to_union(it);
        return 1;
      }));
    }
  }
  for (auto &&result : results) {
    result.get();
  }
  results.clear();

  // 3. Union the labels of the group
  // union_groupLabels_with_unionedLabels();
  // 4. clean all global values
  hop_constrained_clear_global_values();
}

void Group_HBPLL_generation(graph_v_of_v<int> &input_graph,
                            hop_constrained_case_info &case_info,
                            std::unordered_map<int, std::vector<int>> &groups) {

  //----------------------------------- step 1: initialization
  //-----------------------------------
  auto begin = std::chrono::high_resolution_clock::now();

  labal_size_599 = 0;
  // begin_time_599 = std::chrono::high_resolution_clock::now();
  max_run_time_nanoseconds_599 = case_info.max_run_time_seconds * 1e9;
  max_labal_size_599 =
      case_info.max_bit_size / sizeof(hop_constrained_two_hop_label);

  int N = input_graph.size();
  L_temp_599.resize(N);
  unionedLabels.resize(N);
  if (N > max_N_ID_for_mtx_599) {
    cout << "N > max_N_ID_for_mtx_599!" << endl;
    exit(1);
  }

  int num_of_threads = case_info.thread_num;
  ThreadPool pool(num_of_threads);
  std::vector<std::future<int>> results;

  ideal_graph_599 = input_graph;
  global_use_2M_prune = case_info.use_2M_prune;

  auto end = std::chrono::high_resolution_clock::now();
  case_info.time_initialization =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
          .count() /
      1e9;

  //----------------------------------------------- step 2: generate labels
  //---------------------------------------------------------------
  for (auto it : groups) {
    generate_one_group(input_graph, case_info, it.second);
  }

  //----------------------------------------------- step 3:
  // sortL---------------------------------------------------------------
  begin = std::chrono::high_resolution_clock::now();
  L_temp_599.clear();
  L_temp_599.resize(N);
  unionLabel2L_temp_599();

  case_info.L = hop_constrained_sortL(num_of_threads);

  end = std::chrono::high_resolution_clock::now();
  case_info.time_sortL =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
          .count() /
      1e9; // s

  //----------------------------------------------- step 4:
  // canonical_repair---------------------------------------------------------------
  begin = std::chrono::high_resolution_clock::now();

  // case_info.print_L();

  if (case_info.use_canonical_repair) {
    // hop_constrained_clean_L(case_info, num_of_threads);
  }

  // case_info.print_L();

  end = std::chrono::high_resolution_clock::now();
  case_info.time_canonical_repair =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
          .count() /
      1e9; // s
  //---------------------------------------------------------------------------------------------------------------------------------------

  case_info.time_total = case_info.time_initialization +
                         case_info.time_generate_labels + case_info.time_sortL +
                         case_info.time_canonical_repair;

  hop_constrained_clear_global_values();
  clear_unionedLabels();
}