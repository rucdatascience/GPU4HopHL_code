#pragma once
#include <CT/CT_labels.h>
#include <PLL.h>
#include <climits>
#include <cmath>
#include <graph_v_of_v/graph_v_of_v.h>
#include <graph_v_of_v/graph_v_of_v_update_vertexIDs_by_degrees_large_to_small.h>
#include <queue>
#include <unordered_set>
#include <vector>

/*global values*/
graph_v_of_v<int> global_ideal_graph_CT,
    global_eliminated_edge_predecessors; // weight of
                                         // global_eliminated_edge_predecessors
                                         // is predecessor of a merged edge; (it
                                         // is slightly faster to use
                                         // graph_v_of_v<int> than to use map)

void clear_gloval_values_CT() {
  PLL_clear_global_values();
  global_ideal_graph_CT.clear();
  global_eliminated_edge_predecessors.clear();
}

void dfs(int &total, vector<int> &first_pos, int x, vector<vector<int>> &son,
         vector<int> &dfn) {
  total++;
  dfn[total] = x;
  first_pos[x] = total;
  int s_size = son[x].size();
  for (int i = 0; i < s_size; i++) {
    dfs(total, first_pos, son[x][i], son, dfn); // this is a recursive function
    total++;
    dfn[total] = x;
  }
}

void CT(graph_v_of_v<int> &input_graph, CT_case_info &case_info) {

  //--------------------------------- step 1: initialization
  //---------------------------
  auto begin1 = std::chrono::high_resolution_clock::now();

  auto &Bags = case_info.Bags;
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
  vector<vector<int>> bag_predecessors(N);
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

  auto end1 = std::chrono::high_resolution_clock::now();
  case_info.time_initialization =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - begin1)
          .count() /
      1e9;
  //------------------------------------------------------------------------------------------------------------------------------------

  //-------------------------------------------------- step 2: MDE-based tree
  // decomposition ------------------------------------------------------------
  auto begin2 = std::chrono::high_resolution_clock::now();

  /*MDE-based tree decomposition; generating bags*/
  int bound_lambda = N;
  Bags.resize(N);
  vector<int> node_order(N + 1); // merging ID to original ID

  ThreadPool pool(case_info.thread_num);
  std::vector<std::future<int>> results; // return typename: xxx

  for (int i = 1; i <= N; i++) {
    node_degree nd;
    while (1) {
      nd = q.top();
      q.pop();
      if (!isIntree[nd.vertex] &&
          global_ideal_graph_CT[nd.vertex].size() == nd.degree)
        break; // nd.vertex is the lowest degree vertex not in tree
    }
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
    for (int j = 0; j < v_adj_size; j++) {
      // if (adj_temp[j].second > TwoM_value) { // this causes errors, since
      // each bag in a tree contains same interfaces with different distances
      //	cout << "xx" << endl;
      //	continue;
      // }
      Bags[v_x].push_back(
          {adj_temp[j].first,
           adj_temp[j]
               .second}); // Bags[v_x] stores adj vertices and weights of v_x
      int pred = adj_temp[j].first;
      bag_predecessors[v_x].push_back(
          adj_temp[j].first); // bag_predecessors[v_x] stores predecessors (in
                              // merged graphs) for vertices in Bags[v_x]
    }

    /*add new edge*/
    for (int j = 0; j < v_adj_size; j++) {
      int adj_j = adj_temp[j].first;
      for (int k = j + 1; k < v_adj_size; k++) {
        int adj_k = adj_temp[k].first;
        int new_ec = adj_temp[j].second + adj_temp[k].second;

        // if (new_ec > TwoM_value) { // this causes errors
        //	cout << "xx" << endl;
        //	continue;
        // }

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
    vector<pair<int, int>>().swap(global_ideal_graph_CT[v_x]);
  }

  auto end2 = std::chrono::high_resolution_clock::now();
  case_info.time_tree_decomposition =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - begin2)
          .count() /
      1e9;
  //--------------------------------------------------------------------------------------------------------

  //---------------------------------------------- step 3: generate CT-tree
  // indexs ------------------------------------------------

  auto begin3 = std::chrono::high_resolution_clock::now();

  // generate CT-tree indexs
  vector<vector<two_hop_label>> L1(
      N); // Labels for CT-tree index merge later, otherwise may increase query
          // time on PLL
  vector<int> fa(N);
  root.resize(N);
  vector<int> temp_dis(N);
  vector<int> temp_parent(N);
  vector<int> order_mapping(N + 1); // original ID to merging ID
  for (int i = 1; i <= bound_lambda; i++)
    order_mapping[node_order[i]] = i;
  vector<bool> popped_isIncore_node(N, 0);

  for (int i = bound_lambda + 1; i <= N;
       i++) // take advantage of the priority queue
  {
    struct node_degree nd;
    while (1) {
      nd = q.top();
      q.pop();
      if (!isIntree[nd.vertex] && !popped_isIncore_node[nd.vertex] &&
          global_ideal_graph_CT[nd.vertex].size() == nd.degree)
        break;
    }
    node_order[i] = nd.vertex; // merging ID to original ID
    popped_isIncore_node[nd.vertex] = 1;
    order_mapping[nd.vertex] = i; // original ID to merging ID
  }

  dep.resize(N);
  vector<int> islabel(N, 0);
  first_pos.resize(N);
  vector<vector<int>> son(N);
  vector<int> index_node(N);

  vector<int> isneighour(N);
  int neighbournum = 0;
  vector<int> T_temp_n(N);
  vector<int> T_temp_p(N);

  int labelnum = 0;

  for (int i = bound_lambda; i >= 1; i--) {
    int v_x = node_order[i]; //  node_order(N + 1); // merging ID to original ID

    fa[v_x] = INT_MAX; // merging ID
    int v_adj_size = Bags[v_x].size();
    for (int j = 0; j < v_adj_size; j++)
      if (order_mapping[Bags[v_x][j].first] < fa[v_x])
        fa[v_x] = order_mapping[Bags[v_x][j].first]; // renew fa[v_x] to be the
                                                     // smallest merging_ID (the
                                                     // lowest ancestor bag)

    if (fa[v_x] > bound_lambda ||
        v_adj_size == 0) // a root in the forest (a bag with interfaces)
    {
      root[v_x] = v_x; // original ID
      fa[v_x] = -1;
      dep[v_x] = 0; // multi_fa[v_x][0] = -1;

      /*below is the tree indexes of a root;
      Lines 24-25 of CT in 2020 paper;

      tree indexes of a root only contain interfaces, but no ancetors*/
      two_hop_label xx;
      for (int j = 0; j < v_adj_size;
           j++) // (bound_lambda - 1) - local distance to the interface
      {
        xx.vertex = Bags[v_x][j].first;
        xx.distance = Bags[v_x][j].second;
        xx.parent_vertex = bag_predecessors[v_x][j];
        L1[v_x].push_back(xx); // interpfact parts of tree indexes of root v_x
      }
    } else // a non-root in the forest
    {
      fa[v_x] = node_order[fa[v_x]]; //  node_order(N + 1); // merging ID to
                                     //  original ID
      root[v_x] = root[fa[v_x]]; // int i = bound_lambda; i >= 1; i--, already
                                 // has the right root[fa[v_x]];  root[v_x] =
                                 // root[fa[v_x]], from high to low to get roots
      dep[v_x] = dep[fa[v_x]] + 1; // multi_fa[v_x][0] = fa[v_x]; for LCA
      son[fa[v_x]].push_back(v_x); // for LCA

      int index_node_num = 0;
      labelnum++; // to ensure the order, we can not use "push_back"

      int root_adj_size =
          Bags[root[v_x]].size(); // iterfaces, already added above

      /*add interface node*/
      for (int j = 0; j < root_adj_size;
           j++) // put the labels to interface in the beginnig position
      {         // add interface
        islabel[Bags[root[v_x]][j].first] =
            labelnum; // labelnum means that a hub vertex is added into bag v_x
        index_node_num++;
        index_node[index_node_num] = Bags[root[v_x]][j].first;
        temp_dis[Bags[root[v_x]][j].first] =
            std::numeric_limits<int>::max(); // initial dis
      }

      /*add ancestor node: representation nodes of all ancetor bags*/
      int v_y = v_x;
      while (fa[v_y] != -1) {
        // add ancestor
        if (islabel[fa[v_y]] != labelnum) // fa[v_y] is not in bag v_x yet
        {
          index_node_num++;
          index_node[index_node_num] = fa[v_y];
          islabel[fa[v_y]] = labelnum; // add fa[v_y] into bag v_x
          temp_dis[fa[v_y]] = std::numeric_limits<int>::max(); // initial dis
        }
        v_y = fa[v_y];
      }

      /*corresponds to Line 30 of CT: the first value after min: delta_u =
       * Bags[v_x][j].second*/
      for (int j = 0; j < v_adj_size; j++) {
        // add neighbours
        if (islabel[Bags[v_x][j].first] != labelnum) {
          islabel[Bags[v_x][j].first] = labelnum;
          index_node.push_back(Bags[v_x][j].first);
          temp_dis[Bags[v_x][j].first] = Bags[v_x][j].second;
          temp_parent[Bags[v_x][j].first] = bag_predecessors[v_x][j];
        } else {
          temp_dis[Bags[v_x][j].first] = Bags[v_x][j].second;
          temp_parent[Bags[v_x][j].first] = bag_predecessors[v_x][j];
        }
      }
      // query (bound_lambda - 1)_local_distance to ancestor or the interface
      // through neighbours

      /*corresponds to Line 30 of CT: the second min value: dis_vj +
       * L1[vj][k].distance*/
      for (int j = 0; j < v_adj_size; j++)
        if (isIntree[Bags[v_x][j].first]) // isneighour and isintree --> can be
                                          // used as an intermediate node to
                                          // update labels
        {
          int dis_vj = Bags[v_x][j].second;
          int vj = Bags[v_x][j].first;
          int Lj_size = L1[vj].size();
          for (int k = 0; k < Lj_size;
               k++) // update the (bound_lambda-1)_local_distance
          {
            if (islabel[L1[vj][k].vertex] == labelnum &&
                dis_vj + L1[vj][k].distance < temp_dis[L1[vj][k].vertex]) {
              temp_dis[L1[vj][k].vertex] = dis_vj + L1[vj][k].distance;
              temp_parent[L1[vj][k].vertex] = bag_predecessors[v_x][j];
            }
          }
        }

      /*add correct indexes of Lines 29-30 of CT into L1, and possibly wrong
       * distances for Lines 31-32 into L1*/
      // add labels to L1
      L1[v_x].resize(index_node_num);
      for (int j = 1; j <= index_node_num; j++) {
        two_hop_label xx;
        xx.vertex = index_node[j];
        xx.distance = temp_dis[index_node[j]];
        xx.parent_vertex = temp_parent[index_node[j]];
        L1[v_x][j - 1] = xx;
      }

      /*Lines 31-32 of CT; update possibly wrong distances for Lines 31-32 in
       * L1*/
      // update conversely
      neighbournum++;
      for (int j = 0; j < v_adj_size; j++) {
        isneighour[Bags[v_x][j].first] = neighbournum;
        T_temp_n[Bags[v_x][j].first] = Bags[v_x][j].second;
        T_temp_p[Bags[v_x][j].first] = bag_predecessors[v_x][j];
      }
      for (int j = 1; j <= index_node_num; j++) {
        int vj = index_node[j];
        int Lj_size = L1[vj].size();
        for (int k = 0; k < Lj_size; k++) {
          int vk = L1[vj][k].vertex;
          if ((isneighour[vk] == neighbournum) &&
              (T_temp_n[vk] + L1[vj][k].distance < temp_dis[vj])) {
            temp_dis[vj] = T_temp_n[vk] + L1[vj][k].distance;
            temp_parent[vj] = T_temp_p[vk];
          }
        }
      }
      for (int j = 1; j <= index_node_num; j++)
        if (temp_dis[index_node[j]] < L1[v_x][j - 1].distance) {
          L1[v_x][j - 1].distance = temp_dis[index_node[j]];
          L1[v_x][j - 1].parent_vertex = temp_parent[index_node[j]];
        }
    }
  }

  /*add distance-0 labels to tree nodes; this is needed in querying functions*/
  two_hop_label node;
  for (int i = 0; i < N; i++) {
    if (isIntree[i]) {
      node.vertex = i;
      node.distance = 0;
      node.parent_vertex = i;
      L1[i].push_back(node);
    }
  }

  auto end3 = std::chrono::high_resolution_clock::now();
  case_info.time_tree_indexs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - begin3)
          .count() /
      1e9;
  //-------------------------------------------------------------------------------------------------------

  //---------------------------------------------------P2H
  // begin---------------------------------------------------
  auto begin_p2h = std::chrono::high_resolution_clock::now();

  if (case_info.use_P2H_pruning) {
    int bag_size_before = 0;
    for (auto &it : Bags) {
      bag_size_before += it.size();
    }
    std::vector<std::vector<pair<int, int>>> ZBags(
        N); // bag nodes of decomposied tree

    for (int i = 0; i < N; i++) {
      if (isIntree[i] == 0) // core has no bag
        continue;
      if (fa[i] == -1) {
        ZBags[i] = Bags[i];
        continue;
      }

      int child_num = son[i].size();

      if (child_num < 2) {
        continue;
      } else if (child_num == 2) {
        int leftson = son[i][0];
        int rightson = son[i][1];
        if (Bags[leftson].size() < Bags[rightson].size()) {
          for (auto &w : Bags[leftson]) {
            if (w.first != i)
              ZBags[i].push_back(w);
          }
        } else {
          for (auto &w : Bags[rightson]) {
            if (w.first != i)
              ZBags[i].push_back(w);
          }
        }
      } else {
        vector<int> u_count(N), in_bag(N);

        // each u in bags[i]
        for (auto &u : Bags[i]) {
          in_bag[u.first] = 1;
        }
        // u in sonbags how many times?
        for (auto sonnumber : son[i]) {
          for (auto w : Bags[sonnumber]) {
            if (in_bag[w.first] == 1) {
              u_count[w.first]++;
            }
          }
        }

        vector<int> A(son[i].size());
        for (int j = 0; j < son[i].size(); j++) {
          for (auto w : Bags[son[i][j]]) {
            if (u_count[w.first] == 1)
              A[j]++;
          }
        }

        int index = max_element(A.begin(), A.end()) - A.begin();
        for (auto u : Bags[son[i][index]]) {
          u_count[u.first]--;
        }

        for (auto u : Bags[i]) {
          if (u_count[u.first] > 0) {
            ZBags[i].push_back(u);
          }
        }
      }
    }

    Bags.swap(ZBags);

    int bag_size_after = 0;
    for (auto &it : Bags) {
      bag_size_after += it.size();
    }

    case_info.bag_size_before_P2H = bag_size_before;
    case_info.bag_size_after_P2H = bag_size_after;
  }

  auto end_p2h = std::chrono::high_resolution_clock::now();
  case_info.time_P2H_pruning =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end_p2h - begin_p2h)
          .count() /
      1e9;
  //------------------------------------------------------------------------------------------------------

  //------------------------------------------------ step 4: LCA
  //--------------------------------------------------------------
  auto begin4 = std::chrono::high_resolution_clock::now();

  /* LCA code; already get the root, the father and the depth, here is the
   * preprocessing of querying LCA */
  int total = 0;
  vector<int> dfn(2 * N + 5);
  for (int i = 1; i <= bound_lambda; i++) {
    int v_x = node_order[i];
    if (root[v_x] == v_x)
      dfs(total, first_pos, v_x, son, dfn);
  }

  if (total > 0) {
    int multi_step = ceil(log(total) / log(2)) + 2;

    tree_st.resize(total + 5);
    tree_st_r.resize(total + 5);
    for (int i = 1; i <= total; i++) {
      tree_st[i].resize(multi_step + 2);
      tree_st_r[i].resize(multi_step + 2);
    }

    vector<int> pow_2(multi_step);

    pow_2[0] = 1;
    for (int i = 1; i < multi_step; i++)
      pow_2[i] = pow_2[i - 1] << 1;

    for (int i = 1; i <= total; i++) {
      tree_st[i][0] = dfn[i];
      tree_st_r[i][0] = dfn[i];
    }

    for (int j = 1; j < multi_step; j++)
      for (int i = 1; i <= total; i++) {
        int k = i + pow_2[j - 1];
        if (k > total || dep[tree_st[i][j - 1]] <= dep[tree_st[k][j - 1]])
          tree_st[i][j] = tree_st[i][j - 1];
        else
          tree_st[i][j] = tree_st[k][j - 1];
        k = i - pow_2[j - 1];
        if (k <= 0 || dep[tree_st_r[i][j - 1]] <= dep[tree_st_r[k][j - 1]])
          tree_st_r[i][j] = tree_st_r[i][j - 1];
        else
          tree_st_r[i][j] = tree_st_r[k][j - 1];
      }
  }

  lg.resize(total + 1);
  for (int i = 1; i <= total; i++)
    lg[i] = floor(log(i) / log(2));

  /*clear variables not used below*/
  vector<vector<int>>().swap(bag_predecessors);
  priority_queue<node_degree>().swap(q);
  vector<int>().swap(node_order);
  vector<int>().swap(fa);
  vector<int>().swap(temp_dis);
  vector<int>().swap(temp_parent);
  vector<int>().swap(order_mapping);
  vector<int>().swap(islabel);
  vector<vector<int>>().swap(son);
  vector<int>().swap(index_node);
  vector<int>().swap(isneighour);
  vector<int>().swap(T_temp_n);
  vector<int>().swap(T_temp_p);
  vector<int>().swap(dfn);

  auto end4 = std::chrono::high_resolution_clock::now();
  case_info.time_lca =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end4 - begin4)
          .count() /
      1e9;
  //--------------------------------------------------------------------------------------------------------------------------

  //----------------------------------------------- step 5: 2-hop labeling
  //-------------------------------------------
  auto begin5 = std::chrono::high_resolution_clock::now();

  /*update limits*/
  double to_date_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end4 - begin1)
          .count() /
      1e9;
  case_info.two_hop_info.max_run_time_seconds =
      case_info.max_run_time_seconds - to_date_time;
  case_info.two_hop_info.thread_num = case_info.thread_num;
  if (case_info.two_hop_info.max_run_time_seconds < 0) {
    throw reach_limit_error_string_time;
  }
  long long int to_date_bit_size = case_info.compute_label_bit_size();
  case_info.two_hop_info.max_labal_bit_size =
      case_info.max_bit_size - to_date_bit_size;
  if (case_info.two_hop_info.max_labal_bit_size < 0) {
    throw reach_limit_error_string_MB;
  }

  /* construct 2-hop labels on core */
  if (case_info.d == 0) {
    PLL(global_ideal_graph_CT, case_info.two_hop_info);
  } else {
    vector<int> vertexID_new_to_old;
    auto new_instance_graph =
        graph_v_of_v_update_vertexIDs_by_degrees_large_to_small(
            global_ideal_graph_CT, vertexID_new_to_old); // sort vertices
    two_hop_case_info new_two_hop_case_info = case_info.two_hop_info;
    PLL(new_instance_graph, new_two_hop_case_info);

    auto &old_L = case_info.two_hop_info.L;
    old_L.resize(N);
    auto &L = new_two_hop_case_info.L;
    for (int v_k = 0; v_k < N;
         v_k++) { // it seems that parallel computing is slower
      results.emplace_back(
          pool.enqueue([&old_L, &L, v_k,
                        &vertexID_new_to_old] { // pass const type value j to
                                                // thread; [] can be empty
            for (auto &xx : L[v_k]) {
              xx.vertex = vertexID_new_to_old[xx.vertex];
              xx.parent_vertex = vertexID_new_to_old[xx.parent_vertex];
            }
            sort(L[v_k].begin(), L[v_k].end(),
                 compare_two_hop_label_small_to_large);
            old_L[vertexID_new_to_old[v_k]] = L[v_k];
            vector<two_hop_label>().swap(L[v_k]);

            return 1; // return to results; the return type must be the same
                      // with results
          }));
    }
    for (auto &&result : results)
      result.get(); // all threads finish here
    results.clear();

    /*info*/
    case_info.two_hop_info.label_size_before_canonical_repair =
        new_two_hop_case_info.label_size_before_canonical_repair;
    case_info.two_hop_info.label_size_after_canonical_repair =
        new_two_hop_case_info.label_size_after_canonical_repair;
    case_info.two_hop_info.canonical_repair_remove_label_ratio =
        new_two_hop_case_info.canonical_repair_remove_label_ratio;
    case_info.two_hop_info.time_initialization =
        new_two_hop_case_info.time_initialization;
    case_info.two_hop_info.time_generate_labels =
        new_two_hop_case_info.time_generate_labels;
    case_info.two_hop_info.time_sortL = new_two_hop_case_info.time_sortL;
    case_info.two_hop_info.time_canonical_repair =
        new_two_hop_case_info.time_canonical_repair;
    case_info.two_hop_info.time_total = new_two_hop_case_info.time_total;
    case_info.two_hop_info.max_labal_bit_size =
        new_two_hop_case_info.max_labal_bit_size;
    case_info.two_hop_info.max_run_time_seconds =
        new_two_hop_case_info.max_run_time_seconds;
  }

  auto end5 = std::chrono::high_resolution_clock::now();
  case_info.time_core_indexs =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end5 - begin5)
          .count() /
      1e9;
  //--------------------------------------------------------------------------------------------------------------------

  //-------------------------------------------------- step 6: postprocessing
  //-------------------------------------------------------------------
  auto begin6 = std::chrono::high_resolution_clock::now();

  /* merge tree_index: L1 into case_info.two_hop_info.L */
  for (int v_k = 0; v_k < N; v_k++) {
    if (L1[v_k].size() > 0) { // direcly removing >=2M tree labels here causes
                              // errors, since Bags should also be changed?
      vector<two_hop_label>(L1[v_k]).swap(L1[v_k]);
      case_info.two_hop_info.L[v_k] = L1[v_k];
      vector<two_hop_label>().swap(L1[v_k]);
    }
  }

  /*update predecessors in tree and core*/
  auto &L = case_info.two_hop_info.L;
  for (int u = 0; u < N; u++) {
    results.emplace_back(pool.enqueue([&L, u] {
      for (auto &xx : L[u]) {
        while (1) {
          auto it = global_eliminated_edge_predecessors.edge_weight(
              u, xx.parent_vertex);
          if (it == std::numeric_limits<int>::max()) {
            break;
          }
          xx.parent_vertex = it;
        }
        /*it is too slow to use CT_extract_distance to update predecessors in
         * large graphs*/
      }
      return 1;
    }));
  }
  for (auto &&result : results) {
    result.get();
  }
  results.clear();

  auto end6 = std::chrono::high_resolution_clock::now();
  case_info.time_post =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end6 - begin6)
          .count() /
      1e9;
  //---------------------------------------------------------------------------------------------------------------------------------

  clear_gloval_values_CT();

  case_info.time_total =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end6 - begin1)
          .count() /
      1e9;
}

void CT_cores(graph_v_of_v<int> &input_graph, CT_case_info &case_info) {
  auto begin1 = std::chrono::high_resolution_clock::now();

  auto &Bags = case_info.Bags;
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
  vector<vector<int>> bag_predecessors(N);
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

  auto end1 = std::chrono::high_resolution_clock::now();
  case_info.time_initialization =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - begin1)
          .count() /
      1e9;
  //------------------------------------------------------------------------------------------------------------------------------------

  //-------------------------------------------------- step 2: MDE-based tree
  // decomposition ------------------------------------------------------------
  auto begin2 = std::chrono::high_resolution_clock::now();

  /*MDE-based tree decomposition; generating bags*/
  int bound_lambda = N;
  Bags.resize(N);
  vector<int> node_order(N + 1); // merging ID to original ID

  ThreadPool pool(case_info.thread_num);
  std::vector<std::future<int>> results; // return typename: xxx

  for (int i = 1; i <= N; i++) {
    node_degree nd;
    while (1) {
      nd = q.top();
      q.pop();
      if (!isIntree[nd.vertex] &&
          global_ideal_graph_CT[nd.vertex].size() == nd.degree)
        break; // nd.vertex is the lowest degree vertex not in tree
    }
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
    for (int j = 0; j < v_adj_size; j++) {
      // if (adj_temp[j].second > TwoM_value) { // this causes errors, since
      // each bag in a tree contains same interfaces with different distances
      //	cout << "xx" << endl;
      //	continue;
      // }
      Bags[v_x].push_back(
          {adj_temp[j].first,
           adj_temp[j]
               .second}); // Bags[v_x] stores adj vertices and weights of v_x
      int pred = adj_temp[j].first;
      bag_predecessors[v_x].push_back(
          adj_temp[j].first); // bag_predecessors[v_x] stores predecessors (in
                              // merged graphs) for vertices in Bags[v_x]
    }

    /*add new edge*/
    for (int j = 0; j < v_adj_size; j++) {
      int adj_j = adj_temp[j].first;
      for (int k = j + 1; k < v_adj_size; k++) {
        int adj_k = adj_temp[k].first;
        int new_ec = adj_temp[j].second + adj_temp[k].second;

        // if (new_ec > TwoM_value) { // this causes errors
        //	cout << "xx" << endl;
        //	continue;
        // }

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
    vector<pair<int, int>>().swap(global_ideal_graph_CT[v_x]);
  }

  auto end2 = std::chrono::high_resolution_clock::now();
  case_info.time_tree_decomposition =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - begin2)
          .count() /
      1e9;
}