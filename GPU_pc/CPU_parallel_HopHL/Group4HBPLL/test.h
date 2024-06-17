#ifndef TEST_H
#define TEST_H
#pragma once

/*the following codes are for testing

---------------------------------------------------
a cpp file (try.cpp) for running the following test code:
----------------------------------------

#include <fstream>
#include <iostream>
using namespace std;

// header files in the Boost library: https://www.boost.org/
#include <boost/random.hpp>
boost::random::mt19937 boost_random_time_seed{
static_cast<std::uint32_t>(std::time(0)) };

#include <build_in_progress/HL/Group4HBPLL/test.h>


int main()
{
        test_PLL();
}

------------------------------------------------------------------------------------------
Commends for running the above cpp file on Linux:

g++ -std=c++17 -I/home/boost_1_75_0 -I/root/rucgraph try.cpp -lpthread -Ofast -o
A
./A
rm A

(optional to put the above commends in run.sh, and then use the comment: sh
run.sh)


*/

#include <CT/CT.h>
//#include <build_in_progress/HL/Group4HBPLL/GST.h>
#include <PLL.h>
#include <graph_v_of_v/graph_v_of_v.h>
#include <graph_v_of_v/graph_v_of_v_generate_random_graph.h>
#include <graph_v_of_v/graph_v_of_v_hop_constrained_shortest_distance.h>
#include <graph_v_of_v/graph_v_of_v_shortest_paths.h>
#include <graph_v_of_v/graph_v_of_v_update_vertexIDs_by_degrees_large_to_small.h>
#include <hop_constrained_two_hop_labels_generation.h>
#include <text_mining/print_items.h>
#include <two_hop_labels.h>

#include <unordered_map>
#include <unordered_set>

#define DATASET_PATH "../../data/email-Enron2.txt"

boost::random::mt19937 boost_random_time_seed{
    static_cast<std::uint32_t>(std::time(0))};

inline void add_vertex_groups(graph_v_of_v<int> &instance_graph,
                              int group_num) {

  double dummy_edge_probability = 0.2;
  boost::random::uniform_int_distribution<> dist{static_cast<int>(1),
                                                 static_cast<int>(100)};

  int N = instance_graph.size();

  instance_graph.ADJs.resize(N + group_num);
  for (int i = N; i < N + group_num; i++) {
    for (int j = 0; j < N; j++) {
      if ((double)dist(boost_random_time_seed) / 100 < dummy_edge_probability) {
        instance_graph.add_edge(i, j, 1e6); // add a dummy edge
      }
    }
  }
}

void hop_constrained_check_correctness(hop_constrained_case_info &case_info,
                                       graph_v_of_v<int> &instance_graph,
                                       int iteration_source_times,
                                       int iteration_terminal_times,
                                       int upper_k) {

  boost::random::uniform_int_distribution<> vertex_range{
      static_cast<int>(0), static_cast<int>(instance_graph.size() - 1)};
  boost::random::uniform_int_distribution<> hop_range{
      static_cast<int>(1), static_cast<int>(upper_k)};

  for (int yy = 0; yy < iteration_source_times; yy++) {
    int source = vertex_range(boost_random_time_seed);
    std::vector<int> distances(instance_graph.size());

    int hop_cst = hop_range(boost_random_time_seed);

    graph_v_of_v_hop_constrained_shortest_distance(instance_graph, source,
                                                   hop_cst, distances);

    for (int xx = 0; xx < iteration_terminal_times; xx++) {
      int terminal = vertex_range(boost_random_time_seed);

      int query_dis = hop_constrained_extract_distance(case_info.L, source,
                                                       terminal, hop_cst);

      if (abs(query_dis - distances[terminal]) > 1e-4 &&
          (query_dis < TwoM_value || distances[terminal] < TwoM_value)) {
        instance_graph.print();
        case_info.print_L();
        cout << "source = " << source << endl;
        cout << "terminal = " << terminal << endl;
        cout << "hop_cst = " << hop_cst << endl;
        cout << "query_dis = " << query_dis << endl;
        cout << "distances[terminal] = " << distances[terminal] << endl;
        cout << "abs(dis - distances[terminal]) > 1e-5!" << endl;
        getchar();
      }

      vector<pair<int, int>> path = hop_constrained_extract_shortest_path(
          case_info.L, source, terminal, hop_cst);
      int path_dis = 0;
      for (auto xx : path) {
        path_dis += instance_graph.edge_weight(xx.first, xx.second);
      }
      if (abs(query_dis - path_dis) > 1e-4 &&
          (query_dis < TwoM_value || distances[terminal] < TwoM_value)) {
        instance_graph.print();
        case_info.print_L();
        cout << "source = " << source << endl;
        cout << "terminal = " << terminal << endl;
        cout << "hop_cst = " << hop_cst << endl;
        std::cout << "print_vector_pair_int:" << std::endl;
        for (int i = 0; i < path.size(); i++) {
          std::cout << "item: |" << path[i].first << "," << path[i].second
                    << "|" << std::endl;
        }
        cout << "query_dis = " << query_dis << endl;
        cout << "path_dis = " << path_dis << endl;
        cout << "abs(dis - path_dis) > 1e-5!" << endl;
        getchar();
      }
    }
  }
}

void test_HSDL() {

  /* problem parameters */
  int iteration_graph_times = 1e2, iteration_source_times = 10,
      iteration_terminal_times = 10;
  int V = 100, E = 500, group_num = 100;
  int ec_min = 1, ec_max = 10;

  bool generate_new_graph = 1;

  /* hop bounded info */
  hop_constrained_case_info mm;
  mm.upper_k = 5;
  mm.use_2M_prune = 1;
  mm.use_2023WWW_generation = 0;
  mm.use_canonical_repair = 1;
  mm.max_run_time_seconds = 10;
  mm.thread_num = 10;

  /* result info */
  double avg_index_time = 0;
  double avg_time_initialization = 0, avg_time_generate_labels = 0,
         avg_time_sortL = 0, avg_time_canonical_repair = 0;
  double avg_canonical_repair_remove_label_ratio = 0;

  /* iteration */
  for (int i = 0; i < iteration_graph_times; i++) {
    cout << ">>>iteration_graph_times: " << i << endl;

    graph_v_of_v<int> instance_graph;
    if (generate_new_graph == 1) {
      instance_graph = graph_v_of_v_generate_random_graph<int>(
          V, E, ec_min, ec_max, 1, boost_random_time_seed);
      /*add vertex groups*/
      if (group_num > 0) {
        add_vertex_groups(instance_graph, group_num);
      }
      instance_graph = graph_v_of_v_update_vertexIDs_by_degrees_large_to_small(
          instance_graph); // sort vertices
      instance_graph.txt_save("simple_iterative_tests.txt");
    } else {
      instance_graph.txt_read("simple_iterative_tests.txt");
    }

    // instance_graph.print();

    auto begin = std::chrono::high_resolution_clock::now();
    try {
      hop_constrained_two_hop_labels_generation(instance_graph, mm);
    } catch (string s) {
      cout << s << endl;
      hop_constrained_clear_global_values();
      continue;
    }
    auto end = std::chrono::high_resolution_clock::now();
    double runningtime =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
            .count() /
        1e9; // s
    avg_index_time = avg_index_time + runningtime / iteration_graph_times;
    avg_time_initialization += mm.time_initialization / iteration_graph_times;
    avg_time_generate_labels += mm.time_generate_labels / iteration_graph_times;
    avg_time_sortL += mm.time_sortL / iteration_graph_times;
    avg_time_canonical_repair +=
        mm.time_canonical_repair / iteration_graph_times;
    avg_canonical_repair_remove_label_ratio +=
        mm.canonical_repair_remove_label_ratio / iteration_graph_times;

    // if (mm.canonical_repair_remove_label_ratio > 0) {
    //	cout << "ss" << endl;
    //	getchar();
    // }

    hop_constrained_check_correctness(mm, instance_graph,
                                      iteration_source_times,
                                      iteration_terminal_times, mm.upper_k);

    mm.clear_labels();
  }

  cout << "avg_canonical_repair_remove_label_ratio: "
       << avg_canonical_repair_remove_label_ratio << endl;
  cout << "avg_index_time: " << avg_index_time << "s" << endl;
  cout << "	 avg_time_initialization: " << avg_time_initialization << endl;
  cout << "	 avg_time_generate_labels: " << avg_time_generate_labels
       << endl;
  cout << "	 avg_time_sortL: " << avg_time_sortL << endl;
  cout << "	 avg_time_canonical_repair: " << avg_time_canonical_repair
       << endl;
}

void test_Group4HBPLL(int TEST_ALGO = 1) {
  /* problem parameters */
  int iteration_graph_times = 1, iteration_source_times = 10,
      iteration_terminal_times = 10;
  int V = 1000, E = 3000, group_num = 100;
  int ec_min = 1, ec_max = 10;

  bool generate_new_graph = 1;

  graph_v_of_v<int> instance_graph;

  instance_graph.txt_read(DATASET_PATH);
  double avg_runningtime = 0.0;
  double avg_st_label_size = 0.0;
  double avg_st_max_sublabel_size = 0.0;
  double avg_st_min_sublabel_size = 0.0;
  double avg_group_generation_time = 0.0;
  double avg_group_HBPLL_generation_time = 0.0;
  double avg_group_HBPLL_total_time = 0.0;
  double avg_group_HBPLL_label_size = 0.0;
  double avg_group_HBPLL_max_sublabel_size = 0.0;
  double avg_group_HBPLL_min_sublabel_size = 0.0;

  hop_constrained_case_info mm, mm_standard;
  mm.upper_k = 10;
  mm.use_2M_prune = 1;
  mm.use_2023WWW_generation = 0;
  mm.use_canonical_repair = 1;
  // mm.max_run_time_seconds = 10;
  mm.thread_num = 96;
  mm.max_run_time_seconds = 1e9;

  /*生成标准label，得到 label size*/
  mm_standard.upper_k = 10;
  mm_standard.use_2M_prune = 1;
  mm_standard.use_2023WWW_generation = 0;
  mm_standard.use_canonical_repair = 1;
  // mm_standard.max_run_time_seconds = 10;
  mm_standard.thread_num = 96;
  mm_standard.max_run_time_seconds = 1e9;

  printf("construct standard 2-hop-labels\n");
  auto begin = std::chrono::high_resolution_clock::now();
  try {
    hop_constrained_two_hop_labels_generation(instance_graph, mm_standard);
  } catch (string s) {
    cout << s << endl;
    hop_constrained_clear_global_values();
  }
  auto end = std::chrono::high_resolution_clock::now();
  printf("construct standard 2-hop-labels finished\n");
  double runningtime =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
          .count() /
      1e9; // s
           // iteration

  avg_runningtime += runningtime;

  int st_label_size = 0;
  int st_max_sublabel_size = 0;
  int st_min_sublabel_size = std::numeric_limits<int>::max();

  for (auto it = mm_standard.L.begin(); it != mm_standard.L.end(); it++) {
    st_label_size = st_label_size + (*it).size();
    st_max_sublabel_size = max(st_max_sublabel_size, (int)(*it).size());
    st_min_sublabel_size = min(st_min_sublabel_size, (int)(*it).size());
  }

  avg_st_label_size += st_label_size;
  avg_st_max_sublabel_size += st_max_sublabel_size;
  avg_st_min_sublabel_size += st_min_sublabel_size;

  /* iteration */

  printf(">>>algo: %d\n", TEST_ALGO);
  // 生成group
  printf("generate groups\n");
  auto begin2 = std::chrono::high_resolution_clock::now();
  std::unordered_map<int, std::vector<int>> groups;

  if (TEST_ALGO == 1) {
    generate_Group_kmeans(instance_graph, mm_standard, group_num,
                          mm_standard.upper_k, groups);
  }
  if (TEST_ALGO == 2 && 0) {

    // generate_Group_louvain(groups);
    // 暂不支持
  }
  if (TEST_ALGO == 3) {
    generate_Group_CT_cores(instance_graph, mm_standard, mm_standard.upper_k,
                            group_num, groups);
  }
  printf("start sorting\n");
  // sort groups by degree
  for (auto it : groups) {
    sort(it.second.begin(), it.second.end(), [&](int a, int b) {
      return instance_graph[a].size() > instance_graph[b].size();
    });
  }
  printf("generate groups finished\n");
  auto end2 = std::chrono::high_resolution_clock::now();

  //测试按group并行
  try {
    Group_HBPLL_generation(instance_graph, mm, groups);
  } catch (string s) {
    cout << s << endl;
    hop_constrained_clear_global_values();
  }
  auto end3 = std::chrono::high_resolution_clock::now();

  double group_generation_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - begin2)
          .count() /
      1e9; // s
  double group_HBPLL_generation_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - end2)
          .count() /
      1e9; // s
  double group_HBPLL_total_time =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - begin2)
          .count() /
      1e9; // s

  int group_HBPLL_label_size = 0;
  int group_HBPLL_max_sublabel_size = 0;
  int group_HBPLL_min_sublabel_size = std::numeric_limits<int>::max();

  for (auto it = mm.L.begin(); it != mm.L.end(); it++) {
    group_HBPLL_label_size = group_HBPLL_label_size + (*it).size();
    group_HBPLL_max_sublabel_size =
        max(group_HBPLL_max_sublabel_size, (int)(*it).size());
    group_HBPLL_min_sublabel_size =
        min(group_HBPLL_min_sublabel_size, (int)(*it).size());
  }

  avg_group_generation_time = group_generation_time;
  avg_group_HBPLL_generation_time = group_HBPLL_generation_time;
  avg_group_HBPLL_total_time = group_HBPLL_total_time;

  avg_group_HBPLL_label_size = group_HBPLL_label_size;
  avg_group_HBPLL_max_sublabel_size = group_HBPLL_max_sublabel_size;
  avg_group_HBPLL_min_sublabel_size = group_HBPLL_min_sublabel_size;
  mm.clear_labels();

  cout << "avg_ST_label generation time: "
       << (avg_runningtime / iteration_graph_times) << "s" << endl;
  cout << "avg_st_label_size: " << (avg_st_label_size / iteration_graph_times)
       << endl;
  cout << "avg_st_max_subLabel_size: "
       << (avg_st_max_sublabel_size / iteration_graph_times) << endl;
  cout << "avg_st_min_subLabel_size: "
       << (avg_st_min_sublabel_size / iteration_graph_times) << endl
       << endl;
  cout << "avg_group_generation_time: "
       << (avg_group_generation_time / iteration_graph_times) << "s" << endl;
  cout << "avg_group_HBPLL_generation_time: "
       << (avg_group_HBPLL_generation_time / iteration_graph_times) << "s"
       << endl;
  cout << "avg_group_HBPLL_total_time: "
       << (avg_group_HBPLL_total_time / iteration_graph_times) << "s" << endl;
  cout << "avg_group_HBPLL_label_size: "
       << (avg_group_HBPLL_label_size / iteration_graph_times) << endl;
  cout << "avg_group_HBPLL_max_subLabel_size: "
       << (avg_group_HBPLL_max_sublabel_size / iteration_graph_times) << endl;
  cout << "avg_group_HBPLL_min_subLabel_size: "
       << (avg_group_HBPLL_min_sublabel_size / iteration_graph_times) << endl;

  cout << "avg_group_HBPLL_label_size / avg_st_label_size: "
       << ((avg_group_HBPLL_label_size / iteration_graph_times) /
           (avg_st_label_size / iteration_graph_times))
       << endl;
}
#endif // TEST_H
