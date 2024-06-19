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

#include <HBPLL/test.h>


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
//#include <HBPLL/PLL.h>
//#include <HBPLL/two_hop_labels.h>
#include <graph_v_of_v/graph_v_of_v.h>
#include <graph_v_of_v/graph_v_of_v_generate_random_graph.h>
#include <graph_v_of_v/graph_v_of_v_shortest_paths.h>
#include <graph_v_of_v/graph_v_of_v_update_vertexIDs_by_degrees_large_to_small.h>

#include <HBPLL/hop_constrained_two_hop_labels_generation.h>
#include <graph_v_of_v/graph_v_of_v_hop_constrained_shortest_distance.h>
#include <limits>
#include <text_mining/binary_save_read_vector.h>
#include <text_mining/print_items.h>


#define DATASET_PATH "../../data/email-Enron2.txt"
string reach_limit_error_string_MB = "reach limit error MB";
string reach_limit_error_string_time = "reach limit error time";

boost::random::mt19937 boost_random_time_seed{
    static_cast<std::uint32_t>(std::time(0))};

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

    // while (!is_mock[source]) {
    //   source = vertex_range(boost_random_time_seed);
    // }

    std::vector<int> distances(instance_graph.size());

    int hop_cst = hop_range(boost_random_time_seed);

    graph_v_of_v_hop_constrained_shortest_distance(instance_graph, source,
                                                   hop_cst, distances);

    for (int xx = 0; xx < iteration_terminal_times; xx++) {
      int terminal = vertex_range(boost_random_time_seed);

      // while (is_mock[terminal]) {
      //   terminal = vertex_range(boost_random_time_seed);
      // }

      int query_dis = hop_constrained_extract_distance(case_info.L, source,
                                                       terminal, hop_cst);

      if (abs(query_dis - distances[terminal]) > 1e-4) {
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
      if (path.size() == 0 && source != terminal) {
        path_dis = std::numeric_limits<int>::max();
      }
      for (auto xx : path) {
        path_dis += instance_graph.edge_weight(xx.first, xx.second);
      }
      if (abs(query_dis - path_dis) > 1e-4) {
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
  int iteration_graph_times = 1, iteration_source_times = 10,
      iteration_terminal_times = 10;
  int V = 10, E = 10;
  int ec_min = 1, ec_max = 10;

  bool generate_new_random_graph = 0;
  bool load_new_graph = 1;

  /* hop bounded info */
  hop_constrained_case_info mm;
  mm.upper_k = 5;
  mm.use_rank_prune = 1;
  mm.use_2023WWW_generation = 0;
  mm.use_canonical_repair = 1;
  mm.max_run_time_seconds = 10;
  mm.thread_num = 10;

  /* result info */
  double avg_index_time = 0;
  double avg_time_initialization = 0, avg_time_generate_labels = 0,
         avg_time_sortL = 0, avg_time_canonical_repair = 0;
  double avg_canonical_repair_remove_label_ratio = 0, avg_index_size_per_v = 0;

  /* iteration */
  for (int i = 0; i < iteration_graph_times; i++) {
    cout << ">>>iteration_graph_times: " << i << endl;

    graph_v_of_v<int> instance_graph;
    if (generate_new_random_graph == 1) {
      instance_graph = graph_v_of_v_generate_random_graph<int>(
          V, E, ec_min, ec_max, 1, boost_random_time_seed);
      /*add vertex groups*/
      instance_graph = graph_v_of_v_update_vertexIDs_by_degrees_large_to_small(
          instance_graph); // sort vertices
      instance_graph.txt_save("simple_iterative_tests.txt");

    } else if (load_new_graph) {
      instance_graph.txt_read(DATASET_PATH);

      instance_graph = graph_v_of_v_update_vertexIDs_by_degrees_large_to_small(
          instance_graph); // sort vertices

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

    long long int index_size = 0;
    for (auto it = mm.L.begin(); it != mm.L.end(); it++) {
      index_size = index_size + (*it).size();
    }
    avg_index_size_per_v =
        avg_index_size_per_v + (double)index_size / V / iteration_graph_times;

    hop_constrained_check_correctness(mm, instance_graph,
                                      iteration_source_times,
                                      iteration_terminal_times, mm.upper_k);

    mm.clear_labels();
  }

  cout << "avg_canonical_repair_remove_label_ratio: "
       << avg_canonical_repair_remove_label_ratio << endl;
  cout << "avg_index_time: " << avg_index_time << "s" << endl;
  cout << "\t avg_time_initialization: " << avg_time_initialization << endl;
  cout << "\t avg_time_generate_labels: " << avg_time_generate_labels << endl;
  cout << "\t avg_time_sortL: " << avg_time_sortL << endl;
  cout << "\t avg_time_canonical_repair: " << avg_time_canonical_repair << endl;
  cout << "\t avg_index_size_per_v: " << avg_index_size_per_v << endl;
}
