#pragma once

/*the following codes are for testing

---------------------------------------------------
a cpp file (try.cpp) for running the following test code:
----------------------------------------

#include <build_in_progress/HL/Hop/test_HBPLL.h>
using namespace std;

int main()
{
    test_HBPLL();
}


------------------------------------------------------------------------------------------
Commends for running the above cpp file on Linux:

g++ -std=c++17 -I/__PATH__/boost_1_75_0 -I/__PATH__/rucgraph run.cpp -lpthread -Ofast -o A
./A
rm A 

(optional to put the above commends in run.sh, and then use the comment: sh run.sh)

*/

#include <Hop/graph_v_of_v_idealID_HB_v1.h>
#include <Hop/graph_v_of_v_idealID_HB_query.h>
#include <Hop/graph_v_of_v_idealID_HB_shortest_distance.h>
#include <graph_v_of_v_idealID/random_graph/graph_v_of_v_idealID_generate_random_connected_graph.h>
#include <graph_v_of_v_idealID/read_save/graph_v_of_v_idealID_read.h>
#include <graph_v_of_v_idealID/read_save/graph_v_of_v_idealID_save.h>
#include <graph_v_of_v_idealID/graph_v_of_v_idaelID_sort.h>
#include "graph_v_of_v_idealID/graph_v_of_v_idealID.h"


#include <boost/random.hpp>
boost::random::mt19937 boost_random_time_seed{ static_cast<std::uint32_t>(std::time(0)) };

bool debug = 0;
int source_debug = 0;
int terminal_debug = 0;
int hop_cst_debug = 0;

void graph_v_of_v_idealID_HB_v2_check_correctness(graph_v_of_v_idealID_two_hop_case_info_v1& case_info, graph_v_of_v_idealID& instance_graph,
    int iteration_source_times, int iteration_terminal_times, bool check_path) {
    /*
    below is for checking whether the above labels are right (by randomly computing shortest paths)

    this function can only be used when 0 to n-1 is in the graph, i.e., the graph is an ideal graph
    */

    boost::random::uniform_int_distribution<> vertex_range{ static_cast<int>(0), static_cast<int>(instance_graph.size() - 1) };
    boost::random::uniform_int_distribution<> hop_range{ static_cast<int>(1), static_cast<int>(10) };

    for (int yy = 0; yy < iteration_source_times; yy++) {
        int source = vertex_range(boost_random_time_seed);
        std::vector<double> distances;
        distances.resize(instance_graph.size());
        std::vector<int> predecessors;
        predecessors.resize(instance_graph.size());

        int hop_cst = hop_range(boost_random_time_seed);

        if (debug) {
            source = source_debug;
            hop_cst = hop_cst_debug;
        }

        graph_v_of_v_idealID_HB_shortest_distance(instance_graph, source, hop_cst, distances);

        for (int xx = 0; xx < iteration_terminal_times; xx++) {
            int terminal = vertex_range(boost_random_time_seed);

            if (debug)
                terminal = terminal_debug;

            double dis;
            auto begin = std::chrono::high_resolution_clock::now();
            if (case_info.value_M == 0) {
                if (case_info.use_2019R2 || case_info.use_enhanced2019R2 || case_info.use_non_adj_reduc_degree) {
                    dis = graph_v_of_v_idealID_two_hop_v2_extract_distance_st_no_R1(case_info.L2, case_info.reduction_measures_2019R2, source, terminal, hop_cst);
                }
                else {
                    dis = graph_v_of_v_idealID_two_hop_v2_extract_distance_no_reduc(case_info.L2, source, terminal, hop_cst);
                }
            }
            else {
                if (case_info.use_2019R2 || case_info.use_enhanced2019R2 || case_info.use_non_adj_reduc_degree) {
                    dis = graph_v_of_v_idealID_two_hop_v2_extract_distance_st_no_R1_for_M(case_info.L2, case_info.reduction_measures_2019R2, source, terminal, hop_cst, case_info.value_M);
                }
                else {
                    dis = graph_v_of_v_idealID_two_hop_v2_extract_distance_no_reduc_for_M(case_info.L2, source, terminal, hop_cst, case_info.value_M);
                }
            }
            auto end = std::chrono::high_resolution_clock::now();
            case_info.time_query += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;

            if (abs(dis - distances[terminal]) > 1e-4 && (dis < std::numeric_limits<double>::max() || distances[terminal] < std::numeric_limits<double>::max())) {
                cout << "source = " << source << endl;
                cout << "terminal = " << terminal << endl;
                cout << "hop_cst = " << hop_cst << endl;
                cout << "source vector:" << endl;
                case_info.print_L_vk(source);
                cout << "terminal vector:" << endl;
                case_info.print_L_vk(terminal);
                cout << "dis = " << dis << endl;
                cout << "distances[terminal] = " << distances[terminal] << endl;
                cout << "abs(dis - distances[terminal]) > 1e-5!" << endl;
                getchar();
            }

            if (check_path) {
                vector<pair<int, int>> path;
                if (case_info.value_M == 0) {
                    path = graph_v_of_v_idealID_two_hop_v2_extract_shortest_path_st_no_R1(case_info.L2, case_info.reduction_measures_2019R2, source, terminal, hop_cst);
                }
                else {
                    path = graph_v_of_v_idealID_two_hop_v2_extract_shortest_path_st_no_R1_for_M(case_info.L2, case_info.reduction_measures_2019R2, source, terminal, hop_cst, case_info.value_M);
                }

                double path_dis = 0;
                if (path.size() == 0) {
                    if (source != terminal) {
                        path_dis = std::numeric_limits<double>::max();
                    }
                }
                else {
                    for (auto it = path.begin(); it != path.end(); it++) {
                        double edge_weight = graph_v_of_v_idealID_edge_weight(instance_graph, it->first, it->second);
                        path_dis += edge_weight;
                        if (path_dis > std::numeric_limits<double>::max()) {
                            path_dis = std::numeric_limits<double>::max();
                        }
                    }
                }
                if (abs(dis - path_dis) > 1e-4 && (dis < std::numeric_limits<double>::max() || distances[terminal] < std::numeric_limits<double>::max())) {
                    cout << "source = " << source << endl;
                    cout << "terminal = " << terminal << endl;
                    cout << "hop_cst = " << hop_cst << endl;
                    cout << "source vector:" << endl;
                    case_info.print_L_vk(source);
                    cout << "terminal vector:" << endl;
                    case_info.print_L_vk(terminal);
                    std::cout << "print_vector_pair_int:" << std::endl;
                    for (int i = 0; i < path.size(); i++) {
                        std::cout << "item: |" << path[i].first << "," << path[i].second << "|" << std::endl;
                    }
                    cout << "dis = " << dis << endl;
                    cout << "path_dis = " << path_dis << endl;
                    cout << "abs(dis - path_dis) > 1e-5!" << endl;
                    getchar();
                }
            }
        }
    }
}

void test_HBPLL() {

    /* problem parameters */
    int iteration_graph_times = 1;
    int V = 100, E = 500, precision = 1, thread_num = 1; //if generate_new_graph=true, set these params
    double ec_min = 1, ec_max = 10;
    int upper_k = 10;

    /* test parameters */
    bool generate_new_graph = false;
    string data_path = "../../../data/wiki-RfA2.txt";
    bool print_time_details = true;
    bool print_label_before_canonical_fix = false;
    bool print_L = false;
    bool check_correctness = true;
    bool check_path = true;

    /* algorithm parameters */
    bool use_2019R2 = false, use_enhanced2019R2 = false, use_non_adj_reduc_degree = false;
    bool use_rank_pruning = true;
    bool use_canonical_repair = true;
    bool use_M = false;

    /* debug parameters */
    int iteration_source_times = 100, iteration_terminal_times = 100;
    debug = 0;
    if (debug) {
        source_debug = 0;
        terminal_debug = 5;
        hop_cst_debug = 2;
        generate_new_graph = 0;
        iteration_graph_times = 1;
        iteration_source_times = 1;
        iteration_terminal_times = 1;
        print_L = 1;
    }

    /* hop bounded info */
    graph_v_of_v_idealID_two_hop_case_info_v1 mm;
    mm.use_rank_pruning = use_rank_pruning;
    mm.value_M = use_M ? ec_max * E : 0;
    mm.upper_k = upper_k;
    mm.use_2019R2 = use_2019R2;
    mm.use_enhanced2019R2 = use_enhanced2019R2;
    mm.use_non_adj_reduc_degree = use_non_adj_reduc_degree;
    mm.print_label_before_canonical_fix = print_label_before_canonical_fix;
    mm.use_canonical_repair = use_canonical_repair;

    /* result info */
    double avg_index_time = 0, avg_index_size_per_v = 0, avg_MG_num = 0;
    double avg_query_time = 0, avg_canonical_repair_remove_label_ratio = 0;
    double total_time_initialization = 0, total_time_reduction = 0, total_time_generate_labels = 0;
    double total_time_update_predecessors = 0, total_time_canonical_repair = 0;

    /* iteration */
    for (int i = 0; i < iteration_graph_times; i++) {
        cout << ">>>iteration_graph_times: " << i << endl;

        graph_v_of_v_idealID instance_graph;
        if (generate_new_graph) {
            instance_graph = graph_v_of_v_idealID_generate_random_connected_graph(V, E, ec_min, ec_max, precision, boost_random_time_seed);
            instance_graph = graph_v_of_v_idealID_sort(instance_graph);
            graph_v_of_v_idealID_save("simple_iterative_tests_HBPLL.txt", instance_graph);
        } else {
          graph_v_of_v_idealID_read(data_path.data(), instance_graph);
          printf("size:%lu\n",instance_graph.size());
        }

        auto begin = std::chrono::high_resolution_clock::now();
        try {
            graph_v_of_v_idealID_HB_v2(instance_graph, instance_graph.size(), thread_num, mm);
            if (print_time_details) {
                total_time_initialization += mm.time_initialization;
                total_time_reduction += mm.time_reduction;
                total_time_generate_labels += mm.time_generate_labels;
                total_time_update_predecessors += mm.time_update_predecessors;
                total_time_canonical_repair += mm.time_canonical_repair;
            }
        } catch (string s) {
            cout << s << endl;
            graph_v_of_v_idealID_two_hop_clear_global_values();
            continue;
        }

        cout << "finish generating labels" << endl;

        auto end = std::chrono::high_resolution_clock::now();

        double runningtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;  // s
        avg_index_time = avg_index_time + runningtime / iteration_graph_times;
        avg_MG_num = avg_MG_num + (double) mm.MG_num / iteration_graph_times;

        if (print_L)
            mm.print_L();

        if (check_correctness) {
            graph_v_of_v_idealID_HB_v2_check_correctness(mm, instance_graph, iteration_source_times, iteration_terminal_times, check_path);
        }

        avg_query_time += mm.time_query;
        avg_index_size_per_v += (double)mm.compute_label_size() / V / iteration_graph_times;
        avg_canonical_repair_remove_label_ratio += (double) ((double)canonical_removed_labels / (double)mm.compute_label_size()) / iteration_graph_times;

        graph_v_of_v_idealID_two_hop_clear_global_values2();
        mm.clear_labels();
    }

    cout << "avg_index_time: " << avg_index_time << "s" << endl;
    cout << "avg_index_size_per_v: " << avg_index_size_per_v << endl;
    if (mm.use_2019R2 || mm.use_enhanced2019R2 || mm.use_non_adj_reduc_degree)
        cout << "avg_MG_num: " << avg_MG_num << endl;
    if (mm.use_canonical_repair)
        cout << "avg_canonical_repair_remove_label_ratio: " << avg_canonical_repair_remove_label_ratio << endl;
    if (check_correctness)
        cout << "avg_query_time: " << avg_query_time / (iteration_graph_times) << endl;
    if (print_time_details) {
        cout << "\t total_time_initialization: " << total_time_initialization << endl;
        cout << "\t total_time_reduction: " << total_time_reduction << endl;
        cout << "\t total_time_generate_labels: " << total_time_generate_labels << endl;
        cout << "\t total_time_update_predecessors: " << total_time_update_predecessors << endl;
        cout << "\t total_time_canonical_repair: " << total_time_canonical_repair << endl;
    }
}
