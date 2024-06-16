#pragma once
#include <map>

#include <shared_mutex>
#include <tool_functions/ThreadPool.h>
#include <graph_v_of_v_idealID/graph_v_of_v_idealID.h>

using namespace std;

/* label format */
class two_hop_label_v1 {
public:
    int vertex, parent_vertex;
    int hop;
    double distance;
};

class two_hop_label_v2 {
public:
    int vertex;
    // distance, parent_vertex, hop
    vector<tuple<double, int, int>> dist_info;
};

/*
    global values
    unique code for this file: 599
*/
string reach_limit_error_string_MB = "reach limit error MB";
string reach_limit_error_string_time = "reach limit error time";
long long int max_labal_size_599;
long long int labal_size_599;
int max_N_ID_for_mtx_599 = 1e7;
double max_run_time_nanoseconds_599;
auto begin_time_599 = std::chrono::high_resolution_clock::now();
vector<std::shared_timed_mutex> mtx_599(max_N_ID_for_mtx_599);
long long int canonical_removed_labels;

graph_v_of_v_idealID ideal_graph_599;
map<pair<int, int>, int> new_edges_with_middle_v_599;
map<pair<int, int>, double> new_edges_with_origin_ec_599;
vector<vector<two_hop_label_v1>> L_temp_599;
vector<vector<two_hop_label_v2>> L2_temp_599;
vector<int> reduction_measures_2019R2;

vector<vector<vector<pair<double, int>>>> Temp_L_vk_599;
vector<vector<pair<double, int>>> dist_hop_599;
queue<int> Qid_599;

void graph_v_of_v_idealID_two_hop_clear_global_values() {
    vector<vector<two_hop_label_v1>>().swap(L_temp_599);
    ideal_graph_599.clear();
    vector<int>().swap(reduction_measures_2019R2);
    map<pair<int, int>, int>().swap(new_edges_with_middle_v_599);
    map<pair<int, int>, double>().swap(new_edges_with_origin_ec_599);
    vector<vector<two_hop_label_v2>>().swap(L2_temp_599);
    vector<vector<vector<pair<double, int>>>>().swap(Temp_L_vk_599);
    vector<vector<pair<double, int>>>().swap(dist_hop_599);
    queue<int>().swap(Qid_599);
}

/* global querying values, used in the query func */
map<int, vector<pair<int, double>>> R2_reduced_vertices;

void graph_v_of_v_idealID_two_hop_clear_global_values2() {
    map<int, vector<pair<int, double>>>().swap(R2_reduced_vertices);
}

class graph_v_of_v_idealID_two_hop_case_info_v1 {
public:
    /*hop bounded*/
    int upper_k = 0;
    int value_M = 0;
    bool use_rank_pruning = true;
    bool use_canonical_repair = false;
    bool print_label_before_canonical_fix = 0;

    /*use reduction info*/
    bool use_2019R2 = false;
    bool use_enhanced2019R2 = false;
    bool use_non_adj_reduc_degree = false;
    int max_degree_MG_enhanced2019R2 = 100;
    int MG_num = 0;

    /*running time records*/
    double time_initialization = 0;
    double time_reduction = 0;
    double time_generate_labels = 0;
    double time_update_predecessors = 0;
    double time_canonical_repair = 0;

    double time_query = 0;

    /*running limits*/
    long long int max_labal_size = 1e12;
    double max_run_time_seconds = 1e12;

    /*labels*/
    vector<int> reduction_measures_2019R2;
    vector<vector<two_hop_label_v2>> L2;

    long long int compute_label_bit_size() {
        long long int size = 0;
        size = size + reduction_measures_2019R2.size() * 4;
        for (auto it = L2.begin(); it != L2.end(); it++) {
            size = size + (*it).size() * sizeof(two_hop_label_v2);
        }
        return size;
    }

    long long int compute_label_size() {
        long long int size = 0;
        for (auto it = L2.begin(); it != L2.end(); it++) {
            for(auto it2 = it->begin(); it2 != it->end(); it2++) {
                for(auto it3 = it2->dist_info.begin(); it3 != it2->dist_info.end(); it3++) {
                    size++;
                }
            }
        }
        return size;
    }

    /*clear labels*/
    void clear_labels() {
        vector<int>().swap(reduction_measures_2019R2);
        vector<vector<two_hop_label_v2>>().swap(L2);
    }

    /*printing*/
    void print_L() {
        cout << "print_L:" << endl;
        for (int i = 0; i < L2.size(); i++) {
            cout << "L[" << i << "]=";
            for (auto it = L2[i].begin(); it != L2[i].end(); it++) {
                cout << "{" << it->vertex << ",";
                for (auto j = it->dist_info.begin(); j != it->dist_info.end(); j++) {
                    cout << "(" << get<0>(*j) << "," << get<1>(*j) << "," << get<2>(*j) << ")";
                }
                cout << "} ";
            }
            cout << endl;
        }
    }

    void print_L_vk(int v_k) {
        for (auto it = L2[v_k].begin(); it != L2[v_k].end(); it++) {
            cout << "<" << it->vertex << ",";
            for (auto j = it->dist_info.begin(); j != it->dist_info.end(); j++) {
                cout << "(" << get<0>(*j) << "," << get<1>(*j) << "," << get<2>(*j) << ")";
            }
            cout << ">";
        }
        cout << endl;
    }

    void print_reduction_measures_2019R2() {
        cout << "print_reduction_measures_2019R2:" << endl;
        for (int i = 0; i < reduction_measures_2019R2.size(); i++) {
            cout << "reduction_measures_2019R2[" << i << "]=" << reduction_measures_2019R2[i] << endl;
        }
    }
};

bool compare_two_hop_label_small_to_large(two_hop_label_v1 &i, two_hop_label_v1 &j) {
    if (i.vertex != j.vertex) {
        return i.vertex < j.vertex;
    } else if (i.hop != j.hop) {
        return i.hop < j.hop;
    } else {
        return i.distance > j.distance;
    }
}