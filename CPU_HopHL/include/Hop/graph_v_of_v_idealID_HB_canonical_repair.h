#pragma once
#include <Hop/graph_v_of_v_idealID_two_hop_labels_v1.h>

/**
 * new canonical repair
 *
 *
 *
 */

// return query(u,v,h) using L_{>=r(v)}[u] and L[v]
double canonical_repair_query(int u, int j, int i) {
    // j: v's position in L[u]
    // i: label's position in L[u][j].dist_info
    int v = L2_temp_599[u][j].vertex;
    int hop_cst = get<2>(L2_temp_599[u][j].dist_info[i]);

    if (u == v) {
        return 0;
    }
    double distance = std::numeric_limits<double>::max();

    auto ptr_u = L2_temp_599[u].begin(), ptr_v = L2_temp_599[v].begin();
    auto ptr_u_end = L2_temp_599[u].end(), ptr_v_end = L2_temp_599[v].end();

    while (ptr_u != ptr_u_end && ptr_v != ptr_v_end) {
        if (ptr_u->vertex == ptr_v->vertex) {
            int ii = 0, size_u = ptr_u->dist_info.size();
            auto begin1 = ptr_u->dist_info.begin();
            auto end2 = ptr_v->dist_info.end();
            while (size_u > 0) {
                auto ptr2 = ptr_v->dist_info.begin();
                while (1) {
                    if (get<2>(*(begin1+ii)) + get<2>(*ptr2) <= hop_cst) {
                        double dis = get<0>(*(begin1+ii)) + get<0>(*ptr2);
                        if (distance > dis) {
                            distance = dis;
                        }
                    } else {
                        break;
                    }
                    ptr2++;
                    if (ptr2 == end2) {
                        break;
                    }
                }
                ii++;
                if (ii == size_u) {
                    break;
                }
                if (ptr_u->vertex >= v && ii > i) {
                    return  distance;
                }
            }
            ptr_u++;
        } else if (ptr_u->vertex > ptr_v->vertex) {
            ptr_v++;
        } else {
            ptr_u++;
        }
    }

    return distance;
}

double canonical_repair_query_for_M(int u, int j, int i, int value_M) {
    int v = L2_temp_599[u][j].vertex;
    int hop_cst = int(get<0>(L2_temp_599[u][j].dist_info[i]) / value_M);

    if (u == v) {
        return 0;
    }
    double distance = std::numeric_limits<double>::max();

    auto ptr_u = L2_temp_599[u].begin(), ptr_v = L2_temp_599[v].begin();
    auto ptr_u_end = L2_temp_599[u].end(), ptr_v_end = L2_temp_599[v].end();

    while (ptr_u != ptr_u_end && ptr_v != ptr_v_end) {
        if (ptr_u->vertex == ptr_v->vertex) {
            int ii = 0, size_u = ptr_u->dist_info.size();
            auto begin1 = ptr_u->dist_info.begin();
            auto end2 = ptr_v->dist_info.end();
            while (size_u > 0) {
                auto ptr2 = ptr_v->dist_info.begin();
                while (1) {
                    int hop_sum = int(get<0>(*(begin1+ii)) / value_M) + int(get<0>(*ptr2) / value_M);
                    if (hop_sum <= hop_cst) {
                        double dis = get<0>(*(begin1+ii)) + get<0>(*ptr2) - value_M * hop_sum;
                        if (distance > dis) {
                            distance = dis;
                        }
                    } else {
                        break;
                    }
                    ptr2++;
                    if (ptr2 == end2) {
                        break;
                    }
                }
                ii++;
                if (ii == size_u) {
                    break;
                }
                if (ptr_u->vertex >= v && ii > i) {
                    return  distance;
                }
            }
            ptr_u++;
        } else if (ptr_u->vertex > ptr_v->vertex) {
            ptr_v++;
        } else {
            ptr_u++;
        }
    }

    return distance;
}

void canonical_repair_element_v2(int u, vector<two_hop_label_v2> &L2_u, int value_M) {
    int size1 = L2_temp_599[u].size();
    auto it1 = L2_temp_599[u].begin();
    for (int j = 0; j < size1; j++) {
        int v = (it1+j)->vertex;
        if (v == u) {
            continue;
        }

        int size2 = (it1+j)->dist_info.size();
        auto it2 = (it1+j)->dist_info.begin();
        if (value_M == 0){
            for (int i = 0; i < size2; i++) {
                double query_dis = canonical_repair_query(u, j, i);
                if (query_dis < get<0>(*(it2 + i))) {
                    L2_u[j].dist_info[i] = {-1, -1, 100};
                    canonical_removed_labels++;
                }
            }
        } else {
            for (int i = 0; i < size2; i++) {
                double query_dis = canonical_repair_query_for_M(u, j, i, value_M);
                double dist = get<0>(*(it2+i)) - (int(get<0>(*(it2+i))/value_M) * value_M);
                if (query_dis < dist) {
                    L2_u[j].dist_info[i] = {-1,-1,100};
                    canonical_removed_labels++;
                }
            }
        }
    }
}

void canonical_repair_multi_threads_v2(graph_v_of_v_idealID_two_hop_case_info_v1 &case_info, int num_of_threads) {

    int N = case_info.L2.size();
    int value_M = case_info.value_M;
    ThreadPool pool(num_of_threads);
    std::vector<std::future<int>> results;

    auto begin = std::chrono::high_resolution_clock::now();

    for (int target_v = 0; target_v < N; target_v++) {
        int size = L2_temp_599[target_v].size();
        auto L2_v = &case_info.L2[target_v];
        if (size > 0) {
            results.emplace_back(
                    pool.enqueue([target_v, L2_v, value_M] {
                        canonical_repair_element_v2(target_v, *L2_v, value_M);
                        return 1;
                    }));
        }
    }
    for (auto &&result : results)
        result.get();
    results.clear();
}

/**
 * v3
 *
 *
 *
 */
void canonical_repair_element1_v3(int u, vector<two_hop_label_v2> &L2_u, int value_M) {
    int size1 = L2_temp_599[u].size();
    auto it1 = L2_temp_599[u].begin();
    for (int j = 0; j < size1; j++) {
        int v = (it1+j)->vertex;
        if (v == u) {
            continue;
        }
        vector<tuple<double,int,int>>().swap(L2_u[j].dist_info);

        //  iterate through vertex v's dist_info
        int size2 = (it1+j)->dist_info.size();
        auto it2 = (it1+j)->dist_info.begin();
        if (value_M == 0){
            // do not use M
            for (int i = 0; i < size2; i++) {
                double query_dis = canonical_repair_query(u, j, i);
                if (query_dis >= get<0>(*(it2 + i))) {
                    L2_u[j].dist_info.push_back(L2_temp_599[u][j].dist_info[i]);
                } else {
                    canonical_removed_labels++;
                }
            }
        } else {
            // use M
            for (int i = 0; i < size2; i++) {
                double query_dis = canonical_repair_query_for_M(u, j, i, value_M);
                double dist = get<0>(*(it2+i)) - (int(get<0>(*(it2+i))/value_M) * value_M);
                if (query_dis >= dist) {
                    L2_u[j].dist_info.push_back(L2_temp_599[u][j].dist_info[i]);
                } else {
                    canonical_removed_labels++;
                }
            }
        }
    }
}

void canonical_repair_element2_v3(int v, vector<two_hop_label_v2> &L2_u) {
    vector<two_hop_label_v2> L2_u_temp(L2_u);
    vector<two_hop_label_v2>().swap(L2_u);
    int size = L2_u_temp.size();
    for (int i = 0; i < size; i++) {
        if (L2_u_temp[i].dist_info.size() != 0) {
            L2_u.push_back(L2_u_temp[i]);
        }
    }
}

void canonical_repair_multi_threads_v3(graph_v_of_v_idealID_two_hop_case_info_v1 &case_info, int num_of_threads) {

    int N = case_info.L2.size();
    int value_M = case_info.value_M;

    ThreadPool pool(num_of_threads);
    std::vector<std::future<int>> results;

    auto begin = std::chrono::high_resolution_clock::now();

    /*find labels_to_be_removed*/
    for (int target_v = 0; target_v < N; target_v++) {
        int size = L2_temp_599[target_v].size();
        auto L2_v = &case_info.L2[target_v];
        if (size > 0) {
            results.emplace_back(
                    pool.enqueue([target_v, L2_v, value_M] {
                        canonical_repair_element1_v3(target_v, *L2_v, value_M);
                        return 1;
                    }));
        }
    }
    for (auto &&result : results)
        result.get();
    results.clear();

    for (int target_v = 0; target_v < N; target_v++) {
        int old_size = L2_temp_599[target_v].size();
        auto L2_v = &case_info.L2[target_v];
        if (0 < old_size) {
            results.emplace_back(
                    pool.enqueue([target_v, L2_v] {
                        canonical_repair_element2_v3(target_v, *L2_v);
                        return 1;
                    })
            );
        }
    }
    for (auto &&result : results)
        result.get();
    results.clear();
}
