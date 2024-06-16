#pragma once

#include <Hop/graph_v_of_v_idealID_two_hop_labels_v1.h>

double graph_v_of_v_idealID_two_hop_v2_extract_distance_no_reduc(vector<vector<two_hop_label_v2>> &L2, int source, int terminal, int hop_cst) {
    /*return std::numeric_limits<double>::max() is not connected*/
    if (hop_cst < 0) {
        return std::numeric_limits<double>::max();
    }
    if (source == terminal) {
        return 0;
    } else if (hop_cst == 0) {
        return std::numeric_limits<double>::max();
    }

    double distance = std::numeric_limits<double>::max();
    auto vector1_check_pointer = L2[source].begin();
    auto vector2_check_pointer = L2[terminal].begin();
    auto pointer_L_s_end = L2[source].end(), pointer_L_t_end = L2[terminal].end();

    while (vector1_check_pointer != pointer_L_s_end && vector2_check_pointer != pointer_L_t_end) {
        if (vector1_check_pointer->vertex == vector2_check_pointer->vertex) {
            // in the common vertex
            auto ptr1 = vector1_check_pointer->dist_info.end() - 1;
            auto ptr2 = vector2_check_pointer->dist_info.end() - 1;
            auto begin1 = vector1_check_pointer->dist_info.begin();
            auto begin2 = vector2_check_pointer->dist_info.begin();
            while (1) {
                if (get<0>(*ptr1) != -1) {
                    if (get<2>(*ptr1) + get<2>(*ptr2) <= hop_cst && get<0>(*ptr2) != -1) {
                        double dis = get<0>(*ptr1) + get<0>(*ptr2);
                        if (distance > dis) {
                            distance = dis;
                        }
                        break;
                    }
                    if (ptr2 != begin2) {
                        ptr2--;
                        while (1) {
                            if (get<0>(*ptr2) != -1) {
                                if (get<2>(*ptr1) + get<2>(*ptr2) <= hop_cst) {
                                    double dis = get<0>(*ptr1) + get<0>(*ptr2);
                                    if (distance > dis) {
                                        distance = dis;
                                    }
                                    break;
                                }
                            }
                            if (ptr2 == begin2) {
                                break;
                            }
                            ptr2--;
                        }
                    }
                }
                if (ptr1 == begin1) {
                    break;
                }
                ptr1--;
                ptr2 = vector2_check_pointer->dist_info.end() - 1;
            }
            vector1_check_pointer++;
        } else if (vector1_check_pointer->vertex > vector2_check_pointer->vertex) {
            vector2_check_pointer++;
        } else {
            vector1_check_pointer++;
        }
    }

    return distance;
}

double graph_v_of_v_idealID_two_hop_v2_extract_distance_st_no_R1(vector<vector<two_hop_label_v2>> &L, vector<int> &reduction_measures_2019R2, int source, int terminal, int hop_cst) {
    if (hop_cst < 0) {
        return std::numeric_limits<double>::max();
    }
    if (source == terminal) {
        return 0;
    } else if (hop_cst == 0) {
        return std::numeric_limits<double>::max();
    }

    double min_selected_distance = std::numeric_limits<double>::max();

    if (reduction_measures_2019R2[source] == 2) {
        if (reduction_measures_2019R2[terminal] == 2) {
            /*"Both reduced"*/
            // cout << "case 1" << endl;
            auto s_adj_begin = R2_reduced_vertices[source].begin();
            auto s_adj_end = R2_reduced_vertices[source].end();
            auto t_adj_begin = R2_reduced_vertices[terminal].begin();
            auto t_adj_end = R2_reduced_vertices[terminal].end();
            for (auto it1 = s_adj_begin; it1 != s_adj_end; it1++) {
                for (auto it2 = t_adj_begin; it2 != t_adj_end; it2++) {
                    double x = graph_v_of_v_idealID_two_hop_v2_extract_distance_no_reduc(L, it1->first, it2->first, hop_cst - 2);
                    if (x == std::numeric_limits<double>::max()) {
                        continue;
                    } else {
                        min_selected_distance = min(min_selected_distance, x + double(it1->second) + double(it2->second));
                    }
                }
            }
        } else {
            /*"Only source reduced"*/
            // cout << "case 2" << endl;
            auto s_adj_begin = R2_reduced_vertices[source].begin();
            auto s_adj_end = R2_reduced_vertices[source].end();
            for (auto it1 = s_adj_begin; it1 != s_adj_end; it1++) {
                double x = graph_v_of_v_idealID_two_hop_v2_extract_distance_no_reduc(L, it1->first, terminal, hop_cst - 1);
                if (x == std::numeric_limits<double>::max()) {
                    continue;
                } else {
                    min_selected_distance = min(min_selected_distance, x + double(it1->second));
                }
            }
        }
    } else {
        if (reduction_measures_2019R2[terminal] == 2) {
            /*"Only terminal reduced"*/
            // cout << "case 3" << endl;
            auto t_adj_begin = R2_reduced_vertices[terminal].begin();
            auto t_adj_end = R2_reduced_vertices[terminal].end();
            for (auto it2 = t_adj_begin; it2 != t_adj_end; it2++) {
                double x = graph_v_of_v_idealID_two_hop_v2_extract_distance_no_reduc(L, source, it2->first, hop_cst - 1);
                if (x == std::numeric_limits<double>::max()) {
                    continue;
                } else {
                    min_selected_distance = min(min_selected_distance, x + double(it2->second));
                }
            }
        } else {
            min_selected_distance = min(min_selected_distance, graph_v_of_v_idealID_two_hop_v2_extract_distance_no_reduc(L, source, terminal, hop_cst));
        }
    }

    return min_selected_distance;
}

vector<pair<int, int>> graph_v_of_v_idealID_two_hop_v2_extract_shortest_path_st_no_R1(vector<vector<two_hop_label_v2>> &L2, vector<int> &reduction_measures_2019R2, int source, int terminal, int hop_cst) {
    vector<pair<int, int>> paths;
    if (source == terminal) {
        return paths;
    }

    double min_dis = std::numeric_limits<double>::max();
    vector<pair<int, int>> partial_edges(2);

    if (reduction_measures_2019R2[source] == 2) {
        auto s_adj_begin = R2_reduced_vertices[source].begin();
        auto s_adj_end = R2_reduced_vertices[source].end();
        if (reduction_measures_2019R2[terminal] == 2) {
            /*"Both reduced"*/
            auto t_adj_begin = R2_reduced_vertices[terminal].begin();
            auto t_adj_end = R2_reduced_vertices[terminal].end();
            for (auto it1 = s_adj_begin; it1 != s_adj_end; it1++) {
                for (auto it2 = t_adj_begin; it2 != t_adj_end; it2++) {
                    double x = graph_v_of_v_idealID_two_hop_v2_extract_distance_no_reduc(L2, it1->first, it2->first, hop_cst - 2) + double(it1->second) + double(it2->second);
                    if (min_dis > x) {
                        min_dis = x;
                        partial_edges[0] = {it1->first, source};
                        partial_edges[1] = {it2->first, terminal};
                    }
                }
            }
            if (min_dis == std::numeric_limits<double>::max()) {
                return paths;
            }
            paths.push_back(partial_edges[0]);
            paths.push_back(partial_edges[1]);

            vector<pair<int, int>> new_edges;
            new_edges = graph_v_of_v_idealID_two_hop_v2_extract_shortest_path_st_no_R1(L2, reduction_measures_2019R2, partial_edges[0].first, partial_edges[1].first, hop_cst - 2);
            if (new_edges.size() > 0) {
                for (int i = new_edges.size() - 1; i >= 0; i--) {
                    paths.push_back(new_edges[i]);
                }
            }
        } else {
            /*"Only source reduced"*/
            for (auto it1 = s_adj_begin; it1 != s_adj_end; it1++) {
                double x = graph_v_of_v_idealID_two_hop_v2_extract_distance_no_reduc(L2, it1->first, terminal, hop_cst - 1) + double(it1->second);
                if (min_dis > x) {
                    min_dis = x;
                    partial_edges[0] = {it1->first, source};
                }
            }
            if (min_dis == std::numeric_limits<double>::max()) {
                return paths;
            }
            paths.push_back(partial_edges[0]);

            vector<pair<int, int>> new_edges;
            new_edges = graph_v_of_v_idealID_two_hop_v2_extract_shortest_path_st_no_R1(L2, reduction_measures_2019R2, partial_edges[0].first, terminal, hop_cst - 1);
            if (new_edges.size() > 0) {
                for (int i = new_edges.size() - 1; i >= 0; i--) {
                    paths.push_back(new_edges[i]);
                }
            }
        }
    } else {
        /*"Only terminal reduced"*/
        if (reduction_measures_2019R2[terminal] == 2) {
            auto t_adj_begin = R2_reduced_vertices[terminal].begin();
            auto t_adj_end = R2_reduced_vertices[terminal].end();
            for (auto it2 = t_adj_begin; it2 != t_adj_end; it2++) {
                double x = graph_v_of_v_idealID_two_hop_v2_extract_distance_no_reduc(L2, source, it2->first, hop_cst - 1) + double(it2->second);
                if (min_dis > x) {
                    min_dis = x;
                    partial_edges[0] = {it2->first, terminal};
                }
            }
            if (min_dis == std::numeric_limits<double>::max()) {
                return paths;
            }
            paths.push_back(partial_edges[0]);
            vector<pair<int, int>> new_edges;
            new_edges = graph_v_of_v_idealID_two_hop_v2_extract_shortest_path_st_no_R1(L2, reduction_measures_2019R2, source, partial_edges[0].first, hop_cst - 1);
            if (new_edges.size() > 0) {
                for (int i = new_edges.size() - 1; i >= 0; i--) {
                    paths.push_back(new_edges[i]);
                }
            }
        } else {
            /* Nothing happened */
            /* In this case, the problem that the removed vertices appear in the path needs to be solved */
            int vector1_capped_v_parent = 0, vector2_capped_v_parent = 0;
            double distance = std::numeric_limits<double>::max();
            bool connected = false;
            auto vector1_check_pointer = L2[source].begin();
            auto vector2_check_pointer = L2[terminal].begin();
            auto pointer_L_s_end = L2[source].end(), pointer_L_t_end = L2[terminal].end();

            while (vector1_check_pointer != pointer_L_s_end && vector2_check_pointer != pointer_L_t_end) {
                if (vector1_check_pointer->vertex == vector2_check_pointer->vertex) {
                    auto ptr1 = vector1_check_pointer->dist_info.end() - 1;
                    auto ptr2 = vector2_check_pointer->dist_info.end() - 1;
                    auto begin1 = vector1_check_pointer->dist_info.begin();
                    auto begin2 = vector2_check_pointer->dist_info.begin();
                    while (1) {
                        if (get<0>(*ptr1) != -1) {
                            if (get<2>(*ptr1) + get<2>(*ptr2) <= hop_cst && get<0>(*ptr2) != -1) {
                                connected = true;
                                double dis = get<0>(*ptr1) + get<0>(*ptr2);
                                if (distance > dis) {
                                    distance = dis;
                                    vector1_capped_v_parent = get<1>(*ptr1);
                                    vector2_capped_v_parent = get<1>(*ptr2);
                                }
                                break;
                            }
                            if (ptr2 != begin2) {
                                ptr2--;
                                while (1) {
                                    if (get<0>(*ptr2) != -1) {
                                        if (get<2>(*ptr1) + get<2>(*ptr2) <= hop_cst) {
                                            connected = true;
                                            double dis = get<0>(*ptr1) + get<0>(*ptr2);
                                            if (distance > dis) {
                                                distance = dis;
                                                vector1_capped_v_parent = get<1>(*ptr1);
                                                vector2_capped_v_parent = get<1>(*ptr2);
                                            }
                                            break;
                                        }
                                    }
                                    if (ptr2 == begin2) {
                                        break;
                                    }
                                    ptr2--;
                                }
                            }
                        }
                        if (ptr1 == begin1) {
                            break;
                        }
                        ptr1--;
                        ptr2 = vector2_check_pointer->dist_info.end() - 1;
                    }
                    vector1_check_pointer++;
                } else if (vector1_check_pointer->vertex > vector2_check_pointer->vertex) {
                    vector2_check_pointer++;
                } else {
                    vector1_check_pointer++;
                }
            }

            if (connected) {
                if (source != vector1_capped_v_parent) {
                    paths.push_back({source, vector1_capped_v_parent});
                    source = vector1_capped_v_parent;
                    hop_cst--;
                }
                if (terminal != vector2_capped_v_parent) {
                    paths.push_back({terminal, vector2_capped_v_parent});
                    terminal = vector2_capped_v_parent;
                    hop_cst--;
                }
            } else {
                return paths;
            }

            vector<pair<int, int>> new_edges;
            new_edges = graph_v_of_v_idealID_two_hop_v2_extract_shortest_path_st_no_R1(L2, reduction_measures_2019R2, source, terminal, hop_cst);
            if (new_edges.size() > 0) {
                for (int i = new_edges.size() - 1; i >= 0; i--) {
                    paths.push_back(new_edges[i]);
                }
            }
        }
    }

    return paths;
}

/**
 *
 *
 *    for M
 */
double graph_v_of_v_idealID_two_hop_v2_extract_distance_no_reduc_for_M(vector<vector<two_hop_label_v2>> &L2, int source, int terminal, int hop_cst, int value_M) {
    /*return std::numeric_limits<double>::max() is not connected*/
    if (hop_cst < 0) {
        return std::numeric_limits<double>::max();
    }
    if (source == terminal) {
        return 0;
    } else if (hop_cst == 0) {
        return std::numeric_limits<double>::max();
    }

    double distance = std::numeric_limits<double>::max();
    auto vector1_check_pointer = L2[source].begin();
    auto vector2_check_pointer = L2[terminal].begin();
    auto pointer_L_s_end = L2[source].end(), pointer_L_t_end = L2[terminal].end();

    while (vector1_check_pointer != pointer_L_s_end && vector2_check_pointer != pointer_L_t_end) {
        if (vector1_check_pointer->vertex == vector2_check_pointer->vertex) {
            auto ptr1 = vector1_check_pointer->dist_info.end() - 1;
            auto ptr2 = vector2_check_pointer->dist_info.end() - 1;
            auto begin1 = vector1_check_pointer->dist_info.begin();
            auto begin2 = vector2_check_pointer->dist_info.begin();
            while (1) {
                if (get<0>(*ptr1) != -1) {
                    int hop_sum = int(get<0>(*ptr1) / value_M) + int(get<0>(*ptr2) / value_M);
                    if (hop_sum <= hop_cst && get<0>(*ptr2) != -1) {
                        double dis = get<0>(*ptr1) + get<0>(*ptr2) - hop_sum * value_M;
                        if (distance > dis) {
                            distance = dis;
                        }
                        break;
                    }
                    if (ptr2 != begin2) {
                        ptr2--;
                        while (1) {
                            if (get<0>(*ptr2) != -1) {
                                hop_sum = int(get<0>(*ptr1) / value_M) + int(get<0>(*ptr2) / value_M);
                                if (hop_sum <= hop_cst) {
                                    double dis = get<0>(*ptr1) + get<0>(*ptr2) - hop_sum * value_M;
                                    if (distance > dis) {
                                        distance = dis;
                                    }
                                    break;
                                }
                            }
                            if (ptr2 == begin2) {
                                break;
                            }
                            ptr2--;
                        }
                    }
                }
                if (ptr1 == begin1) {
                    break;
                }
                ptr1--;
                ptr2 = vector2_check_pointer->dist_info.end() - 1;
            }
            vector1_check_pointer++;
        } else if (vector1_check_pointer->vertex > vector2_check_pointer->vertex) {
            vector2_check_pointer++;
        } else {
            vector1_check_pointer++;
        }
    }

    return distance;
}

double graph_v_of_v_idealID_two_hop_v2_extract_distance_st_no_R1_for_M(vector<vector<two_hop_label_v2>> &L, vector<int> &reduction_measures_2019R2, int source, int terminal, int hop_cst, int value_M) {
    if (hop_cst < 0) {
        return std::numeric_limits<double>::max();
    }
    if (source == terminal) {
        return 0;
    } else if (hop_cst == 0) {
        return std::numeric_limits<double>::max();
    }

    double min_selected_distance = std::numeric_limits<double>::max();

    if (reduction_measures_2019R2[source] == 2) {
        if (reduction_measures_2019R2[terminal] == 2) {
            /*"Both reduced"*/
            // cout << "case 1" << endl;
            auto s_adj_begin = R2_reduced_vertices[source].begin();
            auto s_adj_end = R2_reduced_vertices[source].end();
            auto t_adj_begin = R2_reduced_vertices[terminal].begin();
            auto t_adj_end = R2_reduced_vertices[terminal].end();
            for (auto it1 = s_adj_begin; it1 != s_adj_end; it1++) {
                for (auto it2 = t_adj_begin; it2 != t_adj_end; it2++) {
                    double x = graph_v_of_v_idealID_two_hop_v2_extract_distance_no_reduc_for_M(L, it1->first, it2->first, hop_cst - 2, value_M);
                    if (x == std::numeric_limits<double>::max()) {
                        continue;
                    } else {
                        // here do not need to minus value_m cuz the distance if recorded when reducing graph
                        min_selected_distance = min(min_selected_distance, x + double(it1->second) + double(it2->second));
                    }
                }
            }
        } else {
            /*"Only source reduced"*/
            // cout << "case 2" << endl;
            auto s_adj_begin = R2_reduced_vertices[source].begin();
            auto s_adj_end = R2_reduced_vertices[source].end();
            for (auto it1 = s_adj_begin; it1 != s_adj_end; it1++) {
                double x = graph_v_of_v_idealID_two_hop_v2_extract_distance_no_reduc_for_M(L, it1->first, terminal, hop_cst - 1, value_M);
//                cout << "adj:" << it1->first << "   dist:" << x << "   edge:" << it1->second << endl;
                if (x == std::numeric_limits<double>::max()) {
                    continue;
                } else {
                    min_selected_distance = min(min_selected_distance, x + double(it1->second));
                }
            }
        }
    } else {
        if (reduction_measures_2019R2[terminal] == 2) {
            /*"Only terminal reduced"*/
            // cout << "case 3" << endl;
            auto t_adj_begin = R2_reduced_vertices[terminal].begin();
            auto t_adj_end = R2_reduced_vertices[terminal].end();
            for (auto it2 = t_adj_begin; it2 != t_adj_end; it2++) {
                double x = graph_v_of_v_idealID_two_hop_v2_extract_distance_no_reduc_for_M(L, source, it2->first, hop_cst - 1, value_M);
                if (x == std::numeric_limits<double>::max()) {
                    continue;
                } else {
                    min_selected_distance = min(min_selected_distance, x + double(it2->second));
                }
            }
        } else {
            min_selected_distance = min(min_selected_distance, graph_v_of_v_idealID_two_hop_v2_extract_distance_no_reduc_for_M(L, source, terminal, hop_cst, value_M));
        }
    }

    return min_selected_distance;
}

vector<pair<int, int>> graph_v_of_v_idealID_two_hop_v2_extract_shortest_path_st_no_R1_for_M(vector<vector<two_hop_label_v2>> &L2, vector<int> &reduction_measures_2019R2, int source, int terminal, int hop_cst, int value_M) {
    vector<pair<int, int>> paths;
    if (source == terminal) {
        return paths;
    }

    double min_dis = std::numeric_limits<double>::max();
    vector<pair<int, int>> partial_edges(2);

    if (reduction_measures_2019R2[source] == 2) {
        auto s_adj_begin = R2_reduced_vertices[source].begin();
        auto s_adj_end = R2_reduced_vertices[source].end();
        if (reduction_measures_2019R2[terminal] == 2) {
            /*"Both reduced"*/
            auto t_adj_begin = R2_reduced_vertices[terminal].begin();
            auto t_adj_end = R2_reduced_vertices[terminal].end();
            for (auto it1 = s_adj_begin; it1 != s_adj_end; it1++) {
                for (auto it2 = t_adj_begin; it2 != t_adj_end; it2++) {
                    double x = graph_v_of_v_idealID_two_hop_v2_extract_distance_no_reduc_for_M(L2, it1->first, it2->first, hop_cst - 2, value_M) + double(it1->second) + double(it2->second);
                    if (min_dis > x) {
                        min_dis = x;
                        partial_edges[0] = {it1->first, source};
                        partial_edges[1] = {it2->first, terminal};
                    }
                }
            }
            if (min_dis == std::numeric_limits<double>::max()) {
                return paths;
            }
            paths.push_back(partial_edges[0]);
            paths.push_back(partial_edges[1]);

            vector<pair<int, int>> new_edges;
            new_edges = graph_v_of_v_idealID_two_hop_v2_extract_shortest_path_st_no_R1_for_M(L2, reduction_measures_2019R2, partial_edges[0].first, partial_edges[1].first, hop_cst - 2, value_M);
            if (new_edges.size() > 0) {
                for (int i = new_edges.size() - 1; i >= 0; i--) {
                    paths.push_back(new_edges[i]);
                }
            }
        } else {
            /*"Only source reduced"*/
            for (auto it1 = s_adj_begin; it1 != s_adj_end; it1++) {
                double x = graph_v_of_v_idealID_two_hop_v2_extract_distance_no_reduc_for_M(L2, it1->first, terminal, hop_cst - 1, value_M) + double(it1->second);
                if (min_dis > x) {
                    min_dis = x;
                    partial_edges[0] = {it1->first, source};
                }
            }
            if (min_dis == std::numeric_limits<double>::max()) {
                return paths;
            }
            paths.push_back(partial_edges[0]);
            vector<pair<int, int>> new_edges;
            new_edges = graph_v_of_v_idealID_two_hop_v2_extract_shortest_path_st_no_R1_for_M(L2, reduction_measures_2019R2, partial_edges[0].first, terminal, hop_cst - 1, value_M);
            if (new_edges.size() > 0) {
                for (int i = new_edges.size() - 1; i >= 0; i--) {
                    paths.push_back(new_edges[i]);
                }
            }
        }
    } else {
        /*"Only terminal reduced"*/
        if (reduction_measures_2019R2[terminal] == 2) {
            auto t_adj_begin = R2_reduced_vertices[terminal].begin();
            auto t_adj_end = R2_reduced_vertices[terminal].end();
            for (auto it2 = t_adj_begin; it2 != t_adj_end; it2++) {
                double x = graph_v_of_v_idealID_two_hop_v2_extract_distance_no_reduc_for_M(L2, source, it2->first, hop_cst - 1, value_M) + double(it2->second);
                if (min_dis > x) {
                    min_dis = x;
                    partial_edges[0] = {it2->first, terminal};
                }
            }
            if (min_dis == std::numeric_limits<double>::max()) {
                return paths;
            }
            paths.push_back(partial_edges[0]);
            vector<pair<int, int>> new_edges;
            new_edges = graph_v_of_v_idealID_two_hop_v2_extract_shortest_path_st_no_R1_for_M(L2, reduction_measures_2019R2, source, partial_edges[0].first, hop_cst - 1, value_M);
            if (new_edges.size() > 0) {
                for (int i = new_edges.size() - 1; i >= 0; i--) {
                    paths.push_back(new_edges[i]);
                }
            }
        } else {
            /* Nothing happened */
            /* In this case, the problem that the removed vertices appear in the path needs to be solved */
            int vector1_capped_v_parent = 0, vector2_capped_v_parent = 0;
            double distance = std::numeric_limits<double>::max();
            bool connected = false;
            auto vector1_check_pointer = L2[source].begin();
            auto vector2_check_pointer = L2[terminal].begin();
            auto pointer_L_s_end = L2[source].end(), pointer_L_t_end = L2[terminal].end();

            while (vector1_check_pointer != pointer_L_s_end && vector2_check_pointer != pointer_L_t_end) {
                if (vector1_check_pointer->vertex == vector2_check_pointer->vertex) {
                    auto ptr1 = vector1_check_pointer->dist_info.end() - 1;
                    auto ptr2 = vector2_check_pointer->dist_info.end() - 1;
                    auto begin1 = vector1_check_pointer->dist_info.begin();
                    auto begin2 = vector2_check_pointer->dist_info.begin();
                    while (1) {
                        if (get<0>(*ptr1) != -1) {
                            int hop_sum = int(get<0>(*ptr1) / value_M) + int(get<0>(*ptr2) / value_M);
                            if (hop_sum <= hop_cst && get<0>(*ptr2) != -1) {
                                connected = true;
                                double dis = get<0>(*ptr1) + get<0>(*ptr2) - hop_sum * value_M;
                                if (distance > dis) {
                                    distance = dis;
                                    vector1_capped_v_parent = get<1>(*ptr1);
                                    vector2_capped_v_parent = get<1>(*ptr2);
                                }
                                break;
                            }
                            if (ptr2 != begin2) {
                                ptr2--;
                                while (1) {
                                    if (get<0>(*ptr2) != -1) {
                                        hop_sum = int(get<0>(*ptr1) / value_M) + int(get<0>(*ptr2) / value_M);
                                        if (hop_sum <= hop_cst) {
                                            connected = true;
                                            double dis = get<0>(*ptr1) + get<0>(*ptr2) - hop_sum * value_M;
                                            if (distance > dis) {
                                                distance = dis;
                                                vector1_capped_v_parent = get<1>(*ptr1);
                                                vector2_capped_v_parent = get<1>(*ptr2);
                                            }
                                            break;
                                        }
                                    }
                                    if (ptr2 == begin2) {
                                        break;
                                    }
                                    ptr2--;
                                }
                            }
                        }
                        if (ptr1 == begin1) {
                            break;
                        }
                        ptr1--;
                        ptr2 = vector2_check_pointer->dist_info.end() - 1;
                    }
                    vector1_check_pointer++;
                } else if (vector1_check_pointer->vertex > vector2_check_pointer->vertex) {
                    vector2_check_pointer++;
                } else {
                    vector1_check_pointer++;
                }
            }

            if (connected) {
                if (source != vector1_capped_v_parent) {
                    paths.push_back({source, vector1_capped_v_parent});
                    source = vector1_capped_v_parent;
                    hop_cst--;
                }
                if (terminal != vector2_capped_v_parent) {
                    paths.push_back({terminal, vector2_capped_v_parent});
                    terminal = vector2_capped_v_parent;
                    hop_cst--;
                }
            } else {
                return paths;
            }

            vector<pair<int, int>> new_edges;
            new_edges = graph_v_of_v_idealID_two_hop_v2_extract_shortest_path_st_no_R1_for_M(L2, reduction_measures_2019R2, source, terminal, hop_cst, value_M);
            if (new_edges.size() > 0) {
                for (int i = new_edges.size() - 1; i >= 0; i--) {
                    paths.push_back(new_edges[i]);
                }
            }
        }
    }

    return paths;
}