#pragma once
#include <algorithm>
#include <graph_v_of_v_idealID/graph_v_of_v_idealID.h>
#include <graph_v_of_v_idealID/graph_v_of_v_idealID_change_new_vertexIDs.h>
#include <vector>
using namespace std;

graph_v_of_v_idealID graph_v_of_v_idealID_sort(graph_v_of_v_idealID& input_graph){
    vector<int> new2old(input_graph.size(), 0);
    vector<int> old2new(input_graph.size(), 0);
    for (int i = 0 ; i != new2old.size() ; i++) {
        new2old[i] = i;
    }
    sort(new2old.begin(), new2old.end(), [&](const int& a, const int& b){return input_graph[a].size() > input_graph[b].size();});
    for (int i = 0; i < old2new.size(); i++)
    {
        old2new[new2old[i]] = i;
    }
    graph_v_of_v_idealID output_graph(input_graph.size());
    output_graph = graph_v_of_v_idealID_change_new_vertexIDs(input_graph, old2new);
    return output_graph;
}

bool graph_v_of_v_idealID_check_sorted(graph_v_of_v_idealID& input_graph){
    for (auto it = input_graph.begin()+1; it != input_graph.end(); it++){
        if (it->size() > (it-1)->size())
            return false;
    }
    return true;
}