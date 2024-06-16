#pragma once
#include <graph_v_of_v_idealID/graph_v_of_v_idealID.h>

graph_v_of_v_idealID graph_v_of_v_idealID_change_new_vertexIDs(graph_v_of_v_idealID& input_graph, std::vector<int>& vertexID_old_to_new) {

	int N = input_graph.size();

	graph_v_of_v_idealID output_graph(N);

	for (int i = 0; i < N; i++) {
		int v_size = input_graph[i].size();
		for (int j = 0; j < v_size; j++) {
			if (i < input_graph[i][j].first) {
				graph_v_of_v_idealID_add_edge(output_graph, vertexID_old_to_new[i], vertexID_old_to_new[input_graph[i][j].first], input_graph[i][j].second);
			}
		}
	}

	return output_graph;
}
