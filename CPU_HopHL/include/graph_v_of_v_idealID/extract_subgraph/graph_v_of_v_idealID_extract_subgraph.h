#pragma once
#include<unordered_set>
#include<graph_v_of_v_idealID/graph_v_of_v_idealID.h>
#include<graph_hash_of_mixed_weighted/graph_hash_of_mixed_weighted.h>

graph_hash_of_mixed_weighted graph_v_of_v_idealID_extract_subgraph(graph_v_of_v_idealID& input_graph, std::unordered_set<int>& V_set) {

	/*extract a smaller_graph, which contains all the vertices in V_set,
	and all the edges between vertices in V_set;
	time complexity O(|V_list|+|adj_v of V_list in input_graph|)*/

	/*time complexity O(|V_list|+|adj_v of V_list in input_graph|)*/
	graph_hash_of_mixed_weighted smaller_graph;
	for (auto it = V_set.begin(); it != V_set.end(); it++) {
		int v1 = *it;
		graph_hash_of_mixed_weighted_add_vertex(smaller_graph, v1, 0); // add vertex

		for (int j = input_graph[v1].size() - 1; j >= 0; j--) {
			int v2 = input_graph[v1][j].first;
			if (V_set.count(v2) > 0 && v2 > v1) { // v2 is in the list and only add edge once
				graph_hash_of_mixed_weighted_add_edge(smaller_graph, v1, v2, input_graph[v1][j].second); // add edge
			}
		}
	}

	return smaller_graph;

}
