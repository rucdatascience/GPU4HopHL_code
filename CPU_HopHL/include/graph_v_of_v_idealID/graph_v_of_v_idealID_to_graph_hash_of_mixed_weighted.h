#pragma once

#include <graph_hash_of_mixed_weighted/graph_hash_of_mixed_weighted.h>
#include <graph_v_of_v_idealID/graph_v_of_v_idealID.h>


graph_hash_of_mixed_weighted graph_v_of_v_idealID_to_graph_hash_of_mixed_weighted(graph_v_of_v_idealID& ideal_g) {

	graph_hash_of_mixed_weighted hash_g;

	int N = ideal_g.size();
	for (int i = 0; i < N; i++) {
		graph_hash_of_mixed_weighted_add_vertex(hash_g, i, 0);
		for (auto adj : ideal_g[i]) {
			if (i < adj.first) {
				graph_hash_of_mixed_weighted_add_edge(hash_g, i, adj.first, adj.second);
			}
		}
	}

	return hash_g;
}





