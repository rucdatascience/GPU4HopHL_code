#pragma once
#include<queue>
#include<graph_v_of_v_idealID/graph_v_of_v_idealID.h>

std::vector<int> graph_v_of_v_idealID_breadth_first_search_a_set_of_vertices(graph_v_of_v_idealID& input_graph, int root) {

	int N = input_graph.size();

	std::vector<int> root_component; // v is connected to root; including root

	std::vector<bool> unprocessed(N, true); // mark non-root as un-discovered
	unprocessed[root] = false; // mark root as discovered

	std::queue<int> Q; // Queue is a data structure designed to operate in FIFO (First in First out) context.
	Q.push(root);
	while (Q.size() > 0) {
		int v = Q.front();
		root_component.push_back(v); // v is connected to root; including root
		Q.pop(); //Removing that vertex from queue,whose neighbour will be visited now

		/*processing all the neighbours of v*/
		for (auto it = input_graph[v].begin(); it != input_graph[v].end(); it++) {
			int vertex = it->first;
			if (unprocessed[vertex]) { // vertex has not been discovered
				Q.push(vertex);
				unprocessed[vertex] = false;  // vertex has been discovered
			}
		}

	}

	return root_component;

}
