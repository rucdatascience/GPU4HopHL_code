#pragma once
#include <list>
#include <queue>
#include <vector>
#include <graph_v_of_v_idealID/graph_v_of_v_idealID.h>

std::list<std::list<int>> graph_v_of_v_idealID_connected_components(graph_v_of_v_idealID& input_graph) {

	/*this is to find connected_components using breadth first search; time complexity O(|V|+|E|);
	related content: https://www.boost.org/doc/libs/1_68_0/boost/graph/connected_components.hpp
	https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm*/

	std::list<std::list<int>> components;

	/*time complexity: O(V)*/
	int N = input_graph.size();
	std::vector<bool> discovered(N, false);

	for (int i = 0; i < N; i++) {

		if (discovered[i] == false) {

			std::list<int> component;
			/*below is a depth first search; time complexity O(|V|+|E|)*/
			std::queue<int> Q; // Queue is a data structure designed to operate in FIFO (First in First out) context.
			Q.push(i);
			component.push_back(i);
			discovered[i] = true;
			while (Q.size() > 0) {
				int v = Q.front();
				Q.pop(); //Removing that vertex from queue,whose neighbour will be visited now

				int adj_size = input_graph[v].size();
				for (int j = 0; j < adj_size; j++) {
					int adj_v = input_graph[v][j].first;
					if (discovered[adj_v] == false) {
						Q.push(adj_v);
						component.push_back(adj_v);
						discovered[adj_v] = true;
					}
				}
			}

			components.push_back(component);

		}
	}

	return components;

}








/*
------------
#include<iostream>
#include<text_mining/print_items.h>
#include<graph_v_of_v_idealID/common_algorithms/graph_v_of_v_idealID_connected_components.h>

int main()
{
	graph_v_of_v_idealID g(5); // a graph with 5 vertices: 0, 1, 2, 3, 4, 5
	graph_v_of_v_idealID_add_edge(g, 0, 1, 0.3); // add edge (0,1) with the weight of 0.3
	graph_v_of_v_idealID_add_edge(g, 3, 4, 0.5); // add edge (1,2) with the weight of 0.5
	std::list<std::list<int>> cpns = graph_v_of_v_idealID_connected_components(g); // find the connected components of g
	for (auto it = cpns.begin(); it != cpns.end(); it++) {
		print_a_sequence_of_elements(*it);
	}
}
-------------
*/