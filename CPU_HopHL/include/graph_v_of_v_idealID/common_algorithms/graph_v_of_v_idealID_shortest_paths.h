
#pragma once
#include <vector>
#include <numeric>
#include <iostream>
#include <unordered_map>
#include <boost/heap/fibonacci_heap.hpp> 
#include <graph_v_of_v_idealID/graph_v_of_v_idealID.h>

using namespace std;


struct graph_v_of_v_idealID_node_for_sp {
	int index;
	double priority_value;
}; // define the node in the queue
bool operator<(graph_v_of_v_idealID_node_for_sp const& x, graph_v_of_v_idealID_node_for_sp const& y) {
	return x.priority_value > y.priority_value; // < is the max-heap; > is the min heap
}
typedef typename boost::heap::fibonacci_heap<graph_v_of_v_idealID_node_for_sp>::handle_type handle_t_for_graph_v_of_v_idealID_sp;


template<typename T> // T is float or double
void graph_v_of_v_idealID_shortest_paths(graph_v_of_v_idealID& input_graph, int source, std::vector<T>& distances, std::vector<int>& predecessors) {

	/*Dijkstra��s shortest path algorithm: https://www.geeksforgeeks.org/dijkstras-shortest-path-algorithm-greedy-algo-7/
	time complexity: O(|E|+|V|log|V|);
	the output distances and predecessors only contain vertices connected to source*/

	T inf = std::numeric_limits<T>::max();

	int N = input_graph.size();
	distances.resize(N, inf); // initial distance from source is inf
	predecessors.resize(N);
	std::iota(std::begin(predecessors), std::end(predecessors), 0); // initial predecessor of each vertex is itself

	graph_v_of_v_idealID_node_for_sp node;
	boost::heap::fibonacci_heap<graph_v_of_v_idealID_node_for_sp> Q;
	std::vector<T> Q_keys(N, inf); // if the key of a vertex is inf, then it is not in Q yet
	std::vector<handle_t_for_graph_v_of_v_idealID_sp> Q_handles(N);

	/*initialize the source*/
	Q_keys[source] = 0;
	node.index = source;
	node.priority_value = 0;
	Q_handles[source] = Q.push(node);

	/*time complexity: O(|E|+|V|log|V|) based on fibonacci_heap, not on pairing_heap, which is O((|E|+|V|)log|V|)*/
	while (Q.size() > 0) {

		int top_v = Q.top().index;
		T top_key = Q.top().priority_value;

		Q.pop();

		distances[top_v] = top_key; // top_v is touched

		for (auto it = input_graph[top_v].begin(); it != input_graph[top_v].end(); it++) {
			int adj_v = it->first;
			T ec = it->second;
			if (Q_keys[adj_v] == inf) { // adj_v is not in Q yet
				Q_keys[adj_v] = top_key + ec;
				node.index = adj_v;
				node.priority_value = Q_keys[adj_v];
				Q_handles[adj_v] = Q.push(node);
				predecessors[adj_v] = top_v;
			}
			else { // adj_v is in Q
				if (Q_keys[adj_v] > top_key + ec) { // needs to update key
					Q_keys[adj_v] = top_key + ec;
					node.index = adj_v;
					node.priority_value = Q_keys[adj_v];
					Q.update(Q_handles[adj_v], node);
					predecessors[adj_v] = top_v;
				}
			}
		}

	}

}






/*debug codes*/
#include <graph_v_of_v_idealID/random_graph/graph_v_of_v_idealID_generate_random_connected_graph.h>
#include <graph_v_of_v_idealID/read_save/graph_v_of_v_idealID_read_for_GSTP.h>
#include <graph_v_of_v_idealID/read_save/graph_v_of_v_idealID_save_for_GSTP.h>
#include <graph_hash_of_mixed_weighted/read_save/graph_hash_of_mixed_weighted_read_for_GSTP.h>
#include <graph_hash_of_mixed_weighted/common_algorithms/graph_hash_of_mixed_weighted_shortest_paths.h>

#include <boost/random.hpp>
boost::random::mt19937 boost_random_time_seed_test_graph_v_of_v_idealID_shortest_paths{ static_cast<std::uint32_t>(std::time(0)) };

void test_graph_v_of_v_idealID_shortest_paths() {

	/*parameters*/
	int iteration_times = 100;
	int V = 100, E = 500, precision = 3;
	double ec_min = 1, ec_max = 10;

	/*iteration*/
	std::time_t now = std::time(0);
	boost::random::mt19937 gen{ static_cast<std::uint32_t>(now) };
	for (int i = 0; i < iteration_times; i++) {

		std::cout << "Iteration " << i << std::endl;

		/*input and output*/
		int generate_new_graph = 1;

		graph_hash_of_mixed_weighted old_hash_graph, old_hash_generated_group_graph;;
		graph_v_of_v_idealID instance_graph;
		if (generate_new_graph == 1) {
			instance_graph = graph_v_of_v_idealID_generate_random_connected_graph(V, E, ec_min, ec_max, precision, boost_random_time_seed_test_graph_v_of_v_idealID_shortest_paths);
			std::unordered_set<int> generated_group_vertices;
			graph_v_of_v_idealID generated_group_graph;
			graph_v_of_v_idealID_save_for_GSTP("simple_iterative_tests.txt", instance_graph, generated_group_graph, generated_group_vertices);
			double lambda;
			graph_hash_of_mixed_weighted_read_for_GSTP("simple_iterative_tests.txt", old_hash_graph,
				old_hash_generated_group_graph, generated_group_vertices, lambda);
		}
		else {
			std::unordered_set<int> generated_group_vertices;
			graph_v_of_v_idealID generated_group_graph;
			graph_v_of_v_idealID_read_for_GSTP("simple_iterative_tests.txt", instance_graph, generated_group_graph, generated_group_vertices);
			double lambda;
			graph_hash_of_mixed_weighted_read_for_GSTP("simple_iterative_tests.txt", old_hash_graph,
				old_hash_generated_group_graph, generated_group_vertices, lambda);
		}

		boost::random::uniform_int_distribution<> dist{ static_cast<int>(0), static_cast<int>(V - 1) };
		int source = dist(gen);

		std::vector<double> distances;
		std::vector<int> predecessors;
		graph_v_of_v_idealID_shortest_paths(instance_graph, source, distances, predecessors);


		std::unordered_map<int, double> distances_hash;
		std::unordered_map<int, int> predecessors_hash;
		graph_hash_of_mixed_weighted_shortest_paths_source_to_all(old_hash_graph, source, distances_hash, predecessors_hash);


		for (int xx = 0; xx < 10; xx++) {
			int terminal = dist(gen);

			if (abs(distances[terminal] - distances_hash[terminal]) > 1e-5) {
				std::cout << "abs(distances[terminal] - distances_hash[terminal]) > 1e-5" << std::endl;
				std::cout << "source = " << source << std::endl;
				std::cout << "terminal = " << terminal << std::endl;
				std::cout << "distances[terminal] = " << distances[terminal] << std::endl;
				std::cout << "distances_hash[terminal] = " << distances_hash[terminal] << std::endl;
				getchar();
			}

			if (predecessors[terminal] != predecessors_hash[terminal]) {
				std::cout << "predecessors[terminal] != predecessors[terminal]" << std::endl;
				std::cout << "source = " << source << std::endl;
				std::cout << "terminal = " << terminal << std::endl;
				std::cout << "predecessors[terminal] = " << predecessors[terminal] << std::endl;
				std::cout << "predecessors_hash[terminal] = " << predecessors_hash[terminal] << std::endl;
				getchar();
			}

		}
	}
}

