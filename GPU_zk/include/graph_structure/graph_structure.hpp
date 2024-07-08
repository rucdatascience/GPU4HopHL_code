#pragma once

/*the following codes are for testing

---------------------------------------------------
a cpp file (try.cpp) for running the following example code:
----------------------------------------

#include <iostream>
#include <fstream>
using namespace std;

#include <graph_structure/graph_structure.h>


int main()
{
	graph_structure_example();
}

------------------------------------------------------------------------------------------
Commends for running the above cpp file on Linux:

g++ -std=c++17 -I/home/boost_1_75_0 -I/root/CPU_GPU_project try.cpp -lpthread -O3 -o A
./A
rm A

(optional to put the above commends in run.sh, and then use the comment: sh run.sh)


*/

#include <vector>
#include <iostream>
#include <string>
#include <vector>
#include <cstring>
#include <fstream>
#include <unordered_map>
#include "parse_string.hpp"
#include "sorted_vector_binary_operations.hpp"

template <typename weight_type>
class CSR_graph;

template <typename weight_type> // weight_type may be int, long long int, float, double...
class graph_structure {
public:
	/*
	this class is for directed and edge-weighted graph
	*/

	int V = 0; // the number of vertices
	long long E = 0; // the number of edges

	// OUTs[u] = v means there is an edge starting from u to v
	std::vector<std::vector<std::pair<int, weight_type>>> OUTs;
	// INs is transpose of OUTs. INs[u] = v means there is an edge starting from v to u
	std::vector<std::vector<std::pair<int, weight_type>>> INs;

	std::vector<int> vertex_degree; // vertex_degree[i] = out_degree(i)
	std::vector<int> sorted_vertices; // sorted by out_degree, descending

	/*constructors*/
	graph_structure() {}
	graph_structure(int n) {
		V = n;
		OUTs.resize(n); // initialize n vertices
		INs.resize(n);
	}
	int size() {
		return V;
	}

	/*class member functions*/
	inline void add_edge(int, int, weight_type); // this function can change edge weights
	inline void remove_edge(int, int);
	inline void remove_all_adjacent_edges(int);
	inline bool contain_edge(int, int); // whether there is an edge
	inline weight_type edge_weight(int, int);
	inline long long int edge_number(); // the total number of edges
	inline void read(std::string);
	inline void print();
	inline void clear();
	inline int out_degree(int);
	inline int in_degree(int);
	inline CSR_graph<weight_type> toCSR();

	int add_vertice(std::string);
	void add_edge(std::string, std::string, weight_type);
};

/*for GPU*/

template <typename weight_type>
class CSR_graph {
	public:
		std::vector<int> INs_Neighbor_start_pointers, OUTs_Neighbor_start_pointers; // Neighbor_start_pointers[i] is the start point of neighbor information of vertex i in Edges and Edge_weights
		/*
			Now, Neighbor_sizes[i] = Neighbor_start_pointers[i + 1] - Neighbor_start_pointers[i].
			And Neighbor_start_pointers[V] = Edges.size() = Edge_weights.size() = the total number of edges.
		*/
		std::vector<int> INs_Edges, OUTs_Edges;  // Edges[Neighbor_start_pointers[i]] is the start of Neighbor_sizes[i] neighbor IDs
		std::vector<weight_type> INs_Edge_weights, OUTs_Edge_weights; // Edge_weights[Neighbor_start_pointers[i]] is the start of Neighbor_sizes[i] edge weights
};


template <typename weight_type>
void graph_structure<weight_type>::add_edge(int e1, int e2, weight_type ec) {

	/*we assume that the size of g is larger than e1 or e2;
	 this function can update edge weight; there will be no redundent edge*/

	 /*
	 Add the edges (e1,e2) with the weight ec
	 When the edge exists, it will update its weight.
	 Time complexity:
		 O(log n) When edge already exists in graph
		 O(n) When edge doesn't exist in graph
	 */

	sorted_vector_binary_operations_insert(OUTs[e1], e2, ec);
	sorted_vector_binary_operations_insert(INs[e2], e1, ec);

	if (!is_directed) {
		sorted_vector_binary_operations_insert(OUTs[e2], e1, ec);
		sorted_vector_binary_operations_insert(INs[e1], e2, ec);
	}
}

template <typename weight_type>
void graph_structure<weight_type>::remove_edge(int e1, int e2) {

	/*we assume that the size of g is larger than e1 or e2*/
	/*
	 Remove the edges (e1,e2)
	 If the edge does not exist, it will do nothing.
	 Time complexity: O(n)
	*/

	sorted_vector_binary_operations_erase(OUTs[e1], e2);
	sorted_vector_binary_operations_erase(INs[e2], e1);

	if (!is_directed) {
		sorted_vector_binary_operations_erase(OUTs[e2], e1);
		sorted_vector_binary_operations_erase(INs[e1], e2);
	}
}

template <typename weight_type>
void graph_structure<weight_type>::remove_all_adjacent_edges(int v) {
	for (auto it = OUTs[v].begin(); it != OUTs[v].end(); it++)
		sorted_vector_binary_operations_erase(INs[it->first], v);

	for (auto it = INs[v].begin(); it != INs[v].end(); it++)
		sorted_vector_binary_operations_erase(OUTs[it->first], v);

	std::vector<std::pair<int, weight_type>>().swap(OUTs[v]);
	std::vector<std::pair<int, weight_type>>().swap(INs[v]);
}

template <typename weight_type>
bool graph_structure<weight_type>::contain_edge(int e1, int e2) {

	/*
	Return true if graph contain edge (e1,e2)
	Time complexity: O(logn)
	*/

	return sorted_vector_binary_operations_search(OUTs[e1], e2);
}

template <typename weight_type>
weight_type graph_structure<weight_type>::edge_weight(int e1, int e2) {

	/*
	Return the weight of edge (e1,e2)
	If the edge does not exist, return std::numeric_limits<double>::max()
	Time complexity: O(logn)
	*/

	return sorted_vector_binary_operations_search_weight(OUTs[e1], e2);
}

template <typename weight_type>
long long int graph_structure<weight_type>::edge_number() {

	/*
	Returns the number of edges in the figure
	Time complexity: O(n)
	*/

	long long int num = 0;
	for (auto it : OUTs)
		num = num + it.size();

	return is_directed ? num : num / 2;
}

template <typename weight_type>
void graph_structure<weight_type>::clear() {
	std::vector<std::vector<std::pair<int, weight_type>>>().swap(OUTs);
	std::vector<std::vector<std::pair<int, weight_type>>>().swap(INs);
}

template <typename weight_type>
int graph_structure<weight_type>::out_degree(int v) {
	return OUTs[v].size();
}

template <typename weight_type>
int graph_structure<weight_type>::in_degree(int v) {
	return INs[v].size();
}

template <typename weight_type>
void graph_structure<weight_type>::read(std::string file_name) {

	/*
	Read the graph from the file
	Each line of the file is in the format "u v w" which means there is an edge from u to v with weight w
	*/

	std::ifstream file(file_name);
	std::string line;
	
	std::getline(file, line);
	std::vector<std::string> Parsed_content = parse_string(line, " ");
	if (Parsed_content.size() != 2) {
		std::cout << "Format error in the first line of the file" << std::endl;
		return;
	}
	V = std::stoi(Parsed_content[1]);
	OUTs.resize(V);
	INs.resize(V);
	std::getline(file, line);
	Parsed_content = parse_string(line, " ");
	if (Parsed_content.size() != 2) {
		std::cout << "Format error in the second line of the file" << std::endl;
		return;
	}
	E = std::stoi(Parsed_content[1]);

	// initalize the vertex_degree
	vertex_degree.resize(V, 0);

	while (std::getline(file, line)) {
		Parsed_content = parse_string(line, " ");
		if (Parsed_content.size() != 4) {
			std::cout << "Format error in the file" << std::endl;
			return;
		}
		int u = std::stoi(Parsed_content[1]);
		int v = std::stoi(Parsed_content[2]);
		weight_type w = std::stod(Parsed_content[3]);
		add_edge(u, v, w);
		vertex_degree[u]++;
	}

	// sort the vertices by out_degree
	sorted_vertices.resize(V);
	for (int i = 0; i < V; i++) {
		sorted_vertices[i] = i;
	}
	std::sort(sorted_vertices.begin(), sorted_vertices.end(), [&](int a, int b) {
		return vertex_degree[a] > vertex_degree[b];
	});

	file.close();

	return;
}


template <typename weight_type>
void graph_structure<weight_type>::print() {

	std::cout << "graph_structure_print:" << std::endl;

	for (int i = 0; i < V; i++) {
		std::cout << "Vertex " << i << " OUTs List: ";
		int v_size = OUTs[i].size();
		for (int j = 0; j < v_size; j++) {
			std::cout << "<" << OUTs[i][j].first << "," << OUTs[i][j].second << "> ";
		}
		std::cout << std::endl;
	}
	std::cout << "graph_structure_print END" << std::endl;

}

/*for GPU*/

template <typename weight_type>
CSR_graph<weight_type> graph_structure<weight_type>::toCSR() {

	CSR_graph<weight_type> ARRAY;

	int V = OUTs.size();
	ARRAY.INs_Neighbor_start_pointers.resize(V + 1); // Neighbor_start_pointers[V] = Edges.size() = Edge_weights.size() = the total number of edges.
	ARRAY.OUTs_Neighbor_start_pointers.resize(V + 1);

	int pointer = 0;
	for (int i = 0; i < V; i++) {
		ARRAY.INs_Neighbor_start_pointers[i] = pointer;
		for (auto& xx : INs[i]) {
			ARRAY.INs_Edges.push_back(xx.first);
			ARRAY.INs_Edge_weights.push_back(xx.second);
		}
		pointer += INs[i].size();
	}
	ARRAY.INs_Neighbor_start_pointers[V] = pointer;

	pointer = 0;
	for (int i = 0; i < V; i++) {
		ARRAY.OUTs_Neighbor_start_pointers[i] = pointer;
		for (auto& xx : OUTs[i]) {
			ARRAY.OUTs_Edges.push_back(xx.first);
			ARRAY.OUTs_Edge_weights.push_back(xx.second);
		}
		pointer += OUTs[i].size();
	}
	ARRAY.OUTs_Neighbor_start_pointers[V] = pointer;

	return ARRAY;
}

inline void graph_structure_example() {

	/*
	Create a complete graph of 10 nodes
	Weight of edge (u,v) and (v,u) equal to min(u,v)+max(u,v)*0.1
	*/
	using std::cout;
	int N = 10;
	graph_structure<float> g(N);

	/*
	Insert the edge
	When the edge exists, it will update its weight.
	*/
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < i; j++) {
			g.add_edge(i, j, j + 0.1 * i); // Insert the edge(i,j) with value j+0.1*i
		}
	}

	/*
	Get the number of edges, (u,v) and (v,u) only be counted once
	The output is 45 (10*9/2)
	*/
	std::cout << g.edge_number() << '\n';

	/*
	Check if graph contain the edge (3,1) and get its value
	The output is 1 1.3
	*/
	std::cout << g.contain_edge(3, 1) << " " << g.edge_weight(3, 1) << '\n';

	/*
	Remove half of the edge
	If the edge does not exist, it will do nothing.
	*/
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < i; j++) {
			if ((i + j) % 2 == 1)
				g.remove_edge(i, j);
		}
	}

	/*
	Now the number of edges is 20
	*/
	std::cout << g.edge_number() << '\n';;

	/*
	Now the graph no longer contain the edge (3,0) and its value become std::numeric_limits<double>::max()
	*/
	std::cout << g.contain_edge(3, 0) << " " << g.edge_weight(3, 0) << '\n';

	g.print(); // print the graph

	g.remove_all_adjacent_edges(1);

	g.print(); // print the graph

	std::cout << "g.size()= " << g.size() << '\n';
}
