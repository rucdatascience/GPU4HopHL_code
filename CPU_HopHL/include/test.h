#pragma once

/*the following codes are for testing

---------------------------------------------------
a cpp file (try.cpp) for running the following test code:
----------------------------------------

#include <iostream>
#include <fstream>
using namespace std;

// header files in the Boost library: https://www.boost.org/
#include <boost/random.hpp>
boost::random::mt19937 boost_random_time_seed{ static_cast<std::uint32_t>(std::time(0)) };

#include <build_in_progress/HL/HL4GST/test.h>


int main()
{
	test_PLL();
}

------------------------------------------------------------------------------------------
Commends for running the above cpp file on Linux:

g++ -std=c++17 -I/home/boost_1_75_0 -I/root/rucgraph try.cpp -lpthread -Ofast -o A
./A
rm A

(optional to put the above commends in run.sh, and then use the comment: sh run.sh)


*/
#include <graph_v_of_v/graph_v_of_v.h>
#include <graph_v_of_v/graph_v_of_v_update_vertexIDs_by_degrees_large_to_small.h>
#include <graph_v_of_v/graph_v_of_v_generate_random_graph.h>
#include <graph_v_of_v/graph_v_of_v_shortest_paths.h>
#include <build_in_progress/HL/HL4GST/two_hop_labels.h>
#include <build_in_progress/HL/HL4GST/PLL.h>
#include <build_in_progress/HL/HL4GST/CT.h>
#include <graph_v_of_v/graph_v_of_v_hop_constrained_shortest_distance.h>
#include <build_in_progress/HL/HL4GST/hop_constrained_two_hop_labels_generation.h>
#include <build_in_progress/HL/HL4GST/GST.h>
#include <text_mining/print_items.h>
#include <text_mining/binary_save_read_vector.h>



void add_vertex_groups(graph_v_of_v<int>& instance_graph, int group_num) {

	double dummy_edge_probability = 0.2;
	boost::random::uniform_int_distribution<> dist{ static_cast<int>(1), static_cast<int>(100) };

	int N = instance_graph.size();

	instance_graph.ADJs.resize(N + group_num);
	for (int i = N; i < N + group_num; i++) {
		for (int j = 0; j < N; j++) {
			if ((double)dist(boost_random_time_seed) / 100 < dummy_edge_probability) {
				instance_graph.add_edge(i, j, 1e6); // add a dummy edge
			}
		}
	}

}

void test_PLL_check_correctness(two_hop_case_info& case_info, graph_v_of_v<int>& instance_graph, int iteration_source_times, int iteration_terminal_times) {

	/*
	below is for checking whether the above labels are right (by randomly computing shortest paths)

	this function can only be used when 0 to n-1 is in the graph, i.e., the graph is an ideal graph
	*/

	boost::random::uniform_int_distribution<> dist{ static_cast<int>(0), static_cast<int>(instance_graph.ADJs.size() - 1) };

	//graph_hash_of_mixed_weighted_print(instance_graph);

	for (int yy = 0; yy < iteration_source_times; yy++) {
		int source = dist(boost_random_time_seed);

		while (!is_mock[source]) {
			source = dist(boost_random_time_seed);
		}

		std::vector<int> distances, predecessors;

		//source = 0; cout << "source = " << source << endl;

		graph_v_of_v_shortest_paths<int>(instance_graph, source, distances, predecessors);

		for (int xx = 0; xx < iteration_terminal_times; xx++) {

			int terminal = dist(boost_random_time_seed);

			while (is_mock[terminal]) {
				terminal = dist(boost_random_time_seed);
			}

			//terminal = 2; cout << "terminal = " << terminal << endl;

			double dis = extract_distance(case_info, source, terminal);

			if (abs(dis - distances[terminal]) > 1e-4 && (dis < TwoM_value || distances[terminal] < TwoM_value)) {
				cout << "source = " << source << endl;
				cout << "terminal = " << terminal << endl;
				cout << "source vector:" << endl;
				for (auto it = case_info.L[source].begin(); it != case_info.L[source].end(); it++) {
					cout << "<" << it->vertex << "," << it->distance << "," << it->parent_vertex << ">";
				}
				cout << endl;
				cout << "terminal vector:" << endl;
				for (auto it = case_info.L[terminal].begin(); it != case_info.L[terminal].end(); it++) {
					cout << "<" << it->vertex << "," << it->distance << "," << it->parent_vertex << ">";
				}
				cout << endl;

				cout << "query dis = " << dis << endl;
				cout << "distances[terminal] = " << distances[terminal] << endl;
				cout << "abs(dis - distances[terminal]) > 1e-5!" << endl;
				getchar();
			}

			if (dis >= TwoM_value) {
				continue;
			}

			vector<pair<int, int>> path; 
			extract_shortest_path(case_info, source, terminal, path);

			double path_dis = 0;
			if (path.size() == 0) {
				if (source != terminal) { // disconnected
					path_dis = std::numeric_limits<int>::max();
				}
			}
			else {
				for (auto it = path.begin(); it != path.end(); it++) {
					path_dis = path_dis + instance_graph.edge_weight(it->first, it->second);
					if (path_dis > std::numeric_limits<int>::max()) {
						path_dis = std::numeric_limits<int>::max();
					}
				}
			}
			if (abs(dis - path_dis) > 1e-4 && (dis < TwoM_value || distances[terminal] < TwoM_value)) {
				cout << "source = " << source << endl;
				cout << "terminal = " << terminal << endl;

				cout << "source vector:" << endl;
				for (auto it = case_info.L[source].begin(); it != case_info.L[source].end(); it++) {
					cout << "<" << it->vertex << "," << it->distance << "," << it->parent_vertex << ">";
				}
				cout << endl;
				cout << "terminal vector:" << endl;
				for (auto it = case_info.L[terminal].begin(); it != case_info.L[terminal].end(); it++) {
					cout << "<" << it->vertex << "," << it->distance << "," << it->parent_vertex << ">";
				}
				cout << endl;

				print_vector_pair_int(path);
				cout << "query dis = " << dis << endl;
				cout << "path_dis = " << path_dis << endl;
				cout << "abs(dis - path_dis) > 1e-5!" << endl;
				getchar();
			}
		}

	}

}

void test_PLL() {

	/*parameters*/
	int iteration_graph_times = 1e2, iteration_source_times = 10, iteration_terminal_times = 10;
	int V = 100, E = 500, group_num = 10;
	int ec_min = 1, ec_max = 10;

	double avg_index_time = 0, avg_index_size_per_v = 0;
	double avg_canonical_repair_remove_label_ratio = 0;

	int generate_new_graph = 1;

	/*reduction method selection*/
	two_hop_case_info mm;
	mm.max_labal_bit_size = 6e9;
	mm.max_run_time_seconds = 1e2;
	mm.use_2M_prune = 1;
	mm.use_rank_prune = 1;
	mm.use_canonical_repair = 1;
	mm.thread_num = 10;

	/*iteration*/
	for (int i = 0; i < iteration_graph_times; i++) {
		cout << "iteration " << i << endl;

		/*input and output; below is for generating random new graph, or read saved graph*/

		graph_v_of_v<int> instance_graph;
		if (generate_new_graph == 1) {
			instance_graph = graph_v_of_v_generate_random_graph<int>(V, E, ec_min, ec_max, 1, boost_random_time_seed);
			/*add vertex groups*/
			if (group_num > 0) {
				add_vertex_groups(instance_graph, group_num);
			}
			is_mock.resize(V + group_num);
			for (int j = 0; j < V; j++) {
				is_mock[j] = false;
			}
			for (int j = 0; j < group_num; j++) {
				is_mock[V + j] = true;
			}
			instance_graph = graph_v_of_v_update_vertexIDs_by_degrees_large_to_small_mock(instance_graph, is_mock); // sort vertices
			instance_graph.txt_save("simple_iterative_tests.txt");
			binary_save_vector("simple_iterative_tests_is_mock.txt", is_mock);
		}
		else {
			instance_graph.txt_read("simple_iterative_tests.txt");
			binary_read_vector("simple_iterative_tests_is_mock.txt", is_mock);
		}

		auto begin = std::chrono::high_resolution_clock::now();
		try {
			PLL(instance_graph, mm);
		}
		catch (string s) {
			cout << s << endl;
			PLL_clear_global_values();
			continue;
		}
		auto end = std::chrono::high_resolution_clock::now();
		double runningtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
		avg_index_time = avg_index_time + runningtime / iteration_graph_times;
		avg_canonical_repair_remove_label_ratio = avg_canonical_repair_remove_label_ratio + mm.canonical_repair_remove_label_ratio / iteration_graph_times;

		/*debug*/
		if (0) {
			instance_graph.print();
			mm.print_L();
		}

		test_PLL_check_correctness(mm, instance_graph, iteration_source_times, iteration_terminal_times);

		long long int index_size = 0;
		for (auto it = mm.L.begin(); it != mm.L.end(); it++) {
			index_size = index_size + (*it).size();
		}
		avg_index_size_per_v = avg_index_size_per_v + (double)index_size / V / iteration_graph_times;

		mm.clear_labels();
	}

	mm.print_times();
	mm.print_L();
	cout << "avg_index_time: " << avg_index_time << "s" << endl;
	cout << "avg_index_size_per_v: " << avg_index_size_per_v << endl;
	cout << "avg_canonical_repair_remove_label_ratio: " << avg_canonical_repair_remove_label_ratio << endl;
}





void test_CT_check_correctness(CT_case_info& case_info, graph_v_of_v<int>& instance_graph, int iteration_source_times, int iteration_terminal_times) {

	/*
	below is for checking whether the above labels are right (by randomly computing shortest paths)

	this function can only be used when 0 to n-1 is in the graph, i.e., the graph is an ideal graph
	*/

	boost::random::uniform_int_distribution<> dist{ static_cast<int>(0), static_cast<int>(instance_graph.ADJs.size() - 1) };

	//graph_hash_of_mixed_weighted_print(instance_graph);

	for (int yy = 0; yy < iteration_source_times; yy++) {
		int source = dist(boost_random_time_seed);
		std::vector<int> distances, predecessors;

		//source = 6; cout << "source = " << source << endl;

		graph_v_of_v_shortest_paths<int>(instance_graph, source, distances, predecessors);

		for (int xx = 0; xx < iteration_terminal_times; xx++) {

			int terminal = dist(boost_random_time_seed);

			//terminal = 12; cout << "terminal = " << terminal << endl;

			double dis = extract_distance(case_info, source, terminal);

			if (abs(dis - distances[terminal]) > 1e-4 && (dis < TwoM_value || distances[terminal] < TwoM_value)) {
				cout << "source = " << source << endl;
				cout << "terminal = " << terminal << endl;
				cout << "query dis = " << dis << endl;
				cout << "distances[terminal] = " << distances[terminal] << endl;
				cout << "abs(dis - distances[terminal]) = " << abs(dis - distances[terminal]) << endl;
				getchar();
			}

			if (dis == std::numeric_limits<int>::max()) {
				continue; // disconnected
			}

			vector<pair<int, int>> path;
			extract_shortest_path(case_info, source, terminal, path);

			//print_vector_pair_int(path);

			double path_dis = 0;
			if (path.size() == 0) {
				if (source != terminal) { // disconnected
					path_dis = std::numeric_limits<int>::max();
				}
			}
			else {
				//print_vector_pair_int(path);
				//cout << "xx" << endl;
				for (auto it = path.begin(); it != path.end(); it++) {
					path_dis = path_dis + instance_graph.edge_weight(it->first, it->second);
					if (path_dis > std::numeric_limits<int>::max()) {
						path_dis = std::numeric_limits<int>::max();
					}
				}
			}
			if (abs(dis - path_dis) > 1e-4 && (dis < TwoM_value || distances[terminal] < TwoM_value)) {
				cout << "source = " << source << endl;
				cout << "terminal = " << terminal << endl;
				print_vector_pair_int(path);
				cout << "query dis = " << dis << endl;
				cout << "path_dis = " << path_dis << endl;
				cout << "abs(dis - path_dis) > 1e-5!" << endl;
				getchar();
			}
		}

	}

}

void test_CT() {

	/*parameters*/
	int iteration_graph_times = 1e2, iteration_source_times = 100, iteration_terminal_times = 100;
	int V = 1000, E = 5000, group_num = 3;
	int ec_min = 1, ec_max = 10;

	double avg_index_time = 0, avg_tree_indexs_time = 0, avg_core_indexs_time = 0;
	double P2H_remove_ratio;

	int generate_new_graph = 1;

	CT_case_info mm;
	mm.d = 20;
	mm.use_P2H_pruning = 0;
	mm.two_hop_info.use_2M_prune = 0;
	mm.two_hop_info.use_canonical_repair = 1;
	mm.thread_num = 10;

	/*iteration*/
	for (int i = 0; i < iteration_graph_times; i++) {
		cout << "iteration " << i << endl;

		/*input and output; below is for generating random new graph, or read saved graph*/

		graph_v_of_v<int> instance_graph;
		if (generate_new_graph == 1) {
			instance_graph = graph_v_of_v_generate_random_graph<int>(V, E, ec_min, ec_max, 1, boost_random_time_seed);
			/*add vertex groups*/
			if (group_num > 0) {
				add_vertex_groups(instance_graph, group_num);
			}
			instance_graph = graph_v_of_v_update_vertexIDs_by_degrees_large_to_small(instance_graph); // sort vertices
			instance_graph.txt_save("simple_iterative_tests.txt");
		}
		else {
			instance_graph.txt_read("simple_iterative_tests.txt");
		}

		auto begin = std::chrono::high_resolution_clock::now();
		try {
			CT(instance_graph, mm);
		}
		catch (string s) {
			cout << s << endl;
			PLL_clear_global_values();
			continue;
		}
		auto end = std::chrono::high_resolution_clock::now();
		double runningtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
		avg_index_time += runningtime / iteration_graph_times;
		avg_tree_indexs_time += mm.time_tree_decomposition / iteration_graph_times;
		avg_core_indexs_time += mm.time_core_indexs / iteration_graph_times;

		/*debug*/
		if (0) {
			instance_graph.print();
			mm.two_hop_info.print_L();
		}

		test_CT_check_correctness(mm, instance_graph, iteration_source_times, iteration_terminal_times);

		mm.clear_labels();
	}

	mm.print_times();
	cout << endl;
	cout << "avg_tree_indexs_time: " << avg_tree_indexs_time << endl;
	cout << "avg_core_indexs_time: " << avg_core_indexs_time << endl;
	cout << "avg_index_time: " << avg_index_time << "s" << endl;
	mm.record_all_details("test.txt");
}

void compare_query_speed() {

	/*parameters*/
	int iteration_graph_times = 1e1, query_times = 1e3;
	int V = 1000, E = 5000, group_num = 50;
	int ec_min = 1, ec_max = 10;

	double avg_MLL_pre_query_time = 0, avg_new_pre_query_time = 0;

	/*iteration*/
	for (int i = 0; i < iteration_graph_times; i++) {
		cout << "iteration " << i << endl;

		graph_v_of_v<int> instance_graph = graph_v_of_v_generate_random_graph<int>(V, E, ec_min, ec_max, 1, boost_random_time_seed);
		/*add vertex groups*/
		if (group_num > 0) {
			add_vertex_groups(instance_graph, group_num);
		}
		instance_graph = graph_v_of_v_update_vertexIDs_by_degrees_large_to_small(instance_graph); // sort vertices

		CT_case_info mm;
		mm.d = 10;
		mm.use_P2H_pruning = 1;
		mm.two_hop_info.use_2M_prune = 1;
		mm.two_hop_info.use_canonical_repair = 1;
		mm.thread_num = 10;

		CT(instance_graph, mm);

		CT_case_info mm2;
		mm2.d = 10;
		mm2.use_P2H_pruning = 0;
		mm2.two_hop_info.use_2M_prune = 0;
		mm2.two_hop_info.use_canonical_repair = 1;
		mm2.thread_num = 10;

		CT(instance_graph, mm2);

		/*query_seeds*/
		boost::random::uniform_int_distribution<> dist{ static_cast<int>(0), static_cast<int>(V + group_num - 1) };
		vector<pair<int, int>> query_seeds;
		while (query_seeds.size() < query_times) {
			int s = dist(boost_random_time_seed), t = dist(boost_random_time_seed), dis = extract_distance(mm, s, t);
			if (dis > 1e6 && dis < 2e6) {
				query_seeds.push_back({ s, t });
			}
		}
		
		if (1) {
			auto begin = std::chrono::high_resolution_clock::now();
			for (int x = 0; x < query_times; x++) {
				vector<pair<int, int>> path;
				extract_shortest_path(mm, query_seeds[x].first, query_seeds[x].second, path);

			}
			auto end = std::chrono::high_resolution_clock::now();
			avg_new_pre_query_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9 / query_times; // s
		}

		if (1) {
			auto begin = std::chrono::high_resolution_clock::now();
			for (int x = 0; x < query_times; x++) {
				vector<pair<int, int>> path;
				extract_shortest_path(mm2, query_seeds[x].first, query_seeds[x].second, path);
			}
			auto end = std::chrono::high_resolution_clock::now();
			avg_MLL_pre_query_time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9 / query_times; // s
		}

		cout << "avg_new_pre_query_time: " << avg_new_pre_query_time << " avg_MLL_pre_query_time: " << avg_MLL_pre_query_time
			<< " ratio: " << avg_new_pre_query_time / avg_MLL_pre_query_time << endl;
	}
}




void hop_constrained_check_correctness(hop_constrained_case_info& case_info, graph_v_of_v<int>& instance_graph,
	int iteration_source_times, int iteration_terminal_times, int upper_k) {

	boost::random::uniform_int_distribution<> vertex_range{ static_cast<int>(0), static_cast<int>(instance_graph.size() - 1) };
	boost::random::uniform_int_distribution<> hop_range{ static_cast<int>(1), static_cast<int>(upper_k) };

	for (int yy = 0; yy < iteration_source_times; yy++) {
		int source = vertex_range(boost_random_time_seed);

		while (!is_mock[source]) {
			source = vertex_range(boost_random_time_seed);
		}

		std::vector<int> distances(instance_graph.size());

		int hop_cst = hop_range(boost_random_time_seed);

		graph_v_of_v_hop_constrained_shortest_distance(instance_graph, source, hop_cst, distances);

		for (int xx = 0; xx < iteration_terminal_times; xx++) {
			int terminal = vertex_range(boost_random_time_seed);

			while (is_mock[terminal]) {
				terminal = vertex_range(boost_random_time_seed);
			}

			int query_dis = hop_constrained_extract_distance(case_info.L, source, terminal, hop_cst);

			if (abs(query_dis - distances[terminal]) > 1e-4 && (query_dis < TwoM_value || distances[terminal] < TwoM_value)) {
				instance_graph.print();
				case_info.print_L();
				cout << "source = " << source << endl;
				cout << "terminal = " << terminal << endl;
				cout << "hop_cst = " << hop_cst << endl;
				cout << "query_dis = " << query_dis << endl;
				cout << "distances[terminal] = " << distances[terminal] << endl;
				cout << "abs(dis - distances[terminal]) > 1e-5!" << endl;
				getchar();
			}

			vector<pair<int, int>> path = hop_constrained_extract_shortest_path(case_info.L, source, terminal, hop_cst);
			int path_dis = 0;
			for (auto xx : path) {
				path_dis += instance_graph.edge_weight(xx.first, xx.second);
			}
			if (abs(query_dis - path_dis) > 1e-4 && (query_dis < TwoM_value || distances[terminal] < TwoM_value)) {
				instance_graph.print();
				case_info.print_L();
				cout << "source = " << source << endl;
				cout << "terminal = " << terminal << endl;
				cout << "hop_cst = " << hop_cst << endl;
				std::cout << "print_vector_pair_int:" << std::endl;
				for (int i = 0; i < path.size(); i++) {
					std::cout << "item: |" << path[i].first << "," << path[i].second << "|" << std::endl;
				}
				cout << "query_dis = " << query_dis << endl;
				cout << "path_dis = " << path_dis << endl;
				cout << "abs(dis - path_dis) > 1e-5!" << endl;
				getchar();
			}
		}
	}
}

void test_HSDL() {

	/* problem parameters */
	int iteration_graph_times = 1e2, iteration_source_times = 10, iteration_terminal_times = 10;
	int V = 100, E = 500, group_num = 10;
	int ec_min = 1, ec_max = 10;

	bool generate_new_graph = 1;

	/* hop bounded info */
	hop_constrained_case_info mm;
	mm.upper_k = 5;
	mm.use_2M_prune = 0;
	mm.use_rank_prune = 1;
	mm.use_2023WWW_generation = 0;
	mm.use_canonical_repair = 1;
	mm.max_run_time_seconds = 10;
	mm.thread_num = 10;

	/* result info */
	double avg_index_time = 0;
	double avg_time_initialization = 0, avg_time_generate_labels = 0, avg_time_sortL = 0, avg_time_canonical_repair = 0;
	double avg_canonical_repair_remove_label_ratio = 0, avg_index_size_per_v = 0;

	/* iteration */
	for (int i = 0; i < iteration_graph_times; i++) {
		cout << ">>>iteration_graph_times: " << i << endl;

		graph_v_of_v<int> instance_graph;
		if (generate_new_graph == 1) {
			instance_graph = graph_v_of_v_generate_random_graph<int>(V, E, ec_min, ec_max, 1, boost_random_time_seed);
			/*add vertex groups*/
			if (group_num > 0) {
				add_vertex_groups(instance_graph, group_num);
			}
			is_mock.resize(V + group_num);
			for (int j = 0; j < V; j++) {
				is_mock[j] = false;
			}
			for (int j = 0; j < group_num; j++) {
				is_mock[V + j] = true;
			}
			instance_graph = graph_v_of_v_update_vertexIDs_by_degrees_large_to_small_mock(instance_graph, is_mock); // sort vertices
			instance_graph.txt_save("simple_iterative_tests.txt");
			binary_save_vector("simple_iterative_tests_is_mock.txt", is_mock);
		}
		else {
			instance_graph.txt_read("simple_iterative_tests.txt");
			binary_read_vector("simple_iterative_tests_is_mock.txt", is_mock);
		}

		//instance_graph.print();

		auto begin = std::chrono::high_resolution_clock::now();
		try {
			hop_constrained_two_hop_labels_generation(instance_graph, mm);
		}
		catch (string s) {
			cout << s << endl;
			hop_constrained_clear_global_values();
			continue;
		}
		auto end = std::chrono::high_resolution_clock::now();
		double runningtime = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;  // s
		avg_index_time = avg_index_time + runningtime / iteration_graph_times;
		avg_time_initialization += mm.time_initialization / iteration_graph_times;
		avg_time_generate_labels += mm.time_generate_labels / iteration_graph_times;
		avg_time_sortL += mm.time_sortL / iteration_graph_times;
		avg_time_canonical_repair += mm.time_canonical_repair / iteration_graph_times;
		avg_canonical_repair_remove_label_ratio += mm.canonical_repair_remove_label_ratio / iteration_graph_times;

		long long int index_size = 0;
		for (auto it = mm.L.begin(); it != mm.L.end(); it++) {
			index_size = index_size + (*it).size();
		}
		avg_index_size_per_v = avg_index_size_per_v + (double)index_size / V / iteration_graph_times;

		hop_constrained_check_correctness(mm, instance_graph, iteration_source_times, iteration_terminal_times, mm.upper_k);

		mm.clear_labels();
	}

	cout << "avg_canonical_repair_remove_label_ratio: " << avg_canonical_repair_remove_label_ratio << endl;
	cout << "avg_index_time: " << avg_index_time << "s" << endl;
	cout << "\t avg_time_initialization: " << avg_time_initialization << endl;
	cout << "\t avg_time_generate_labels: " << avg_time_generate_labels << endl;
	cout << "\t avg_time_sortL: " << avg_time_sortL << endl;
	cout << "\t avg_time_canonical_repair: " << avg_time_canonical_repair << endl;
	cout << "\t avg_index_size_per_v: " << avg_index_size_per_v << endl;
}


void test_GST() {

	/*parameters*/
	int iteration_graph_times = 1e2, iteration_source_times = 100, iteration_terminal_times = 100;
	int V = 100, E = 500, group_num = 5;
	int ec_min = 1, ec_max = 10;

	double avg_index_time = 0, avg_tree_indexs_time = 0, avg_core_indexs_time = 0;
	double P2H_remove_ratio;

	int generate_new_graph = 1;

	/*iteration*/
	for (int i = 0; i < iteration_graph_times; i++) {
		cout << "iteration " << i << endl;

		/*input and output; below is for generating random new graph, or read saved graph*/

		graph_v_of_v<int> instance_graph;
		if (generate_new_graph == 1) {
			instance_graph = graph_v_of_v_generate_random_graph<int>(V, E, ec_min, ec_max, 1, boost_random_time_seed);
			/*add vertex groups*/
			if (group_num > 0) {
				add_vertex_groups(instance_graph, group_num);
			}
			instance_graph = graph_v_of_v_update_vertexIDs_by_degrees_large_to_small(instance_graph); // sort vertices
			instance_graph.txt_save("simple_iterative_tests.txt");
		}
		else {
			instance_graph.txt_read("simple_iterative_tests.txt");
		}

		int use_algo = 2;

		if (use_algo == 0) {
			CT_case_info mm;
			mm.d = 20;
			mm.use_P2H_pruning = 1;
			mm.two_hop_info.use_2M_prune = 1;
			mm.two_hop_info.use_canonical_repair = 1;
			mm.thread_num = 10;

			CT(instance_graph, mm);

			vector<int> groups_IDs;
			for (int i = V; i < V + group_num; i++) {
				groups_IDs.push_back(i);
			}
			unordered_set<int> tree_nodes = GST_nonHOP(mm, instance_graph, groups_IDs);
			if (tree_nodes.size() > 0) {
				vector<pair<int, int>> tree = build_a_tree(instance_graph, tree_nodes);
				//print_vector_pair_int(tree);
				if (tree.size() != tree_nodes.size() - 1) {
					getchar();
				}
				for (auto xx : tree) {
					if (tree_nodes.count(xx.first) == 0 || tree_nodes.count(xx.second) == 0) {
						getchar();
					}
				}
			}
		}
		else if (use_algo == 1) {
			two_hop_case_info mm;
			mm.max_labal_bit_size = 6e9;
			mm.max_run_time_seconds = 1e2;
			mm.use_2M_prune = 1;
			mm.use_canonical_repair = 1;
			mm.thread_num = 10;

			PLL(instance_graph, mm);

			vector<int> groups_IDs;
			for (int i = V; i < V + group_num; i++) {
				groups_IDs.push_back(i);
			}
			unordered_set<int> tree_nodes = GST_nonHOP(mm, instance_graph, groups_IDs);
			if (tree_nodes.size() > 0) {
				vector<pair<int, int>> tree = build_a_tree(instance_graph, tree_nodes);
				//print_vector_pair_int(tree);
				if (tree.size() != tree_nodes.size() - 1) {
					getchar();
				}
				for (auto xx : tree) {
					if (tree_nodes.count(xx.first) == 0 || tree_nodes.count(xx.second) == 0) {
						getchar();
					}
				}
			}
		}
		else {
			/* hop bounded info */
			hop_constrained_case_info mm;
			mm.upper_k = 10;
			mm.use_2M_prune = 1;
			mm.use_2023WWW_generation = 0;
			mm.use_canonical_repair = 1;
			mm.max_run_time_seconds = 1e5;
			mm.thread_num = 10;

			hop_constrained_two_hop_labels_generation(instance_graph, mm);

			vector<int> groups_IDs;
			for (int i = V; i < V + group_num; i++) {
				groups_IDs.push_back(i);
			}
			unordered_set<int> tree_nodes = GST_HOP(mm, instance_graph, groups_IDs, 18);
			if (tree_nodes.size() > 0) {
				vector<pair<int, int>> tree = build_a_tree(instance_graph, tree_nodes);
				//cout << "tree_nodes.size() " << tree_nodes.size() << endl;
				//print_vector_pair_int(tree);
				if (tree.size() != tree_nodes.size() - 1) {
					getchar();
				}
				for (auto xx : tree) {
					if (tree_nodes.count(xx.first) == 0 || tree_nodes.count(xx.second) == 0) {
						getchar();
					}
				}
			}
		}

		cout << endl;
	}
}






