#ifndef TWO_HOP_LABELS_H
#define TWO_HOP_LABELS_H
#pragma once
#include <vector>
#include <string>
#include <iostream>
#include <limits>
#include <fstream>
using namespace std;

class two_hop_label {
public:
	int vertex, parent_vertex, distance;
};

class two_hop_case_info {
public:

	/*use info*/
	bool use_2M_prune = false;
	bool use_rank_prune = true;
	bool use_canonical_repair = false;
	int thread_num = 1;

	/*canonical_repair info*/
	long long int label_size_before_canonical_repair = 0;
	long long int label_size_after_canonical_repair = 0;
	double canonical_repair_remove_label_ratio = 0;

	/*running time records*/
	double time_initialization = 0;
	double time_generate_labels = 0;
	double time_sortL = 0;
	double time_canonical_repair = 0;
	double time_total = 0;

	/*running limits*/
	long long int max_labal_bit_size = 1e12; 
	double max_run_time_seconds = 1e12;

	/*labels*/
	vector<vector<two_hop_label>> L;

	/*compute label size; this should equal label_size_after_canonical_repair when use_canonical_repair==true*/
	long long int compute_label_bit_size() {
		long long int size = 0;
		for (auto it = L.begin(); it != L.end(); it++) {
			size = size + (*it).size() * sizeof(two_hop_label); // 12 bit per two_hop_label
		}
		return size;
	}

	void clear_labels() {
		vector<vector<two_hop_label>>().swap(L);
	}

	void print_L() {
		cout << "print_L:" << endl;
		for (int i = 0; i < L.size(); i++) {
			cout << "L[" << i << "]=";
			for (int j = 0; j < L[i].size(); j++) {
				cout << "{" << L[i][j].vertex << "," << L[i][j].distance << "," << L[i][j].parent_vertex << "}";
			}
			cout << endl;
		}
	}

	void print_times() {
		cout << "print_times:" << endl;
		cout << "time_initialization: " << time_initialization << "s" << endl;
		cout << "time_generate_labels: " << time_generate_labels << "s" << endl;
		cout << "time_sortL: " << time_sortL << "s" << endl;
		cout << "time_canonical_repair: " << time_canonical_repair << "s" << endl;
	}

	/*record_all_details*/
	void record_all_details(string save_name) {
		ofstream outputFile;
		outputFile.precision(6);
		outputFile.setf(ios::fixed);
		outputFile.setf(ios::showpoint);
		outputFile.open(save_name + ".txt");

		outputFile << "PLL info:" << endl;
		outputFile << "thread_num=" << thread_num << endl;
		outputFile << "use_2M_prune=" << use_2M_prune << endl;
		outputFile << "use_rank_prune=" << use_rank_prune << endl;
		outputFile << "use_canonical_repair=" << use_canonical_repair << endl;

		outputFile << "label_size_before_canonical_repair=" << label_size_before_canonical_repair << endl;
		outputFile << "label_size_after_canonical_repair=" << label_size_after_canonical_repair << endl;
		outputFile << "canonical_repair_remove_label_ratio=" << canonical_repair_remove_label_ratio << endl;

		outputFile << "time_initialization=" << time_initialization << endl;
		outputFile << "time_generate_labels=" << time_generate_labels << endl;
		outputFile << "time_sortL=" << time_sortL << endl;
		outputFile << "time_canonical_repair=" << time_canonical_repair << endl;
		outputFile << "time_total=" << time_total << endl;

		outputFile << "max_labal_bit_size=" << max_labal_bit_size << endl;
		outputFile << "max_run_time_seconds=" << max_run_time_seconds << endl;

		outputFile << "compute_label_bit_size()=" << compute_label_bit_size() << endl;

		outputFile.close();
	}

};


/*querying distances or paths*/

int extract_distance(two_hop_case_info& case_info, int source, int terminal) {

	auto& L = case_info.L;

	/*return std::numeric_limits<double>::max() is not connected*/

	if (source == terminal) {
		return 0;
	}

	int distance = std::numeric_limits<int>::max(); // if disconnected, return this large value
	auto vector1_check_pointer = L[source].begin();
	auto vector2_check_pointer = L[terminal].begin();
	auto pointer_L_s_end = L[source].end(), pointer_L_t_end = L[terminal].end();
	while (vector1_check_pointer != pointer_L_s_end && vector2_check_pointer != pointer_L_t_end) {
		if (vector1_check_pointer->vertex == vector2_check_pointer->vertex) {
			int dis = vector1_check_pointer->distance + vector2_check_pointer->distance;
			if (distance > dis) {
				distance = dis;
			}
			vector1_check_pointer++;
		}
		else if (vector1_check_pointer->vertex > vector2_check_pointer->vertex) {
			vector2_check_pointer++;
		}
		else {
			vector1_check_pointer++;
		}
	}

	return distance;

}

void extract_shortest_path(two_hop_case_info& case_info, int source, int terminal, vector<pair<int, int>>& path) {

	auto& L = case_info.L;

	if (source == terminal) {
		return;
	}

	int vector1_capped_v_parent = 0, vector2_capped_v_parent = 0;
	int distance = std::numeric_limits<int>::max(); // if disconnected, return this large value
	bool connected = false;
	auto vector1_check_pointer = L[source].begin();
	auto vector2_check_pointer = L[terminal].begin();
	auto pointer_L_s_end = L[source].end(), pointer_L_t_end = L[terminal].end();
	while (vector1_check_pointer != pointer_L_s_end && vector2_check_pointer != pointer_L_t_end)
	{
		if (vector1_check_pointer->vertex == vector2_check_pointer->vertex)
		{
			connected = true;
			int dis = vector1_check_pointer->distance + vector2_check_pointer->distance;
			if (distance > dis)
			{
				distance = dis;
				vector1_capped_v_parent = vector1_check_pointer->parent_vertex;
				vector2_capped_v_parent = vector2_check_pointer->parent_vertex;
			}
			vector1_check_pointer++;
		}
		else if (vector1_check_pointer->vertex > vector2_check_pointer->vertex)
		{
			vector2_check_pointer++;
		}
		else
		{
			vector1_check_pointer++;
		}
	}

	if (connected) {
		if (source != vector1_capped_v_parent) {
			path.push_back({ source, vector1_capped_v_parent });
			source = vector1_capped_v_parent; // ascending from source
		}
		if (terminal != vector2_capped_v_parent) {
			path.push_back({ terminal, vector2_capped_v_parent });
			terminal = vector2_capped_v_parent; // ascending from terminal
		}
	}
	else {
		path.clear();
		//path.push_back({ INT_MAX, INT_MAX });
		return;
	}

	// find new edges
	extract_shortest_path(case_info, source, terminal, path);

	return;
}

pair<int, int> two_hop_extract_two_predecessors(vector<vector<two_hop_label>>& L, int source, int terminal) {

	vector<pair<int, int>> paths;
	if (source == terminal) {
		return { source , terminal };
	}

	int vector1_capped_v_parent = 0, vector2_capped_v_parent = 0;
	int distance = std::numeric_limits<int>::max(); // if disconnected, retun this large value
	bool connected = false;
	auto vector1_check_pointer = L[source].begin();
	auto vector2_check_pointer = L[terminal].begin();
	auto pointer_L_s_end = L[source].end(), pointer_L_t_end = L[terminal].end();
	while (vector1_check_pointer != pointer_L_s_end && vector2_check_pointer != pointer_L_t_end)
	{
		if (vector1_check_pointer->vertex == vector2_check_pointer->vertex)
		{
			connected = true;
			int dis = vector1_check_pointer->distance + vector2_check_pointer->distance;
			if (distance > dis)
			{
				distance = dis;
				vector1_capped_v_parent = vector1_check_pointer->parent_vertex;
				vector2_capped_v_parent = vector2_check_pointer->parent_vertex;
			}
			vector1_check_pointer++;
		}
		else if (vector1_check_pointer->vertex > vector2_check_pointer->vertex)
		{
			vector2_check_pointer++;
		}
		else
		{
			vector1_check_pointer++;
		}
	}

	if (connected) {
		return { vector1_capped_v_parent , vector2_capped_v_parent };
	}
	else {
		return { source , terminal };
	}
}
#endif // TWO_HOP_LABELS_H
