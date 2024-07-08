#pragma once
#include <graph_v_of_v/graph_v_of_v.h>
#include <unordered_map>

using namespace std;

bool compare_graph_v_of_v_update_vertexIDs_by_degrees_large_to_small(const pair<int, int>& i, pair<int, int>& j)
{
	/*< is nearly 10 times slower than >*/
	return i.second > j.second;  // < is from small to big; > is from big to small.  sort by the second item of pair<int, int>
}

template <typename weight_type>
graph_v_of_v<weight_type> graph_v_of_v_update_vertexIDs_by_degrees_large_to_small(graph_v_of_v<weight_type>& input_graph) {

	int N = input_graph.ADJs.size();

	vector<pair<int, int>> sorted_vertices;
	for (int i = 0; i < N; i++) {
		sorted_vertices.push_back({ i, input_graph.ADJs[i].size() });
	}
	sort(sorted_vertices.begin(), sorted_vertices.end(), compare_graph_v_of_v_update_vertexIDs_by_degrees_large_to_small);
	vector<int> vertexID_old_to_new(N);
	for (int i = 0; i < N; i++) {
		vertexID_old_to_new[sorted_vertices[i].first] = i;
	}

	graph_v_of_v<weight_type> output_graph(N);

	for (int i = 0; i < N; i++) {
		int v_size = input_graph.ADJs[i].size();
		for (int j = 0; j < v_size; j++) {
			if (i < input_graph.ADJs[i][j].first) {
				output_graph.add_edge(vertexID_old_to_new[i], vertexID_old_to_new[input_graph.ADJs[i][j].first], input_graph.ADJs[i][j].second);
			}
		}
	}

	return output_graph;
}


template <typename weight_type>
graph_v_of_v<weight_type> graph_v_of_v_update_vertexIDs_by_degrees_large_to_small_mock(graph_v_of_v<weight_type>& input_graph, vector<int>& is_mock) {

	int N = input_graph.ADJs.size();

	vector<int> new_is_mock(N);

	vector<pair<int, int>> sorted_vertices;
	for (int i = 0; i < N; i++) {
		sorted_vertices.push_back({ i, input_graph.ADJs[i].size() });
	}
	sort(sorted_vertices.begin(), sorted_vertices.end(), compare_graph_v_of_v_update_vertexIDs_by_degrees_large_to_small);


	/*ineffective to reduce index time and size and increase query efficiency*/
	//int last_mock_ID = N;
	//for (int i = N - 1; i >= 0; i--) {
	//	if (is_mock[sorted_vertices[i].first]) {
	//		last_mock_ID = i;
	//		break;
	//	}
	//}
	//int NONmock_num_before_LASTmock = 0;
	//for (int i = last_mock_ID - 1; i >= 0; i--) {
	//	if (!is_mock[sorted_vertices[i].first]) {
	//		NONmock_num_before_LASTmock++;
	//	}
	//}
	//int NONmock_num_process = 0.01 * NONmock_num_before_LASTmock;
	//int change_begin = 0;
	//NONmock_num_before_LASTmock = 0;
	//for (int i = 0; i < N; i++) {
	//	if (!is_mock[sorted_vertices[i].first]) {
	//		NONmock_num_before_LASTmock++;
	//		if (NONmock_num_before_LASTmock == NONmock_num_process) {
	//			change_begin = i + 1;
	//			break;
	//		}
	//	}
	//}
	//if (change_begin < N) {
	//	vector<pair<int, int>> mocks, nonmocks;
	//	for (int i = change_begin; i < N; i++) {
	//		if (is_mock[sorted_vertices[i].first]) {
	//			mocks.push_back(sorted_vertices[i]);
	//		}
	//		else {
	//			nonmocks.push_back(sorted_vertices[i]);
	//		}
	//	}
	//	for (int i = 0; i < mocks.size(); i++) {
	//		sorted_vertices[change_begin + i] = mocks[i];
	//	}
	//	for (int i = 0; i < nonmocks.size(); i++) {
	//		sorted_vertices[change_begin + mocks.size() + i] = nonmocks[i];
	//	}
	//}


	vector<int> vertexID_old_to_new(N);
	for (int i = 0; i < N; i++) {
		vertexID_old_to_new[sorted_vertices[i].first] = i;
		new_is_mock[i] = is_mock[sorted_vertices[i].first];
	}

	for (int i = 0; i < N; i++) {
		is_mock[i] = new_is_mock[i];
	}

	graph_v_of_v<weight_type> output_graph(N);

	for (int i = 0; i < N; i++) {
		int v_size = input_graph.ADJs[i].size();
		for (int j = 0; j < v_size; j++) {
			if (i < input_graph.ADJs[i][j].first) {
				output_graph.add_edge(vertexID_old_to_new[i], vertexID_old_to_new[input_graph.ADJs[i][j].first], input_graph.ADJs[i][j].second);
			}
		}
	}

	return output_graph;
}

template <typename weight_type>
graph_v_of_v<weight_type> graph_v_of_v_update_vertexIDs_by_degrees_large_to_small(graph_v_of_v<weight_type>& input_graph, vector<int>& vertexID_new_to_old) {

	int N = input_graph.ADJs.size();
	vertexID_new_to_old.resize(N);

	vector<pair<int, int>> sorted_vertices;
	for (int i = 0; i < N; i++) {
		sorted_vertices.push_back({ i, input_graph.ADJs[i].size() });
	}
	sort(sorted_vertices.begin(), sorted_vertices.end(), compare_graph_v_of_v_update_vertexIDs_by_degrees_large_to_small);
	vector<int> vertexID_old_to_new(N);
	for (int i = 0; i < N; i++) {
		vertexID_old_to_new[sorted_vertices[i].first] = i;
		vertexID_new_to_old[i] = sorted_vertices[i].first;
	}

	graph_v_of_v<weight_type> output_graph(N);

	for (int i = 0; i < N; i++) {
		int v_size = input_graph.ADJs[i].size();
		for (int j = 0; j < v_size; j++) {
			if (i < input_graph.ADJs[i][j].first) {
				output_graph.add_edge(vertexID_old_to_new[i], vertexID_old_to_new[input_graph.ADJs[i][j].first], input_graph.ADJs[i][j].second);
			}
		}
	}

	return output_graph;
}

