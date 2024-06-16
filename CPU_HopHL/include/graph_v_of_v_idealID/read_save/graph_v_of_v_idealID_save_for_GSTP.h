#pragma once
#include <fstream>
#include <iostream>
#include <unordered_set>
#include <graph_v_of_v_idealID/graph_v_of_v_idealID.h>


void graph_v_of_v_idealID_save_for_GSTP(std::string instance_name, graph_v_of_v_idealID& input_graph, graph_v_of_v_idealID& group_graph, std::unordered_set<int>& group_vertices) {

	std::ofstream outputFile;
	outputFile.precision(10);
	outputFile.setf(std::ios::fixed);
	outputFile.setf(std::ios::showpoint);
	outputFile.open(instance_name);

	// comments
	outputFile << "SECTION Comments" << std::endl;
	outputFile << "Name \"" << instance_name << "\"" << std::endl;
	outputFile << "Creator \"graph_v_of_v_idealID_save_for_GSTP\"" << std::endl;
	outputFile << "END" << std::endl;
	outputFile << std::endl;

	// input_graph
	outputFile << "input_graph_size |V|= " << input_graph.size() << " |E|= " << graph_v_of_v_idealID_total_edge_num(input_graph) << std::endl;
	outputFile << std::endl;
	int size = input_graph.size();
	for (int i = 0; i < size; i++) {
		int v_size = input_graph[i].size();
		for (int j = 0; j < v_size; j++) {
			if (i <= input_graph[i][j].first) {
				outputFile << "input_graph Edge " << i << " " << input_graph[i][j].first << " " << input_graph[i][j].second << '\n';
			}
		}
		outputFile << std::endl;
	}
	outputFile << std::endl;


	// group_graph
	outputFile << "group_graph_size |V|= " << group_graph.size() << " |E|= " << graph_v_of_v_idealID_total_edge_num(group_graph) << std::endl;
	outputFile << std::endl;
	size = group_graph.size();
	for (int i = 0; i < size; i++) {
		int v_size = group_graph[i].size();
		for (int j = 0; j < v_size; j++) {
			if (i <= group_graph[i][j].first) {
				outputFile << "group_graph Edge " << i << " " << group_graph[i][j].first << " " << group_graph[i][j].second << '\n';
			}
		}
		outputFile << std::endl;
	}
	outputFile << std::endl;

	// group_vertices
	outputFile << "|T|=" << group_vertices.size() << std::endl;
	for (auto it = group_vertices.begin(); it != group_vertices.end(); ++it) {
		outputFile << "group_vertices " << *it << '\n';
	}
	outputFile << std::endl;

	outputFile << "EOF" << std::endl;

}
