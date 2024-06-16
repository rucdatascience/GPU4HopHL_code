#pragma once
#include <text_mining/parse_string.h> 
#include <graph_v_of_v_idealID/graph_v_of_v_idealID.h> 
#include <text_mining/parse_string.h> 
#include <fstream>

void graph_v_of_v_idealID_read(std::string instance_name, graph_v_of_v_idealID& input_graph) {

	input_graph.clear();

	std::string line_content;
	std::ifstream myfile(instance_name); // open the file
	if (myfile.is_open()) // if the file is opened successfully
	{
		while (getline(myfile, line_content)) // read file line by line
		{
			std::vector<std::string> Parsed_content = parse_string(line_content, " ");

			if (!Parsed_content[0].compare("input_graph_size"))
			{
				int v = std::stoi(Parsed_content[2]);
                graph_v_of_v_idealID g(v);
                input_graph = g;
			}
			else if (!Parsed_content[0].compare("input_graph"))
			{
				int v1 = std::stoi(Parsed_content[2]);
				int v2 = std::stoi(Parsed_content[3]);
				double ec = std::stod(Parsed_content[4]);
				graph_v_of_v_idealID_add_edge(input_graph, v1, v2, ec);
			}
		}

		myfile.close(); //close the file
	}
	else
	{
		std::cout << "Unable to open file " << instance_name << std::endl
			<< "Please check the file location or file name." << std::endl; // throw an error message
		getchar(); // keep the console window
		exit(1); // end the program
	}
}