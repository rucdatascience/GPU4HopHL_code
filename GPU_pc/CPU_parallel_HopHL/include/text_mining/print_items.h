#pragma once
#include<vector>
#include<iostream>



/*
print_a_sequence_of_elements_v1 suits
vector<int>, vector<double>, vector<string>, or lists, or unordered_sets
*/
template <typename T>
void print_a_sequence_of_elements(T& input_sequence) {

	std::cout << "print_a_sequence_of_elements:" << std::endl;
	for (auto it = input_sequence.begin(); it != input_sequence.end(); it++) {
		std::cout << "item: |" << *it << "|" << std::endl;
	}

}







// vector

/* 
print_vector_v1 suits
vector<int>, vector<double>, vector<string>
*/
template <typename T> 
void print_vector_v1(std::vector<T>& input_vector) {

	std::cout << "print_vector_v1:" << std::endl;
	for (int i = 0; i < input_vector.size(); i++) {
		std::cout << "item: |" << input_vector[i] << "|" << std::endl;
	}

}

void print_vector_pair_int(std::vector<std::pair<int, int>>& input_vector) {

	std::cout << "print_vector_pair_int:" << std::endl;
	for (int i = 0; i < input_vector.size(); i++) {
		std::cout << "item: |" << input_vector[i].first << "," << input_vector[i].second << "|" << std::endl;
	}

}







// list 
#include <list>
void print_list_int(std::list<int>& input_list) {

	std::cout << "print_list_int:" << std::endl;
	for (auto i = input_list.begin(); i != input_list.end(); i++) {
		std::cout << "item: |" << *i << "|" << std::endl;
	}
	std::cout << "print_list_int END" << std::endl;
}











// unordered_set

#include <unordered_set>

/*
print_vector_v1 suits
unordered_set<int>, unordered_set<double>, unordered_set<string>
*/
template <typename T>
void print_unordered_set_v1(std::unordered_set<T>& input_set) {

	std::cout << "print_unordered_set_v1:" << std::endl;
	for (auto it = input_set.begin(); it != input_set.end(); it++) {
		std::cout << "item: |" << *it << "|" << std::endl;
	}

}






// unordered_map

#include <unordered_map>

void print_unordered_map_string_int(std::unordered_map<std::string, int>& input_map) {

	std::cout << "print_unordered_map_string_int:" << std::endl;
	std::cout << "size(): " << input_map.size() << std::endl;
	for (auto i = input_map.begin(); i != input_map.end(); i++) {
		std::cout << "key: |" << i->first << "|" << " content: |" << i->second << "|" << '\n';
	}
	std::cout << "print_unordered_set_int END" << std::endl;
}

void print_unordered_map_string_double(std::unordered_map<std::string, double>& input_map) {

	std::cout << "print_unordered_map_string_int:" << std::endl;
	std::cout << "size(): " << input_map.size() << std::endl;
	for (auto i = input_map.begin(); i != input_map.end(); i++) {
		std::cout << "key: |" << i->first << "|" << " content: |" << i->second << "|" << '\n';
	}
	std::cout << "print_unordered_set_int END" << std::endl;
}


void print_unordered_map_int_string(std::unordered_map<int, std::string>& input_map) {

	std::cout << "print_unordered_map_int_string:" << std::endl;
	std::cout << "size(): " << input_map.size() << std::endl;
	for (auto i = input_map.begin(); i != input_map.end(); i++) {
		std::cout << "key: |" << i->first << "|" << " content: |" << i->second << "|" << '\n';
	}
	std::cout << "print_unordered_map_int_string END" << std::endl;
}

void print_unordered_map_int_int(std::unordered_map<int, int>& input_map) {

	std::cout << "print_unordered_map_int_int:" << std::endl;
	std::cout << "size(): " << input_map.size() << std::endl;
	for (auto i = input_map.begin(); i != input_map.end(); i++) {
		std::cout << "key: |" << i->first << "|" << " content: |" << i->second << "|" << '\n';
	}
	std::cout << "print_unordered_map_int_int END" << std::endl;
}


void print_unordered_map_int_double(std::unordered_map<int, double>& input_map) {

	std::cout << "print_unordered_map_int_double:" << std::endl;
	std::cout << "size(): " << input_map.size() << std::endl;
	for (auto i = input_map.begin(); i != input_map.end(); i++) {
		std::cout << "key: |" << i->first << "|" << " content: |" << i->second << "|" << '\n';
	}
	std::cout << "print_unordered_map_int_double END" << std::endl;
}