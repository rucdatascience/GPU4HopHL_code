#ifndef HOP_CONSTRAINED_TWO_HOP_LABELS_H
#define HOP_CONSTRAINED_TWO_HOP_LABELS_H
#pragma once

#include "definition/hub_def.h"
#include <label/hop_constrained_two_hop_labels_v2.cuh>

/* label format */
class hop_constrained_two_hop_label
{
public:
	int hub_vertex, parent_vertex, hop, distance;

	hop_constrained_two_hop_label (int hv, int pv, int h, int d) : hub_vertex(hv), parent_vertex(pv), hop(h), distance(d) {}
	hop_constrained_two_hop_label() {}
	// copy
	hop_constrained_two_hop_label (const hop_constrained_two_hop_label &other) {
		hub_vertex = other.hub_vertex;
		parent_vertex = other.parent_vertex;
		hop = other.hop;
		distance = other.distance;
	}
};

// 定义哈希函数
namespace std {
	template <>
	struct hash<hop_constrained_two_hop_label>{
		size_t operator()(const hop_constrained_two_hop_label &label) const {
			return hash<int>()(label.hub_vertex) ^ (hash<int>()(label.parent_vertex) << 1) ^ (hash<int>()(label.hop) << 2) ^ (hash<int>()(label.distance) << 3);
		}
	};
} // namespace std

// 定义等价性比较操作符
inline static bool operator == (const hop_constrained_two_hop_label &lhs, const hop_constrained_two_hop_label &rhs) {
	return lhs.hub_vertex == rhs.hub_vertex && lhs.parent_vertex == rhs.parent_vertex && lhs.hop == rhs.hop && lhs.distance == rhs.distance;
}

inline static bool operator < (hop_constrained_two_hop_label const &x, hop_constrained_two_hop_label const &y) {
	if (x.distance != y.distance) {
		return x.distance > y.distance; // < is the max-heap; > is the min heap
	} else {
		return x.hop > y.hop; // < is the max-heap; > is the min heap
	}
	// if (x.hub_vertex != y.hub_vertex) {
	// 	return x.hub_vertex < y.hub_vertex;
	// } else if (x.hop != y.hop) {
	// 	return x.hop < y.hop;
	// } else {
	// 	return x.distance < y.distance;
	// }
}

/*sortL*/
// inline bool compare_hop_constrained_two_hop_label(hop_constrained_two_hop_label &i, hop_constrained_two_hop_label &j) {
// 	if (i.hub_vertex != j.hub_vertex) {
// 		return i.hub_vertex < j.hub_vertex;
// 	} else if (i.hop != j.hop) {
// 		return i.hop < j.hop;
// 	} else {
// 		return i.distance < j.distance;
// 	}
// }

class hop_constrained_case_info {
public:
	/*hop bounded*/
	int thread_num = 1;
	int upper_k = 0;
	bool use_rank_prune = false;
	bool use_2023WWW_generation = false;
	bool use_2023WWW_generation_optimized = false;
	bool use_GPU_version_generation = false;
	bool use_GPU_version_generation_optimized = false;
	bool use_canonical_repair = 1;

	/*running time records*/
	double time_initialization = 0;
	double time_generate_labels = 0;
	double time_traverse = 0;
	double time_clear = 0;
	double time_sortL = 0;
	double time_canonical_repair = 0;
	double time_total = 0;
	double label_size = 0;
	
	/*running limits*/
	long long int max_bit_size = 1e12;
	double max_run_time_seconds = 1e12;

	/*labels*/
	std::vector<std::vector<hop_constrained_two_hop_label>> L;

	double label_size_before_canonical_repair, label_size_after_canonical_repair, canonical_repair_remove_label_ratio;

	void compute_label_size_per_node(int V) {
		for (auto &xx : L) {
			label_size = label_size + xx.size();
		}
		label_size = label_size / (double) V;
	}

	long long int compute_label_bit_size() {
		long long int size = 0;
		for (auto &xx : L) {
			size = size + xx.size() * sizeof(hop_constrained_two_hop_label);
		}
		return size;
	}

	/*clear labels*/
	void clear_labels() {
		std::vector<std::vector<hop_constrained_two_hop_label>>().swap(L);
	}

	void print_L() {
		std::cout << "print_L: (hub_vertex, hop, distance, parent_vertex)" << std::endl;
		for (auto &xx : L) {
			for (auto &yy : xx) {
				std::cout << "(" << yy.hub_vertex << "," << yy.hop << "," << yy.distance << "," << yy.parent_vertex << ") ";
			}
			std::cout << std::endl;
		}
	}

	/*record_all_details*/
	void record_all_details(std::string save_name) {
		std::ofstream outputFile;
		outputFile.precision(6);
		outputFile.setf(std::ios::fixed);
		outputFile.setf(std::ios::showpoint);
		outputFile.open(save_name + ".txt");

		outputFile << "hop_constrained_case_info:" << std::endl;
		outputFile << "thread_num=" << thread_num << std::endl;
		outputFile << "upper_k=" << upper_k << std::endl;
		// outputFile << "use_2M_prune=" << use_2M_prune << endl;
		outputFile << "use_2023WWW_generation=" << use_2023WWW_generation << std::endl;
		outputFile << "use_canonical_repair=" << use_canonical_repair << std::endl;

		outputFile << "time_initialization=" << time_initialization << std::endl;
		outputFile << "time_generate_labels=" << time_generate_labels << std::endl;
		outputFile << "time_sortL=" << time_sortL << std::endl;
		outputFile << "time_canonical_repair=" << time_canonical_repair << std::endl;
		outputFile << "time_total=" << time_total << std::endl;

		outputFile << "max_bit_size=" << max_bit_size << std::endl;
		outputFile << "max_run_time_seconds=" << max_run_time_seconds << std::endl;

		outputFile << "label_size_before_canonical_repair=" << label_size_before_canonical_repair << std::endl;
		outputFile << "label_size_after_canonical_repair=" << label_size_after_canonical_repair << std::endl;
		outputFile << "canonical_repair_remove_label_ratio=" << canonical_repair_remove_label_ratio << std::endl;

		outputFile << "compute_label_bit_size()=" << compute_label_bit_size() << std::endl;

		outputFile.close();
	}
};

inline int hop_constrained_extract_distance(std::vector<std::vector<hop_constrained_two_hop_label>> &L, int source, int terminal, int hop_cst) {

	/*return std::numeric_limits<int>::max() is not connected*/

	if (hop_cst < 0) {
		return std::numeric_limits<int>::max();
	}
	if (source == terminal) {
		return 0;
	} else if (hop_cst == 0) {
		return std::numeric_limits<int>::max();
	}

	long long int distance = std::numeric_limits<int>::max();
	auto vector1_check_pointer = L[source].begin();
	auto vector2_check_pointer = L[terminal].begin();
	auto pointer_L_s_end = L[source].end(), pointer_L_t_end = L[terminal].end();

	while (vector1_check_pointer != pointer_L_s_end && vector2_check_pointer != pointer_L_t_end) {
		
		if (vector1_check_pointer->hub_vertex == vector2_check_pointer->hub_vertex) {

			auto vector1_end = vector1_check_pointer;
			while (vector1_check_pointer->hub_vertex == vector1_end->hub_vertex && vector1_end != pointer_L_s_end) {
				vector1_end++;
			}
			auto vector2_end = vector2_check_pointer;
			while (vector2_check_pointer->hub_vertex == vector2_end->hub_vertex && vector2_end != pointer_L_t_end) {
				vector2_end++;
			}

			for (auto vector1_begin = vector1_check_pointer; vector1_begin != vector1_end; vector1_begin++) {
				// cout << "x (" << vector1_begin->hub_vertex << "," <<
				// vector1_begin->hop << "," << vector1_begin->distance << "," <<
				// vector1_begin->parent_vertex << ") " << endl;
				for (auto vector2_begin = vector2_check_pointer; vector2_begin != vector2_end; vector2_begin++) {
					// cout << "y (" << vector2_begin->hub_vertex << "," <<
					// vector2_begin->hop << "," << vector2_begin->distance << "," <<
					// vector2_begin->parent_vertex << ") " << endl;
					if (vector1_begin->hop + vector2_begin->hop <= hop_cst) {
						long long int dis = (long long int)vector1_begin->distance + vector2_begin->distance;
						if (distance > dis) {
							distance = dis;
						}
					} else {
						break;
					}
				}
			}

			vector1_check_pointer = vector1_end;
			vector2_check_pointer = vector2_end;
		} else if (vector1_check_pointer->hub_vertex > vector2_check_pointer->hub_vertex) {
			vector2_check_pointer++;
		} else {
			vector1_check_pointer++;
		}
	}

	return distance;
}

inline std::vector<std::pair<int, int>> hop_constrained_extract_shortest_path(std::vector<std::vector<hop_constrained_two_hop_label>> &L, int source, int terminal, int hop_cst) {

	std::vector<std::pair<int, int>> paths;

	if (source == terminal) {
		return paths;
	}

	/* Nothing happened */
	/* In this case, the problem that the removed vertices appear in the path
	 * needs to be solved */
	int vector1_capped_v_parent, vector2_capped_v_parent;
	long long int distance = std::numeric_limits<int>::max();
	auto vector1_check_pointer = L[source].begin();
	auto vector2_check_pointer = L[terminal].begin();
	auto pointer_L_s_end = L[source].end(), pointer_L_t_end = L[terminal].end();

	while (vector1_check_pointer != pointer_L_s_end && vector2_check_pointer != pointer_L_t_end) {
		if (vector1_check_pointer->hub_vertex == vector2_check_pointer->hub_vertex) {

			auto vector1_end = vector1_check_pointer;
			while (vector1_check_pointer->hub_vertex == vector1_end->hub_vertex && vector1_end != pointer_L_s_end) {
				vector1_end++;
			}
			auto vector2_end = vector2_check_pointer;
			while (vector2_check_pointer->hub_vertex == vector2_end->hub_vertex && vector2_end != pointer_L_t_end) {
				vector2_end++;
			}

			for (auto vector1_begin = vector1_check_pointer; vector1_begin != vector1_end; vector1_begin++) {
				for (auto vector2_begin = vector2_check_pointer; vector2_begin != vector2_end; vector2_begin++) {
					if (vector2_begin->hop + vector1_begin->hop <= hop_cst) {
						long long int dis = (long long int)vector1_begin->distance + vector2_begin->distance;
						if (distance > dis) {
							distance = dis;
							vector1_capped_v_parent = vector1_begin->parent_vertex;
							vector2_capped_v_parent = vector2_begin->parent_vertex;
						}
					} else {
						break;
					}
				}
			}

			vector1_check_pointer = vector1_end;
			vector2_check_pointer = vector2_end;
		} else if (vector1_check_pointer->hub_vertex > vector2_check_pointer->hub_vertex) {
			vector2_check_pointer++;
		} else {
			vector1_check_pointer++;
		}
	}

	if (distance < std::numeric_limits<int>::max()) { // connected
		if (source != vector1_capped_v_parent) {
			paths.push_back({source, vector1_capped_v_parent});
			source = vector1_capped_v_parent;
			hop_cst--;
		}
		if (terminal != vector2_capped_v_parent) {
			paths.push_back({terminal, vector2_capped_v_parent});
			terminal = vector2_capped_v_parent;
			hop_cst--;
		}
	} else {
		return paths;
	}

	// find new edges
	std::vector<std::pair<int, int>> new_edges = hop_constrained_extract_shortest_path(L, source, terminal, hop_cst);

	paths.insert(paths.end(), new_edges.begin(), new_edges.end());

	return paths;
}

#endif // HOP_CONSTRAINED_TWO_HOP_LABELS_H
