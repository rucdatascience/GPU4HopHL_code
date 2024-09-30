#pragma once
#include"vgroup/CT/two_hop_labels.hpp"
#include <unordered_map>
#include <unordered_set>
#include <limits.h>
#include<utility>
using namespace std;
struct node_degree {
	int vertex, degree;
	bool operator<(const node_degree& nd) const {
		return degree > nd.degree;
	}
};

class CT_case_info {
public:
	/*parameters*/
	int thread_num = 1;
	int d = 3;
	// set d as a large number to test the correctness of tree-index
	// set d as 0 to test the correctness of PLL on core
	// set d as a moderate number to test the correctness of the mixed labels
	bool use_P2H_pruning = 1; // more effective when d is small? (like 10)

	/*labels*/
	two_hop_case_info two_hop_info;
	std::vector<std::vector<pair<int, int>>> Bags;                   // bag nodes of decomposied tree
	vector<bool> isIntree;                       // isIntree[v]=1 means vertex v is in the tree_index
	vector<int> root;
	vector<vector<int>> tree_st;    // for lca
	vector<vector<int>> tree_st_r;  // for lca
	vector<int> first_pos;          // for lca
	vector<int> lg;                 // for lca
	vector<int> dep;                // for lca

	int tree_vertex_num = 0;
	int bag_size_before_P2H = 0;
	int bag_size_after_P2H = 0;

	/*running limits*/
	long long int max_bit_size = 1e12;
	double max_run_time_seconds = 1e12;

	/*indexing times*/
	double time_initialization = 0;
	double time_tree_decomposition = 0;
	double time_tree_indexs = 0;
	double time_P2H_pruning = 0;
	double time_lca = 0;
	double time_core_indexs = 0;
	double time_post = 0;
	double time_total = 0;

	/*compute label size*/
	long long int compute_label_bit_size() {
		long long int size = 0;
		size = size + two_hop_info.compute_label_bit_size();  // L includes both tree and core indexes
		for (auto it = Bags.begin(); it != Bags.end(); it++) {
			size = size + (*it).size() * sizeof(pair<int, int>);
		}
		size = size + isIntree.size() * sizeof(bool);
		size = size + root.size() * sizeof(int);
		for (auto it = tree_st.begin(); it != tree_st.end(); it++) {
			size = size + (*it).size() * sizeof(int);
		}
		for (auto it = tree_st_r.begin(); it != tree_st_r.end(); it++) {
			size = size + (*it).size() * sizeof(int);
		}
		size = size + first_pos.size() * sizeof(int);
		size = size + lg.size() * sizeof(int);
		size = size + dep.size() * sizeof(int);
		return size;
	}

	void compare_label_bit_size() {
		long long int size1 = 0, size2 = 0, size3 = 0;
		size1 += two_hop_info.compute_label_bit_size();  // L includes both tree and core indexes
		for (auto it = Bags.begin(); it != Bags.end(); it++) {
			size2 += (*it).size() * sizeof(pair<int, int>);
		}
		size3 += isIntree.size() * sizeof(bool);
		size3 += root.size() * sizeof(int);
		for (auto it = tree_st.begin(); it != tree_st.end(); it++) {
			size3 += (*it).size() * sizeof(int);
		}
		for (auto it = tree_st_r.begin(); it != tree_st_r.end(); it++) {
			size3 += (*it).size() * sizeof(int);
		}
		size3 += first_pos.size() * sizeof(int);
		size3 += lg.size() * sizeof(int);
		size3 += dep.size() * sizeof(int);
		cout << "size1: " << size1 << endl;
		cout << "size2: " << size2 << endl;
		cout << "size3: " << size3 << endl;
	}

	/*clear labels*/
	void clear_labels() {
		two_hop_info.clear_labels();
		std::vector<std::vector<pair<int, int>>>().swap(Bags);
		vector<bool>().swap(isIntree);
		vector<int>().swap(root);
		vector<vector<int>>().swap(tree_st);
		vector<vector<int>>().swap(tree_st_r);
		vector<int>().swap(first_pos);
		vector<int>().swap(lg);
		vector<int>().swap(dep);
	}

	/*printing*/
	void print_root() {
		cout << "print_root:" << endl;
		for (int i = 0; i < root.size(); i++) {
			cout << "root[" << i << "]=" << root[i] << endl;
		}
	}
	void print_isIntree() {
		cout << "print_isIntree:" << endl;
		for (int i = 0; i < isIntree.size(); i++) {
			cout << "isIntree[" << i << "]=" << isIntree[i] << endl;
		}
	}
	void print_times() {
		cout << "print_times:" << endl;
		cout << "time_initialization: " << time_initialization << "s" << endl;
		cout << "time_tree_decomposition: " << time_tree_decomposition << "s" << endl;
		cout << "time_tree_indexs: " << time_tree_indexs << "s" << endl;
		cout << "time_P2H_pruning: " << time_P2H_pruning << "s" << endl;
		cout << "time_lca: " << time_lca << "s" << endl;
		cout << "time_core_indexs: " << time_core_indexs << "s" << endl;
		cout << "time_post: " << time_post << "s" << endl;
		cout << "time_total: " << time_total << "s" << endl;
	}

	/*record_all_details*/
	void record_all_details(string save_name) {

		ofstream outputFile;
		outputFile.precision(6);
		outputFile.setf(ios::fixed);
		outputFile.setf(ios::showpoint);
		outputFile.open(save_name + ".txt");

		outputFile << "CT info:" << endl;
		outputFile << "thread_num=" << thread_num << endl;
		outputFile << "d=" << d << endl;
		outputFile << "use_P2H_pruning=" << use_P2H_pruning << endl;

		outputFile << "tree_vertex_num=" << tree_vertex_num << endl;
		outputFile << "bag_size_before_P2H=" << bag_size_before_P2H << endl;
		outputFile << "bag_size_after_P2H=" << bag_size_after_P2H << endl;

		outputFile << "max_bit_size=" << max_bit_size << endl;
		outputFile << "max_run_time_seconds=" << max_run_time_seconds << endl;

		outputFile << "time_initialization=" << time_initialization << endl;
		outputFile << "time_tree_decomposition=" << time_tree_decomposition << endl;
		outputFile << "time_tree_indexs=" << time_tree_indexs << endl;
		outputFile << "time_P2H_pruning=" << time_P2H_pruning << endl;
		outputFile << "time_lca=" << time_lca << endl;
		outputFile << "time_core_indexs=" << time_core_indexs << endl;
		outputFile << "time_post=" << time_post << endl;
		outputFile << "time_total=" << time_total << endl;

		outputFile << "compute_label_bit_size()=" << compute_label_bit_size() << endl;

		outputFile << endl;

		outputFile << "Core info:" << endl;
		outputFile << "two_hop_info.thread_num=" << two_hop_info.thread_num << endl;
		outputFile << "two_hop_info.use_2M_prune=" << two_hop_info.use_2M_prune << endl;
		outputFile << "two_hop_info.use_rank_prune=" << two_hop_info.use_rank_prune << endl;
		outputFile << "two_hop_info.use_canonical_repair=" << two_hop_info.use_canonical_repair << endl;

		outputFile << "two_hop_info.label_size_before_canonical_repair=" << two_hop_info.label_size_before_canonical_repair << endl;
		outputFile << "two_hop_info.label_size_after_canonical_repair=" << two_hop_info.label_size_after_canonical_repair << endl;
		outputFile << "two_hop_info.canonical_repair_remove_label_ratio=" << two_hop_info.canonical_repair_remove_label_ratio << endl;

		outputFile << "two_hop_info.time_initialization=" << two_hop_info.time_initialization << endl;
		outputFile << "two_hop_info.time_generate_labels=" << two_hop_info.time_generate_labels << endl;
		outputFile << "two_hop_info.time_sortL=" << two_hop_info.time_sortL << endl;
		outputFile << "two_hop_info.time_canonical_repair=" << two_hop_info.time_canonical_repair << endl;
		outputFile << "two_hop_info.time_total=" << two_hop_info.time_total << endl;

		outputFile << "two_hop_info.max_labal_bit_size=" << two_hop_info.max_labal_bit_size << endl;
		outputFile << "two_hop_info.max_run_time_seconds=" << two_hop_info.max_run_time_seconds << endl;

		outputFile << "two_hop_info.compute_label_bit_size()=" << two_hop_info.compute_label_bit_size() << endl;

		outputFile.close();
	}
};

/*
query function

note that, the predecessors in the tree part of L are true predecessors in the original graph,
while the predecessors in the core part of L are predecessors in the merged core-graph (true predecessors in the original graph should be retrived using search_midnode)
*/

static int lca(CT_case_info& case_info, int x, int y) {
	auto& first_pos = case_info.first_pos;

	if (first_pos[x] > first_pos[y]) {
		int t = x;
		x = y;
		y = t;
	}

	int len = first_pos[y] - first_pos[x] + 1;
	int j = case_info.lg[len];
	x = case_info.tree_st[first_pos[x]][j];
	y = case_info.tree_st_r[first_pos[y]][j];
	if (case_info.dep[x] < case_info.dep[y])
		return x;
	else
		return y;  // return the vertex with minimum depth between x and y in the dfs sequence.
}

static long long int extract_distance(CT_case_info& case_info, int source, int terminal) {
	auto& L = case_info.two_hop_info.L;
	auto& Bags = case_info.Bags;
	auto& isIntree = case_info.isIntree;
	auto& root = case_info.root;

	if (source == terminal) return 0;

	long long int distance = std::numeric_limits<int>::max();  // if disconnected, return this large value

	if (!isIntree[source] && !isIntree[terminal])  // both on core, use PLL indexs only
	{
		return extract_distance(case_info.two_hop_info, source, terminal);
	}
	else if ((!isIntree[source] && isIntree[terminal]) || (isIntree[source] && !isIntree[terminal]))  // one on core and one on tree
	{
		//cout << 2 << endl;
		// 3-hop, use the interface
		if (isIntree[source]) {
			int t = source;
			source = terminal;
			terminal = t;
		}
		/*the following: source is in core, terminal is in tree*/
		int r = root[terminal];
		int r_size = Bags[r].size();
		for (int i = 0; i < r_size; i++) {
			auto& xx = L[terminal][i];
			if (xx.distance > distance) continue;  // already exceed the present minimum distance
			int x = xx.vertex;
			long long int d_dis = xx.distance;
			int dis = extract_distance(case_info.two_hop_info, source, x);
			if (distance > dis + d_dis) distance = dis + d_dis;
		}
		return distance;
	}
	else if (root[source] != root[terminal])  // locate in different trees, have not used lamma 9 here yet
	{
		int r_s = root[source];
		int r_s_size = Bags[r_s].size();
		int r_t = root[terminal];
		int r_t_size = Bags[r_t].size();

		for (int i = 0; i < r_s_size; i++) {
			int u = L[source][i].vertex; // L[source]¡Á?????root??interface
			int d_s_u = L[source][i].distance;
			for (int j = 0; j < r_t_size; j++) {
				int w = L[terminal][j].vertex;
				int d_t_w = L[terminal][j].distance;
				int d_u_w = extract_distance(case_info.two_hop_info, u, w);
				long long int dis = (long long int)d_s_u + d_t_w + d_u_w;
				if (dis < distance) {
					distance = dis;
				}
			}
		}
		return distance;
	}
	else  // locate in same tree
	{
		int grand = lca(case_info, source, terminal);

		/*compute d2*/
		unordered_set<int> Bk = { grand };
		for (auto& xx : Bags[grand]) { // P2H is only useful in speeding up this step
			Bk.insert(xx.first);
		}
		std::unordered_map<int, int> Bk_source_dis;
		for (auto& xx : L[source]) {
			int v = xx.vertex;
			if (Bk.count(v) > 0) {
				Bk_source_dis[v] = xx.distance;
			}
		}
		for (auto& xx : L[terminal]) {
			int v = xx.vertex;
			if (Bk_source_dis.count(v) > 0) {
				long long int d2 = (long long int)Bk_source_dis[v] + xx.distance;
				if (distance > d2) {
					distance = d2;
				}
			}
		}

		/*compute d4*/
		int r = root[source];  // source and terminal have the same root
		int r_size = Bags[r].size();
		for (int i = 0; i < r_size; i++) {
			int u = L[source][i].vertex;
			int d_s_u = L[source][i].distance;
			for (int j = 0; j < r_size; j++) {
				int w = L[terminal][j].vertex;
				int d_t_w = L[terminal][j].distance;
				int d_u_w = extract_distance(case_info.two_hop_info, u, w);
				long long int d4 = (long long int)d_s_u + d_u_w + d_t_w;
				if (distance > d4) {
					distance = d4;
				}
			}
		}

		return distance;
	}
}

static void extract_shortest_path(CT_case_info& case_info, int source, int terminal, vector<pair<int, int>>& path) {
	/*may return INT_MAX, INT_MAX*/

	// cout << "extract_shortest_path " << source << " " << terminal << endl;
	// getchar();

	auto& root = case_info.root;
	auto& isIntree = case_info.isIntree;
	auto& Bags = case_info.Bags;
	auto& L = case_info.two_hop_info.L;

	/*if source and terminal are disconnected, then return empty vector; note that, if source==terminal, then also return empty vector*/
	if (source == terminal) return;

	long long int distance = std::numeric_limits<int>::max();  // record the shortest distance

	if (!isIntree[source] && !isIntree[terminal])  // both on core, use PLL indexs only
	{
		pair<int, int> added_edge = make_pair(INT_MAX, INT_MAX);
		pair<int, int> two_predecessors = two_hop_extract_two_predecessors(L, source, terminal);
		if (two_predecessors.first == source && two_predecessors.second == terminal) {  // disconnected
			path.push_back(make_pair(INT_MAX, INT_MAX));
			return;
		}

		//cout << "two_predecessors " << two_predecessors.first << " " << two_predecessors.second << endl;

		if (source != two_predecessors.first) {
			path.push_back({ source, two_predecessors.first });
			added_edge = { source, two_predecessors.first };
			source = two_predecessors.first;
		}
		if (terminal != two_predecessors.second) {
			if (!(added_edge.first == two_predecessors.second && added_edge.second == terminal)) {
				path.push_back({ two_predecessors.second, terminal });
				terminal = two_predecessors.second;  // else, two_predecessors.second == source, terminal should not change to source, since source has changed to terminal above
			}
			else {
				return;
			}
		}

		extract_shortest_path(case_info, source, terminal, path);
		return;
	}
	else if ((!isIntree[source] && isIntree[terminal]) || (isIntree[source] && !isIntree[terminal]))  // source on core and terminal on tree
	{
		// 3-hop, use the interface
		if (isIntree[source]) {
			int t = source;
			source = terminal;
			terminal = t;
		}
		/*the following: source is in core, terminal is in tree*/
		int r = root[terminal];
		int r_size = Bags[r].size();
		int terminal_predecessor = INT_MAX;
		for (int i = 0; i < r_size; i++) {
			if (L[terminal][i].distance > distance) continue;  // already exceed the present minimum distance

			int x = L[terminal][i].vertex;
			int d_dis = L[terminal][i].distance;
			long long int dis = extract_distance(case_info.two_hop_info, source, x);
			if (distance > dis + d_dis) {
				distance = dis + d_dis;
				terminal_predecessor = L[terminal][i].parent_vertex;  // terminal is in the tree, so this is the predecessor in the original graph
			}
		}
		if (terminal_predecessor == INT_MAX) {  // disconnected
			path.push_back(make_pair(INT_MAX, INT_MAX));
			return;
		}
		path.push_back({ terminal_predecessor, terminal });
		terminal = terminal_predecessor;
		extract_shortest_path(case_info, source, terminal, path);

		return;
	}
	else if (root[source] != root[terminal])  // locate in different trees
	{
		//cout << "c" << endl;
		int r_s = root[source];
		int r_s_size = Bags[r_s].size();
		int r_t = root[terminal];
		int r_t_size = Bags[r_t].size();
		int source_predecessor = INT_MAX, terminal_predecessor = INT_MAX;
		for (int i = 0; i < r_s_size; i++) {
			int u = L[source][i].vertex;
			int d_s_u = L[source][i].distance;
			for (int j = 0; j < r_t_size; j++) {
				int w = L[terminal][j].vertex;
				int d_t_w = L[terminal][j].distance;
				int d_u_w = extract_distance(case_info.two_hop_info, u, w);
				long long int dis = (long long int)d_s_u + d_t_w + d_u_w;
				if (dis < distance) {
					distance = dis;
					source_predecessor = L[source][i].parent_vertex;
					terminal_predecessor = L[terminal][j].parent_vertex;
				}
			}
		}
		if (source_predecessor == INT_MAX || terminal_predecessor == INT_MAX) {  // disconnected
			path.push_back(make_pair(INT_MAX, INT_MAX));
			return;
		}
		pair<int, int> added_edge = make_pair(INT_MAX, INT_MAX);
		if (source != source_predecessor) {
			path.push_back({ source, source_predecessor });
			added_edge = { source, source_predecessor };
			source = source_predecessor;
		}
		if (terminal != terminal_predecessor) {
			if (!(added_edge.first == terminal_predecessor && added_edge.second == terminal)) {
				path.push_back({ terminal_predecessor, terminal });
				terminal = terminal_predecessor;
			}
			else {
				return;
			}
		}
		extract_shortest_path(case_info, source, terminal, path);

		return;
	}
	else {
		//cout << "d" << endl;
		int source_predecessor = INT_MAX, terminal_predecessor = INT_MAX;

		int grand = lca(case_info, source, terminal);

		/*compute d2*/
		unordered_set<int> Bk = { grand };
		for (int i = Bags[grand].size() - 1; i >= 0; i--) {
			Bk.insert(Bags[grand][i].first);
		}
		unordered_map<int, pair<int, int>> Bk_source_dis;
		for (auto& xx : L[source]) {
			int v = xx.vertex;
			if (Bk.count(v) > 0) {
				Bk_source_dis[v] = { xx.distance, xx.parent_vertex };
			}
		}
		for (auto& xx : L[terminal]) {
			int v = xx.vertex;
			if (Bk_source_dis.count(v) > 0) {
				long long int d2 = (long long int)Bk_source_dis[v].first + xx.distance;
				if (distance > d2) {
					distance = d2;
					source_predecessor = Bk_source_dis[v].second;
					terminal_predecessor = xx.parent_vertex;
				}
			}
		}

		/*compute d4*/
		int r = root[source];  // source and terminal have the same root
		int r_size = Bags[r].size();
		for (int i = 0; i < r_size; i++) {
			int u = L[source][i].vertex;
			int d_s_u = L[source][i].distance;
			for (int j = 0; j < r_size; j++) {
				int w = L[terminal][j].vertex;
				int d_t_w = L[terminal][j].distance;
				int d_u_w = extract_distance(case_info.two_hop_info, u, w);
				long long int d4 = (long long int)d_s_u + d_u_w + d_t_w;
				if (distance > d4) {
					distance = d4;
					source_predecessor = L[source][i].parent_vertex;
					terminal_predecessor = L[terminal][j].parent_vertex;
				}
			}
		}

		if (source_predecessor == INT_MAX || terminal_predecessor == INT_MAX) {  // disconnected
			path.push_back(make_pair(INT_MAX, INT_MAX));
			return;
		}
		pair<int, int> added_edge = make_pair(INT_MAX, INT_MAX);
		if (source != source_predecessor) {
			path.push_back({ source, source_predecessor });
			added_edge = { source, source_predecessor };
			source = source_predecessor;
		}
		if (terminal != terminal_predecessor) {
			if (!(added_edge.first == terminal_predecessor && added_edge.second == terminal)) {
				path.push_back({ terminal_predecessor, terminal });
				terminal = terminal_predecessor;
			}
			else {
				return;
			}
		}
		extract_shortest_path(case_info, source, terminal, path);

		return;
	}
}