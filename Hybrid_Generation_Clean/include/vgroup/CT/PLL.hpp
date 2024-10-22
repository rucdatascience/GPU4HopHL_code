#ifndef PLL_H
#define PLL_H

#include "two_hop_labels.hpp"
#include "tool_functions/ThreadPool.h"
#include <shared_mutex>
#include <graph/graph_v_of_v.h>
#include <iostream>
#include <chrono>
#include <boost/heap/fibonacci_heap.hpp>
#include <vector>

using namespace std;

/*global values that should be cleared after usig PLL*/
string reach_limit_error_string_MB = "reach limit error MB";
string reach_limit_error_string_time = "reach limit error time";
long long int max_labal_num_595;
long long int labal_num_595;
auto begin_time_595 = std::chrono::high_resolution_clock::now();
double max_run_time_nanoseconds_595;
bool this_parallel_PLL_is_running_595 = false;
vector<vector<two_hop_label>> L_temp_595;
int max_N_ID_for_mtx_595 = 1e7;  // this is the max N to run
vector<shared_mutex> mtx_595(max_N_ID_for_mtx_595);  // std::mutex has no copy or move constructor, while std::vector::resize() requires that; you cannot resize mtx; moreover, do not change mtx to a pointer and then points to local values, it is very slow!!
queue<int> Qid_595; // IDs of available elements of P T
vector<vector<int>> P_dij_595;
vector<vector<int>> T_dij_595;
long long int label_size_before_canonical_repair_595 = 0;
long long int label_size_after_canonical_repair_595 = 0;
bool global_use_2M_prune = false, global_use_rank_prune = true;
int TwoM_value = 2 * 1e6; // suppose that dummy edge has a weight of 1e6


typedef typename boost::heap::fibonacci_heap<two_hop_label>::handle_type PLL_handle_t_for_sp;
vector<vector<PLL_handle_t_for_sp>> Q_handles_595;

static void PLL_clear_global_values() {
	this_parallel_PLL_is_running_595 = false;
	vector<vector<two_hop_label>>().swap(L_temp_595);
	queue<int>().swap(Qid_595);
	vector<vector<int>>().swap(P_dij_595);
	vector<vector<int>>().swap(T_dij_595);
	vector<vector<PLL_handle_t_for_sp>>().swap(Q_handles_595);
}


void PLL_dij_function(int v_k, graph_v_of_v<int>& input_graph) {

	/*Pruned Dijkstra from vertex v_k*/

	if (labal_num_595 > max_labal_num_595) {
		throw reach_limit_error_string_MB;  // after catching error, must call PLL_clear_global_values(), otherwise PLL cannot be reused
	}

	if (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin_time_595).count() > max_run_time_nanoseconds_595) {
		throw reach_limit_error_string_time;  // after catching error, must call PLL_clear_global_values(), otherwise PLL cannot be reused
	}

	mtx_595[max_N_ID_for_mtx_595 - 1].lock();
	int used_id = Qid_595.front();
	Qid_595.pop();
	mtx_595[max_N_ID_for_mtx_595 - 1].unlock();

	vector<int> P_changed_vertices, T_changed_vertices;
	vector<int>& T_dij = T_dij_595[used_id], P_dij = P_dij_595[used_id];
	vector<PLL_handle_t_for_sp>& Q_handles = Q_handles_595[used_id];

	boost::heap::fibonacci_heap<two_hop_label> Q;
	two_hop_label node;
	node.vertex = v_k;
	node.parent_vertex = v_k;
	node.distance = 0;
	Q_handles[v_k] = Q.push(node);
	P_dij[v_k] = 0;
	P_changed_vertices.push_back(v_k);

	mtx_595[v_k].lock_shared();
	for (auto xx : L_temp_595[v_k]) { //因为v-k的标签在从自己出发的过程中不会发生改变，并且在求query的过程中每次都会用到，所以可以提前取出来放在T数组，节省后面查找的时间
		int L_v_k_i_vertex = xx.vertex;
		T_dij[L_v_k_i_vertex] = xx.distance; //allocate T values for L_temp_595[v_k]
		T_changed_vertices.push_back(L_v_k_i_vertex);
	}
	mtx_595[v_k].unlock_shared();

	int new_label_num = 0;

	while (Q.size()) {

		node = Q.top();
		Q.pop();
		int u = node.vertex;

		if (!global_use_rank_prune || v_k <= u) { // this pruning condition is not in the original 2013 PLL paper
			int u_parent = node.parent_vertex;
			int P_u = node.distance;

			int query_v_k_u = std::numeric_limits<int>::max();
			mtx_595[u].lock_shared(); // put lock in for loop is very slow
			for (auto xx : L_temp_595[u]) {
				long long int dis = xx.distance + (long long int)T_dij[xx.vertex]; // long long int is to avoid overflow
				if (query_v_k_u > dis) { query_v_k_u = dis; }
			} //求query的值
			mtx_595[u].unlock_shared();

			if (P_u < query_v_k_u) {
				node.vertex = v_k;
				node.distance = P_u;
				node.parent_vertex = u_parent;

				mtx_595[u].lock();
				L_temp_595[u].push_back(node); // 并行时L_temp_595[u]里面的标签不一定是按照vertex ID排好序的，但是因为什么query时用了T_dij_595的trick，没必要让L_temp_595[u]里面的标签排好序
				mtx_595[u].unlock();
				new_label_num++;

				/*下面是dij更新邻接点的过程，同时更新优先队列和距离*/
				for (auto xx : input_graph.ADJs[u]) {
					int adj_v = xx.first, ec = xx.second;
					if (global_use_2M_prune && P_u + ec >= TwoM_value) {
						continue;
					}
					if (P_dij[adj_v] == std::numeric_limits<int>::max()) { //尚未到达的点
						node.vertex = adj_v;
						node.parent_vertex = u;
						node.distance = P_u + ec;
						Q_handles[adj_v] = Q.push(node);
						P_dij[adj_v] = node.distance;
						P_changed_vertices.push_back(adj_v);
					}
					else if (P_dij[adj_v] > P_u + ec) {
						node.vertex = adj_v;
						node.parent_vertex = u;
						node.distance = P_u + ec;
						Q.update(Q_handles[adj_v], node);
						P_dij[adj_v] = node.distance;
					}
				}
			}
		}
	}

	for (auto i : P_changed_vertices) {
		P_dij[i] = std::numeric_limits<int>::max(); // reverse-allocate P values
	}
	for (auto i : T_changed_vertices) {
		T_dij[i] = std::numeric_limits<int>::max(); // reverse-allocate T values
	}

	mtx_595[max_N_ID_for_mtx_595 - 1].lock();
	Qid_595.push(used_id);
	labal_num_595 = labal_num_595 + new_label_num;
	mtx_595[max_N_ID_for_mtx_595 - 1].unlock();
}


/*canonical_repair*/
static void clean_L(two_hop_case_info& case_info, int thread_num) {

	auto& L = case_info.L;
	int N = L.size();
	label_size_before_canonical_repair_595 = 0;
	label_size_after_canonical_repair_595 = 0;

	ThreadPool pool(thread_num);
	std::vector<std::future<int>> results;

	for (int v = 0; v < N; v++) {
		results.emplace_back(
			pool.enqueue([v, &L] { // pass const type value j to thread; [] can be empty

				mtx_595[max_N_ID_for_mtx_595 - 1].lock();
				int used_id = Qid_595.front();
				Qid_595.pop();
				mtx_595[max_N_ID_for_mtx_595 - 1].unlock();

				vector<two_hop_label> Lv_final;

				mtx_595[v].lock_shared();
				vector<two_hop_label> Lv = L[v];
				mtx_595[v].unlock_shared();
				label_size_before_canonical_repair_595 += Lv.size();

				auto& T = T_dij_595[used_id];

				for (auto Lvi : Lv) {
					int u = Lvi.vertex;
					if (v == u) {
						Lv_final.push_back(Lvi);
						T[v] = Lvi.distance;
						continue;
					}
					mtx_595[u].lock_shared();
					auto Lu = L[u];
					mtx_595[u].unlock_shared();

					int min_dis = std::numeric_limits<int>::max();
					for (auto label : Lu) {
						long long int query_dis = label.distance + (long long int)T[label.vertex];
						if (query_dis < min_dis) {
							min_dis = query_dis;
						}
					}

					if (min_dis > Lvi.distance) {
						Lv_final.push_back(Lvi);
						T[u] = Lvi.distance;
					}
				}

				for (auto label : Lv_final) {
					T[label.vertex] = std::numeric_limits<int>::max();
				}

				mtx_595[v].lock();
				L[v] = Lv_final;
				L[v].shrink_to_fit();
				mtx_595[v].unlock();
				label_size_after_canonical_repair_595 += Lv_final.size();

				mtx_595[max_N_ID_for_mtx_595 - 1].lock();
				Qid_595.push(used_id);
				mtx_595[max_N_ID_for_mtx_595 - 1].unlock();

				return 1; // return to results; the return type must be the same with results
				}));
	}

	for (auto&& result : results)
		result.get(); // all threads finish here
	results.clear();

	case_info.label_size_before_canonical_repair = label_size_before_canonical_repair_595;
	case_info.label_size_after_canonical_repair = label_size_after_canonical_repair_595;
	case_info.canonical_repair_remove_label_ratio = (double)(label_size_before_canonical_repair_595 - label_size_after_canonical_repair_595) / label_size_before_canonical_repair_595;
}


/*sortL*/
static bool compare_two_hop_label_small_to_large(two_hop_label& i, two_hop_label& j) {
	return i.vertex < j.vertex;  // < is from small to big; > is from big to small
}
static vector<vector<two_hop_label>> sortL(int num_of_threads) {

	/*time complexity: O(V*L*logL), where L is average number of labels per vertex*/

	int N = L_temp_595.size();
	vector<vector<two_hop_label>> output_L(N);

	/*time complexity: O(V*L*logL), where L is average number of labels per vertex*/
	ThreadPool pool(num_of_threads);
	std::vector< std::future<int> > results; // return typename: xxx
	for (int v_k = 0; v_k < N; v_k++) {
		results.emplace_back(
			pool.enqueue([&output_L, v_k] { // pass const type value j to thread; [] can be empty

				sort(L_temp_595[v_k].begin(), L_temp_595[v_k].end(), compare_two_hop_label_small_to_large);
				vector<two_hop_label>(L_temp_595[v_k]).swap(L_temp_595[v_k]); // swap释放vector中多余空间
				output_L[v_k] = L_temp_595[v_k];
				vector<two_hop_label>().swap(L_temp_595[v_k]); // clear new labels for RAM efficiency

				return 1; // return to results; the return type must be the same with results
				})
		);
	}
	for (auto&& result : results)
		result.get(); // all threads finish here

	return output_L;
}


/*the following parallel PLL_with_non_adj_reduction code cannot be run parallelly, due to the above globel values*/

static void PLL(graph_v_of_v<int>& input_graph, two_hop_case_info& case_info) {

	//----------------------------------- step 1: initialization ------------------------------------------------------------------
	auto begin1 = std::chrono::high_resolution_clock::now();

	/*information prepare*/
	begin_time_595 = std::chrono::high_resolution_clock::now();
	max_run_time_nanoseconds_595 = case_info.max_run_time_seconds * 1e9;
	labal_num_595 = 0;
	max_labal_num_595 = case_info.max_labal_bit_size / sizeof(two_hop_label);

	global_use_rank_prune = case_info.use_rank_prune;
	global_use_2M_prune = case_info.use_2M_prune;
	int num_of_threads = case_info.thread_num;

	int N = input_graph.ADJs.size();
	L_temp_595.resize(N);

	if (N > max_N_ID_for_mtx_595 || this_parallel_PLL_is_running_595 == true) {
		this_parallel_PLL_is_running_595 = true;
		cout << "unsuccessful start of PLL" << endl;
		exit(1);
	}

	auto end1 = std::chrono::high_resolution_clock::now();
	case_info.time_initialization = std::chrono::duration_cast<std::chrono::nanoseconds>(end1 - begin1).count() / 1e9; // s
	//---------------------------------------------------------------------------------------------------------------------------------------


	//----------------------------------------------- step 2: generate labels ---------------------------------------------------------------
	auto begin2 = std::chrono::high_resolution_clock::now();

	if (1) { // to save RAM of ThreadPool
		/*seaching shortest paths*/
		ThreadPool pool(num_of_threads);
		std::vector< std::future<int> > results; // return typename: xxx
		P_dij_595.resize(num_of_threads);
		T_dij_595.resize(num_of_threads);
		Q_handles_595.resize(num_of_threads);
		for (int i = 0; i < num_of_threads; i++) {
			P_dij_595[i].resize(N, std::numeric_limits<int>::max());
			T_dij_595[i].resize(N, std::numeric_limits<int>::max());
			Q_handles_595[i].resize(N);
			Qid_595.push(i);
		}
		for (int v_k = 0; v_k < N; v_k++) {
			results.emplace_back(
				pool.enqueue([v_k, &input_graph] { // pass const type value j to thread; [] can be empty
					PLL_dij_function(v_k, input_graph);
					return 1; // return to results; the return type must be the same with results
					})
			);
		}
		for (auto&& result : results)
			result.get(); //all threads finish here
	}

	auto end2 = std::chrono::high_resolution_clock::now();
	case_info.time_generate_labels = std::chrono::duration_cast<std::chrono::nanoseconds>(end2 - begin2).count() / 1e9; // s
	//---------------------------------------------------------------------------------------------------------------------------------------

	//----------------------------------------------- step 3: sortL ---------------------------------------------------------------
	auto begin3 = std::chrono::high_resolution_clock::now();

	case_info.L = sortL(num_of_threads);

	auto end3 = std::chrono::high_resolution_clock::now();
	case_info.time_sortL = std::chrono::duration_cast<std::chrono::nanoseconds>(end3 - begin3).count() / 1e9; // s
	//---------------------------------------------------------------------------------------------------------------------------------------


	//----------------------------------------------- step 3: canonical_repair ---------------------------------------------------------------
	auto begin4 = std::chrono::high_resolution_clock::now();

	if (case_info.use_canonical_repair) {
		clean_L(case_info, num_of_threads);
	}

	auto end4 = std::chrono::high_resolution_clock::now();
	case_info.time_canonical_repair = std::chrono::duration_cast<std::chrono::nanoseconds>(end4 - begin4).count() / 1e9; 
	//---------------------------------------------------------------------------------------------------------------------------------------

	case_info.time_total = case_info.time_initialization + case_info.time_generate_labels + case_info.time_sortL + case_info.time_canonical_repair;

	PLL_clear_global_values();
}
#endif // PLL_H