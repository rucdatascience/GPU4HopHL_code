#ifndef HOP_CONSTRAINED_TWO_HOP_LABELS_GENERATION_H
#define HOP_CONSTRAINED_TWO_HOP_LABELS_GENERATION_H
#pragma once

#include <HBPLL/hop_constrained_two_hop_labels.h>
#include <boost/heap/fibonacci_heap.hpp>
#include <graph_v_of_v/graph_v_of_v.h>
#include <shared_mutex>
#include <text_mining/ThreadPool.h>
#include <unordered_map>

#define num_of_threads_cpu 100

/*unique code for this file: 599*/
long long int max_labal_size_599;
long long int labal_size_599;
int max_N_ID_for_mtx_599 = 1e7;
double max_run_time_nanoseconds_599;
int global_upper_k;
long long int label_size_before_canonical_repair_599, label_size_after_canonical_repair_599;

int V;

auto begin_time_599 = std::chrono::high_resolution_clock::now();

vector<std::shared_timed_mutex> mtx_599(max_N_ID_for_mtx_599);

int num_of_threads;

graph_v_of_v<int> ideal_graph_599;

static std::vector<std::future<int>> results;
static ThreadPool pool(num_of_threads_cpu);

vector<vector<hop_constrained_two_hop_label>> L_temp_599;
vector<vector<vector<pair<int, int>>>> Temp_L_vk_599;
vector<vector<vector<int>>> Temp_L_vk_599_v2;
vector<vector<pair<int, int>>> dist_hop_599;
vector<vector<vector<int>>> Vh_599;

queue<int> Qid_599;

typedef typename boost::heap::fibonacci_heap<hop_constrained_two_hop_label>::handle_type hop_constrained_node_handle;

vector<vector<vector<pair<hop_constrained_node_handle, int>>>> Q_handle_priorities_599;

static void hop_constrained_clear_global_values() {
	vector<vector<hop_constrained_two_hop_label>>().swap(L_temp_599);
	// vector<vector<vector<pair<int, int>>>>().swap(Temp_L_vk_599);
	// vector<vector<pair<int, int>>>().swap(dist_hop_599);
	// vector<vector<vector<pair<hop_constrained_node_handle, int>>>>().swap(Q_handle_priorities_599);
	// vector<vector<vector<int>>>().swap(Vh_599);
	// queue<int>().swap(Qid_599);
}

boost::random::mt19937 qid_random_seed { static_cast<std::uint32_t>(std::time(0)) }; // Random seed
int global_label_generation_tag = 1;
std::vector<std::vector<std::tuple<int, int, int> > > T[2]; // exchange queue as a large queue
std::vector<std::vector<std::pair<int, int> > > temp_D;
// vector<std::shared_timed_mutex> mtx_LM_599(max_N_ID_for_mtx_599);
// vector<std::shared_timed_mutex> mtx_T_599(max_N_ID_for_mtx_599);
// std::map<std::tuple<int, int, int>, int> label_map[100005]; // <to_vertex, hub_vertex, hop>, distance
// static void HSDL_with_large_queue(int qid, int tag, int hop_now, int hop_cst) {
	
// 	if (labal_size_599 > max_labal_size_599)
// 	{
// 		// throw reach_limit_error_string_MB;
// 	}
// 	if (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin_time_599).count() > max_run_time_nanoseconds_599)
// 	{
// 		// throw reach_limit_error_string_time;
// 	}

// 	boost::random::uniform_int_distribution<> qid_range{ static_cast<int>(0), static_cast<int>(num_of_threads_cpu - 1) };

// 	int T_sz = T[qid][tag].size();
// 	std::tuple<int, int, int> item;
// 	for (int i = 0; i < T_sz; ++i) {
// 		item = T[qid][tag][i];

// 		// tranverse hub_vertex to to_vertex and the hop, distance
// 		// should judge this item and tranverse its adj.
// 		int hub_vertex = std::get<0>(item);
// 		int to_vertex = std::get<1>(item);
// 		int distance = std::get<2>(item);

// 		for (auto &xx : ideal_graph_599[to_vertex]) {
// 			int adj_v = xx.first, ec = xx.second;

// 			if (hub_vertex > adj_v) {
// 				continue;
// 			}

// 			// hub_vertex, adj_v, ec
// 			int L_sz = L_temp_599[hub_vertex].size();
// 			int query_v_k_u = std::numeric_limits<int>::max();
// 			for (int j = 0; j < L_sz; ++ j) {
// 				int common_v = L_temp_599[hub_vertex][j].hub_vertex;
// 				int common_hop = L_temp_599[hub_vertex][j].hop;
// 				int common_dis = L_temp_599[hub_vertex][j].distance;
// 				for (int k = hop_now - common_hop; k >= 0; --k) {
// 					mtx_LM_599[adj_v].lock();
// 					if (label_map[adj_v].find(std::make_tuple(common_v, k, 0)) != label_map[adj_v].end()) {
// 						query_v_k_u = min(query_v_k_u, label_map[adj_v][std::make_tuple(common_v, k, 0)] + common_dis);
// 					}
// 					mtx_LM_599[adj_v].unlock();
// 				}
// 			}

// 			if (query_v_k_u > distance + ec) {
// 				int rand_qid = qid_range(qid_random_seed);
// 				mtx_T_599[rand_qid].lock();
// 				T[rand_qid][tag ^ 1].push_back(std::make_tuple(hub_vertex, adj_v, distance + ec));
// 				mtx_T_599[rand_qid].unlock();

// 				mtx_599[adj_v].lock();
// 				L_temp_599[adj_v].push_back(hop_constrained_two_hop_label(hub_vertex, 0, hop_now, distance + ec));
// 				mtx_599[adj_v].unlock();
				
// 				mtx_LM_599[adj_v].lock();
// 				label_map[adj_v][make_tuple(hub_vertex, hop_now, 0)] = distance + ec;
// 				mtx_LM_599[adj_v].unlock();
// 			}
// 		}
// 	}
// 	// vector<tuple<int, int, int>>().swap(T[qid][tag]);
// 	T[qid][tag].clear();

// }
vector<int> Last_iter_label_size;

static void get_Last_iter_label_size(int v_k) {
	Last_iter_label_size[v_k] = L_temp_599[v_k].size();
}

static void HSDL_with_large_queue(int v_k, int tag, int hop_now, int hop_cst) {	
	
	if (labal_size_599 > max_labal_size_599)
	{
		// throw reach_limit_error_string_MB;
	}
	if (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin_time_599).count() > max_run_time_nanoseconds_599)
	{
		// throw reach_limit_error_string_time;
	}

	mtx_599[max_N_ID_for_mtx_599 - 1].lock();
	int used_id = Qid_599.front();
	Qid_599.pop();
	mtx_599[max_N_ID_for_mtx_599 - 1].unlock();

	vector<int> Temp_L_vk_changes;
	vector<tuple<int, int, int> > D;

	auto &Temp_D = temp_D[used_id];
	// auto &Temp_L_vk = Temp_L_vk_599[used_id];
	auto &Temp_L_vk = Temp_L_vk_599[used_id];
	auto &T0 = T[tag][v_k];
	auto &T1 = T[tag ^ 1][v_k];

	/* Temp_L_vk_599 stores the label (dist and hop) of vertex v_k */
	// L_temp_599[v_k].push_back(node); new_label_num++;
	hop_constrained_two_hop_label xx;
	mtx_599[v_k].lock();
	for (auto &xx : L_temp_599[v_k]) {
		int L_vk_vertex = xx.hub_vertex;
		Temp_L_vk[L_vk_vertex].push_back({xx.distance, xx.hop});
		Temp_L_vk_changes.push_back(L_vk_vertex);
	}
	mtx_599[v_k].unlock();

	for (auto &item : T0) {
		int to_vertex = std::get<0>(item);
		int distance = std::get<1>(item);

		for (auto &xx : ideal_graph_599[to_vertex]) {
			int adj_v = xx.first, ec = xx.second;
			if (v_k > adj_v) {
				continue;
			}

			int new_dis = distance + ec;
			if (new_dis < Temp_D[adj_v].first) {
				if (Temp_D[adj_v].first == std::numeric_limits<int>::max()) {
					Temp_D[adj_v].first = new_dis;
					D.push_back(std::make_tuple(adj_v, new_dis, to_vertex));
					Temp_D[adj_v].second = D.size() - 1;
				} else {
					Temp_D[adj_v].first = new_dis;
					D[Temp_D[adj_v].second] = (std::make_tuple(adj_v, new_dis, to_vertex));
				}
			}
		}
	}
	
	// printf("%d\n", T[v_k][tag ^ 1].size());
	for (auto &xxx : D) {
		
		long long query_v_k_u = std::numeric_limits<int>::max();

		mtx_599[std::get<0>(xxx)].lock();
		for (auto &xx : L_temp_599[std::get<0>(xxx)]) {
			int common_v = xx.hub_vertex;
			for (auto &yy : Temp_L_vk[common_v]) {
				if (xx.hop + yy.second <= hop_now) {
					int dis = xx.distance + yy.first;
					if (query_v_k_u > dis) {
						query_v_k_u = dis;
					}
				}
			}
		}
		mtx_599[std::get<0>(xxx)].unlock();

		if (std::get<1>(xxx) < query_v_k_u) {
			mtx_599[std::get<0>(xxx)].lock();
			L_temp_599[std::get<0>(xxx)].push_back(
				hop_constrained_two_hop_label(v_k, std::get<2>(xxx), hop_now, std::get<1>(xxx)));
			mtx_599[std::get<0>(xxx)].unlock();

			T1.push_back(std::make_tuple(std::get<0>(xxx), std::get<1>(xxx), std::get<2>(xxx)));
		}

		Temp_D[std::get<0>(xxx)].first = std::numeric_limits<int>::max();
	}

	if (D.size()) global_label_generation_tag = 1;

	for (auto &xx : Temp_L_vk_changes) {
		vector<pair<int, int>>().swap(Temp_L_vk[xx]);
	}

	mtx_599[v_k].lock();
	vector<hop_constrained_two_hop_label>(L_temp_599[v_k]).swap(L_temp_599[v_k]);
	mtx_599[v_k].unlock();

	mtx_599[max_N_ID_for_mtx_599 - 1].lock();
	Qid_599.push(used_id);
	mtx_599[max_N_ID_for_mtx_599 - 1].unlock();

	// T[v_k][tag].clear();
	vector<tuple<int, int, int>>().swap(T0);
}

static void HSDL_with_large_queue_optimized(int v_k, int tag, int hop_now, int hop_cst) {	
	
	if (labal_size_599 > max_labal_size_599)
	{
		// throw reach_limit_error_string_MB;
	}
	if (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin_time_599).count() > max_run_time_nanoseconds_599)
	{
		// throw reach_limit_error_string_time;
	}

	mtx_599[max_N_ID_for_mtx_599 - 1].lock();
	int used_id = Qid_599.front();
	Qid_599.pop();
	mtx_599[max_N_ID_for_mtx_599 - 1].unlock();

	vector<int> Temp_L_vk_changes;
	vector<tuple<int, int, int> > D;

	auto &Temp_D = temp_D[used_id];
	// auto &Temp_L_vk = Temp_L_vk_599[used_id];
	auto &Temp_L_vk = Temp_L_vk_599_v2[used_id];
	auto &T0 = T[tag][v_k];
	auto &T1 = T[tag ^ 1][v_k];

	/* Temp_L_vk_599 stores the label (dist and hop) of vertex v_k */
	// L_temp_599[v_k].push_back(node); new_label_num++;
	hop_constrained_two_hop_label xx;
	mtx_599[v_k].lock();
	for (auto &xx : L_temp_599[v_k]) {
		int L_vk_vertex = xx.hub_vertex;
		// for (int k = xx.hop; k <= global_upper_k; k++) {
		// 	Temp_L_vk[L_vk_vertex][k] = min(Temp_L_vk[L_vk_vertex][k], xx.distance);
		// }
		for (int k = xx.hop; k <= global_upper_k; k++) {
			if (Temp_L_vk[L_vk_vertex][k] > xx.distance) {
				Temp_L_vk[L_vk_vertex][k] = xx.distance;
			} else {
				break;
			}
		}
		Temp_L_vk_changes.push_back(L_vk_vertex);
	}
	mtx_599[v_k].unlock();

	for (auto &item : T0) {
		int to_vertex = std::get<0>(item);
		int distance = std::get<1>(item);

		for (auto &xx : ideal_graph_599[to_vertex]) {
			int adj_v = xx.first, ec = xx.second;
			if (v_k > adj_v) {
				continue;
			}

			int new_dis = distance + ec;
			if (new_dis < Temp_D[adj_v].first) {
				if (Temp_D[adj_v].first == std::numeric_limits<int>::max()) {
					Temp_D[adj_v].first = new_dis;
					D.push_back(std::make_tuple(adj_v, new_dis, to_vertex));
					Temp_D[adj_v].second = D.size() - 1;
				} else {
					Temp_D[adj_v].first = new_dis;
					D[Temp_D[adj_v].second] = (std::make_tuple(adj_v, new_dis, to_vertex));
				}
			}
		}
	}
	
	// printf("%d\n", T[v_k][tag ^ 1].size());
	for (auto &xxx : D) {
		
		long long query_v_k_u = std::numeric_limits<int>::max();

		mtx_599[std::get<0>(xxx)].lock();
		for (auto &xx : L_temp_599[std::get<0>(xxx)]) {
			int common_v = xx.hub_vertex;
			// for (auto &yy : Temp_L_vk[common_v]) {
				// if (xx.hop + yy.second <= hop_now) {
				// 	int dis = xx.distance + yy.first;
				// 	if (query_v_k_u > dis) {
				// 		query_v_k_u = dis;
				// 	}
				// }
				if (hop_now >= xx.hop) {
					query_v_k_u = min(query_v_k_u, (long long) xx.distance + Temp_L_vk[common_v][hop_now - xx.hop]);
					// printf("Temp_L_vk[common_v][h - xx.hop]: %d %d %d %d!\n", v_k, common_v, xx.hop, Temp_L_vk[common_v][h - xx.hop]);
				}
			// }
		}
		mtx_599[std::get<0>(xxx)].unlock();

		if (std::get<1>(xxx) < query_v_k_u) {
			mtx_599[std::get<0>(xxx)].lock();
			L_temp_599[std::get<0>(xxx)].push_back(
				hop_constrained_two_hop_label(v_k, std::get<2>(xxx), hop_now, std::get<1>(xxx)));
			mtx_599[std::get<0>(xxx)].unlock();

			T1.push_back(std::make_tuple(std::get<0>(xxx), std::get<1>(xxx), std::get<2>(xxx)));
		}

		Temp_D[std::get<0>(xxx)].first = std::numeric_limits<int>::max();
	}

	if (D.size()) global_label_generation_tag = 1;

	for (auto &xx : Temp_L_vk_changes) {
		for (int k = 0; k <= global_upper_k; ++ k) {
			Temp_L_vk[xx][k] = std::numeric_limits<int>::max();
		}
		// vector<pair<int, int>>().swap(Temp_L_vk[xx]);
	}

	mtx_599[v_k].lock();
	vector<hop_constrained_two_hop_label>(L_temp_599[v_k]).swap(L_temp_599[v_k]);
	mtx_599[v_k].unlock();

	mtx_599[max_N_ID_for_mtx_599 - 1].lock();
	Qid_599.push(used_id);
	mtx_599[max_N_ID_for_mtx_599 - 1].unlock();

	// T[v_k][tag].clear();
	vector<tuple<int, int, int>>().swap(T0);
}

static void HSDL_thread_function(int v_k) {

	if (labal_size_599 > max_labal_size_599)
	{
		// throw reach_limit_error_string_MB;
	}
	if (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin_time_599).count() > max_run_time_nanoseconds_599)
	{
		// throw reach_limit_error_string_time;
	}

	/* get unique thread id */
	mtx_599[max_N_ID_for_mtx_599 - 1].lock();
	int used_id = Qid_599.front();
	Qid_599.pop();
	mtx_599[max_N_ID_for_mtx_599 - 1].unlock();

	vector<int> Temp_L_vk_changes, dist_hop_changes;
	auto &Temp_L_vk = Temp_L_vk_599[used_id];
	auto &dist_hop = dist_hop_599[used_id]; // record the minimum distance (and the
											// corresponding hop) of a searched vertex in Q
	vector<pair<int, int>> Q_handle_priorities_changes;
	auto &Q_handle_priorities = Q_handle_priorities_599[used_id];

	long long int new_label_num = 0;

	boost::heap::fibonacci_heap<hop_constrained_two_hop_label> Q;

	hop_constrained_two_hop_label node;
	node.hub_vertex = v_k;
	node.parent_vertex = v_k;
	node.hop = 0;
	node.distance = 0;
	Q_handle_priorities[v_k][0] = {Q.push({node}), node.distance};
	Q_handle_priorities_changes.push_back({v_k, 0});

	/* Temp_L_vk_599 stores the label (dist and hop) of vertex v_k */
	mtx_599[v_k].lock();
	L_temp_599[v_k].push_back(node);
	new_label_num++;
	for (auto &xx : L_temp_599[v_k])
	{
		int L_vk_vertex = xx.hub_vertex;
		Temp_L_vk[L_vk_vertex].push_back({xx.distance, xx.hop});
		Temp_L_vk_changes.push_back(L_vk_vertex);
	}
	mtx_599[v_k].unlock();

	/*  dist_hop_599 stores the shortest distance from vk to any other vertices
	   with its hop_cst, note that the hop_cst is determined by the shortest
	   distance */
	dist_hop[v_k] = {0, 0};
	dist_hop_changes.push_back(v_k);

	while (Q.size() > 0)
	{

		node = Q.top();
		Q.pop();
		int u = node.hub_vertex;

		if (v_k > u)
		{
			continue;
		}

		int u_parent = node.parent_vertex;
		int u_hop = node.hop;
		int P_u = node.distance;

		int query_v_k_u = std::numeric_limits<int>::max();
		mtx_599[u].lock();
		for (auto &xx : L_temp_599[u])
		{
			int common_v = xx.hub_vertex;
			for (auto &yy : Temp_L_vk[common_v])
			{
				if (xx.hop + yy.second <= u_hop)
				{
					long long int dis = (long long int)xx.distance + yy.first;
					if (query_v_k_u > dis)
					{
						query_v_k_u = dis;
					}
				}
			}
		}
		mtx_599[u].unlock();

		if (P_u < query_v_k_u || query_v_k_u == 0)
		{ // query_v_k_u == 0 is to start the while loop by
			// searching neighbors of v_k

			if (P_u < query_v_k_u)
			{
				node.hub_vertex = v_k;
				node.hop = u_hop;
				node.distance = P_u;
				node.parent_vertex = u_parent;
				mtx_599[u].lock();
				L_temp_599[u].push_back(node);
				mtx_599[u].unlock();
				new_label_num++;
			}

			if (u_hop + 1 > global_upper_k)
			{
				continue;
			}

			/* update adj */
			for (auto &xx : ideal_graph_599[u])
			{
				int adj_v = xx.first, ec = xx.second;

				/* update node info */
				node.hub_vertex = adj_v;
				node.parent_vertex = u;
				node.distance = P_u + ec;
				node.hop = u_hop + 1;

				auto &yy = Q_handle_priorities[adj_v][node.hop];

				/*directly using the following codes without dist_hop is OK, but is
				 * slower; dist_hop is a pruning technique without increasing the time
				 * complexity*/
				if (yy.second != std::numeric_limits<int>::max())
				{
					if (yy.second > node.distance)
					{
						Q.update(yy.first, node);
						yy.second = node.distance;
					}
				}
				else
				{
					yy = {Q.push(node), node.distance};
					Q_handle_priorities_changes.push_back({adj_v, node.hop});
				}
			}
		}
	}

	for (auto &xx : Temp_L_vk_changes)
	{
		vector<pair<int, int>>().swap(Temp_L_vk[xx]);
	}
	for (auto &xx : dist_hop_changes)
	{
		dist_hop[xx] = {std::numeric_limits<int>::max(), 0};
	}
	hop_constrained_node_handle handle_x;
	for (auto &xx : Q_handle_priorities_changes)
	{
		Q_handle_priorities[xx.first][xx.second] = {handle_x, std::numeric_limits<int>::max()};
	}

	mtx_599[v_k].lock();
	vector<hop_constrained_two_hop_label>(L_temp_599[v_k]).swap(L_temp_599[v_k]);
	mtx_599[v_k].unlock();

	mtx_599[max_N_ID_for_mtx_599 - 1].lock();
	Qid_599.push(used_id);
	labal_size_599 = labal_size_599 + new_label_num;
	mtx_599[max_N_ID_for_mtx_599 - 1].unlock();
}

static void _2023WWW_thread_function(int v_k) {
	// puts("2023WWW");
	if (labal_size_599 > max_labal_size_599)
	{
		// throw reach_limit_error_string_MB;
	}
	if (std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::high_resolution_clock::now() - begin_time_599)
			.count() > max_run_time_nanoseconds_599)
	{
		// throw reach_limit_error_string_time;
	}

	/* get unique thread id */
	mtx_599[max_N_ID_for_mtx_599 - 1].lock();
	int used_id = Qid_599.front();
	Qid_599.pop();
	mtx_599[max_N_ID_for_mtx_599 - 1].unlock();

	vector<int> Temp_L_vk_changes, dist_hop_changes;
	auto &Temp_L_vk = Temp_L_vk_599[used_id];
	auto &dist_hop = dist_hop_599[used_id]; // record {dis, predecessor}
	auto &Vh = Vh_599[used_id];

	long long int new_label_num = 0;

	hop_constrained_two_hop_label node;
	node.hub_vertex = v_k;
	node.parent_vertex = v_k;
	node.hop = 0;
	node.distance = 0;

	/* Temp_L_vk_599 stores the label (dist and hop) of vertex v_k */
	mtx_599[v_k].lock();
	// L_temp_599[v_k].push_back(node); new_label_num++;
	for (auto &xx : L_temp_599[v_k])
	{
		int L_vk_vertex = xx.hub_vertex;
		Temp_L_vk[L_vk_vertex].push_back({xx.distance, xx.hop});
		Temp_L_vk_changes.push_back(L_vk_vertex);
	}
	// Temp_L_vk[v_k].push_back({0, 0});
	mtx_599[v_k].unlock();

	Vh[0].push_back(v_k);

	dist_hop[v_k] = {0, v_k};
	dist_hop_changes.push_back(v_k);

	vector<tuple<int, int, int>> dh_updates;
	map<int, int> mp;

	for (int h = 0; h <= global_upper_k; h++)
	{

		for (auto &xx : dh_updates)
		{
			if (dist_hop[get<0>(xx)].first > get<1>(xx))
			{
				dist_hop[get<0>(xx)] = {get<1>(xx), get<2>(xx)};
				dist_hop_changes.push_back(get<0>(xx));
			}
		}
		vector<tuple<int, int, int>>().swap(dh_updates);

		for (auto u : Vh[h])
		{
			int P_u = dist_hop[u].first;

			if (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin_time_599).count() > max_run_time_nanoseconds_599)
			{
				// throw reach_limit_error_string_time;
			}

			// if (u < v_k) { // rank pruning
			// 	// 貌似并不能明显加速_2023WWW_thread_function，_2023WWW_thread_function之所以慢是因为BFS生成了过多的冗余的需要被清洗的label（冗余的比非冗余的多几乎一个数量级）
			// 	continue;
			// }

			int query_v_k_u = std::numeric_limits<int>::max();
			mtx_599[u].lock();
			for (auto &xx : L_temp_599[u])
			{
				int common_v = xx.hub_vertex;
				for (auto &yy : Temp_L_vk[common_v])
				{
					if (xx.hop + yy.second <= h)
					{
						long long int dis = (long long int)xx.distance + yy.first;
						if (query_v_k_u > dis)
						{
							query_v_k_u = dis;
						}
					}
				}
			}
			mtx_599[u].unlock();

			// printf("v_k, h, u, P_u, query_v_k_u: %d, %d, %d, %d, %d\n", v_k, h, u, P_u, query_v_k_u);

			if (P_u < query_v_k_u)
			{

				node.hub_vertex = v_k;
				node.hop = h;
				node.distance = P_u;
				node.parent_vertex = dist_hop[u].second;
				mtx_599[u].lock();
				L_temp_599[u].push_back(node);
				mtx_599[u].unlock();
				new_label_num++;

				/* update adj */
				for (auto &xx : ideal_graph_599[u])
				{
					int adj_v = xx.first, ec = xx.second;
					if (P_u + ec < dist_hop[adj_v].first)
					{
						if (mp.find(adj_v) == mp.end()) {
							mp[adj_v] = 1;
							Vh[h + 1].push_back(adj_v);
						}
						dh_updates.push_back({adj_v, P_u + ec, u});
					}
				}
			}
		}
		mp.clear();
	}

	for (auto &xx : Temp_L_vk_changes)
	{
		vector<pair<int, int>>().swap(Temp_L_vk[xx]);
	}
	for (int i = 0; i <= global_upper_k; i++)
	{
		vector<int>().swap(Vh[i]);
	}
	for (auto &xx : dist_hop_changes)
	{
		dist_hop[xx] = {std::numeric_limits<int>::max(), 0};
	}

	mtx_599[v_k].lock();
	vector<hop_constrained_two_hop_label>(L_temp_599[v_k]).swap(L_temp_599[v_k]);
	mtx_599[v_k].unlock();

	mtx_599[max_N_ID_for_mtx_599 - 1].lock();
	Qid_599.push(used_id);
	labal_size_599 = labal_size_599 + new_label_num;
	mtx_599[max_N_ID_for_mtx_599 - 1].unlock();
}

static void _2023WWW_thread_function_optimized(int v_k) {
	// puts("2023WWW");
	if (labal_size_599 > max_labal_size_599)
	{
		// throw reach_limit_error_string_MB;
	}
	if (std::chrono::duration_cast<std::chrono::nanoseconds>(
			std::chrono::high_resolution_clock::now() - begin_time_599)
			.count() > max_run_time_nanoseconds_599)
	{
		// throw reach_limit_error_string_time;
	}

	/* get unique thread id */
	mtx_599[max_N_ID_for_mtx_599 - 1].lock();
	int used_id = Qid_599.front();
	Qid_599.pop();
	mtx_599[max_N_ID_for_mtx_599 - 1].unlock();

	vector<int> Temp_L_vk_changes, dist_hop_changes;
	auto &Temp_L_vk = Temp_L_vk_599_v2[used_id];
	auto &dist_hop = dist_hop_599[used_id]; // record {dis, predecessor}
	auto &Vh = Vh_599[used_id];

	long long int new_label_num = 0;

	hop_constrained_two_hop_label node;
	node.hub_vertex = v_k;
	node.parent_vertex = v_k;
	node.hop = 0;
	node.distance = 0;

	/* Temp_L_vk_599 stores the label (dist and hop) of vertex v_k */
	mtx_599[v_k].lock();
	// L_temp_599[v_k].push_back(node); new_label_num++;
	for (auto &xx : L_temp_599[v_k]) {
		int L_vk_vertex = xx.hub_vertex;
		for (int k = xx.hop; k <= global_upper_k; k++) {
			if (Temp_L_vk[L_vk_vertex][k] > xx.distance) {
				Temp_L_vk[L_vk_vertex][k] = xx.distance;
			} else {
				break;
			}
		}
		Temp_L_vk_changes.push_back(L_vk_vertex);
	}
	// Temp_L_vk[v_k].push_back({0, 0});
	mtx_599[v_k].unlock();

	Vh[0].push_back(v_k);

	dist_hop[v_k] = {0, v_k};
	dist_hop_changes.push_back(v_k);

	vector<tuple<int, int, int>> dh_updates;
	map<int, int> mp;

	for (int h = 0; h <= global_upper_k; h++)
	{

		for (auto &xx : dh_updates)
		{
			if (dist_hop[get<0>(xx)].first > get<1>(xx))
			{
				dist_hop[get<0>(xx)] = {get<1>(xx), get<2>(xx)};
				dist_hop_changes.push_back(get<0>(xx));
			}
		}
		vector<tuple<int, int, int>>().swap(dh_updates);

		for (auto u : Vh[h])
		{
			int P_u = dist_hop[u].first;

			if (std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::high_resolution_clock::now() - begin_time_599).count() > max_run_time_nanoseconds_599)
			{
				// throw reach_limit_error_string_time;
			}

			// if (u < v_k) { // rank pruning
			// 	// 貌似并不能明显加速_2023WWW_thread_function，_2023WWW_thread_function之所以慢是因为BFS生成了过多的冗余的需要被清洗的label（冗余的比非冗余的多几乎一个数量级）
			// 	continue;
			// }

			long long query_v_k_u = std::numeric_limits<int>::max();
			mtx_599[u].lock();
			for (auto &xx : L_temp_599[u])
			{
				int common_v = xx.hub_vertex;
				// for (auto &yy : Temp_L_vk[common_v])
				// {
					// if (xx.hop + yy.second <= h)
					// {
					// 	long long int dis = (long long int)xx.distance + yy.first;
					// 	if (query_v_k_u > dis)
					// 	{
					// 		query_v_k_u = dis;
					// 	}
					// }
				if (h >= xx.hop) {
					query_v_k_u = min(query_v_k_u, (long long) xx.distance + Temp_L_vk[common_v][h - xx.hop]);
					// printf("Temp_L_vk[common_v][h - xx.hop]: %d %d %d %d!\n", v_k, common_v, xx.hop, Temp_L_vk[common_v][h - xx.hop]);
				}
				// }
			}
			mtx_599[u].unlock();

			// printf("v_k, h, u, P_u, query_v_k_u: %d, %d, %d, %d, %d\n", v_k, h, u, P_u, query_v_k_u);

			if (P_u < query_v_k_u)
			{

				node.hub_vertex = v_k;
				node.hop = h;
				node.distance = P_u;
				node.parent_vertex = dist_hop[u].second;
				mtx_599[u].lock();
				L_temp_599[u].push_back(node);
				mtx_599[u].unlock();
				new_label_num++;

				/* update adj */
				for (auto &xx : ideal_graph_599[u])
				{
					int adj_v = xx.first, ec = xx.second;
					if (P_u + ec < dist_hop[adj_v].first)
					{
						if (mp.find(adj_v) == mp.end()) {
							mp[adj_v] = 1;
							Vh[h + 1].push_back(adj_v);
						}
						dh_updates.push_back({adj_v, P_u + ec, u});
					}
				}
			}
		}
		mp.clear();
	}

	for (auto &xx : Temp_L_vk_changes) {
		for (int k = 0; k <= global_upper_k; ++ k) {
			Temp_L_vk[xx][k] = std::numeric_limits<int>::max();
		}
		// vector<int>().swap(Temp_L_vk[xx]);
	}
	for (int i = 0; i <= global_upper_k; i++)
	{
		vector<int>().swap(Vh[i]);
	}
	for (auto &xx : dist_hop_changes)
	{
		dist_hop[xx] = {std::numeric_limits<int>::max(), 0};
	}

	mtx_599[v_k].lock();
	vector<hop_constrained_two_hop_label>(L_temp_599[v_k]).swap(L_temp_599[v_k]);
	mtx_599[v_k].unlock();

	mtx_599[max_N_ID_for_mtx_599 - 1].lock();
	Qid_599.push(used_id);
	labal_size_599 = labal_size_599 + new_label_num;
	mtx_599[max_N_ID_for_mtx_599 - 1].unlock();
}

/*sortL*/
static bool compare_hop_constrained_two_hop_label(hop_constrained_two_hop_label &i, hop_constrained_two_hop_label &j) {
	if (i.hub_vertex != j.hub_vertex) {
		return i.hub_vertex < j.hub_vertex;
	} else if (i.hop != j.hop) {
		return i.hop < j.hop;
	} else {
		return i.distance < j.distance;
	}
}

static vector<vector<hop_constrained_two_hop_label>> hop_constrained_sortL(int num_of_threads) {

	/*time complexity: O(V*L*logL), where L is average number of labels per
	 * vertex*/

	int N = L_temp_599.size();
	vector<vector<hop_constrained_two_hop_label>> output_L(N);

	/*time complexity: O(V*L*logL), where L is average number of labels per
	 * vertex*/
	// ThreadPool pool(num_of_threads);
	// std::vector<std::future<int>> results; // return typename: xxx

	for (int v_k = 0; v_k < N; v_k++)
	{
		results.emplace_back(pool.enqueue(
			[&output_L, v_k] { // pass const type value j to thread; [] can be empty
				sort(L_temp_599[v_k].begin(), L_temp_599[v_k].end(), compare_hop_constrained_two_hop_label);
				vector<hop_constrained_two_hop_label>(L_temp_599[v_k]).swap(L_temp_599[v_k]); // swap释放vector中多余空间
				output_L[v_k] = L_temp_599[v_k];
				vector<hop_constrained_two_hop_label>().swap(L_temp_599[v_k]); // clear new labels for RAM efficiency

				return 1; // return to results; the return type must be the same with
						  // results
			}));
	}
	for (auto &&result : results)
		result.get(); // all threads finish here

	return output_L;
}

/* canonical_repair_distributed */
static void hop_constrained_clean_L_distributed (hop_constrained_case_info &case_info,
vector<vector<hop_constrained_two_hop_label> >& LL, vector<int>& nid_vec, int thread_num) {

	auto &L = LL;
	int N = L.size();
	label_size_before_canonical_repair_599 = 0;
	label_size_after_canonical_repair_599 = 0;

	// ThreadPool pool(thread_num);
	// std::vector<std::future<int>> results;

	int nid_size = nid_vec.size();
	for (auto &v: nid_vec) {
		// int x = nid_vec[v];
		// printf("%d nid_vec %d\n", v, v);
		results.emplace_back(pool.enqueue(
			[v, &L] { // pass const type value j to thread; [] can be empty
				mtx_599[max_N_ID_for_mtx_599 - 1].lock();
				int used_id = Qid_599.front();
				Qid_599.pop();
				mtx_599[max_N_ID_for_mtx_599 - 1].unlock();

				vector<hop_constrained_two_hop_label> Lv_final;

				mtx_599[v].lock_shared();
				vector<hop_constrained_two_hop_label> Lv = L[v];
				mtx_599[v].unlock_shared();
				label_size_before_canonical_repair_599 += Lv.size();

				auto &T = Temp_L_vk_599[used_id];

				for (auto Lvi : Lv)
				{
					int u = Lvi.hub_vertex;
					int u_hop = Lvi.hop;

					mtx_599[u].lock_shared();
					auto Lu = L[u];
					mtx_599[u].unlock_shared();

					int min_dis = std::numeric_limits<int>::max();
					for (auto &label1 : Lu)
					{
						for (auto &label2 : T[label1.hub_vertex])
						{
							if (label1.hop + label2.second <= u_hop)
							{
								long long int query_dis =
									label1.distance + (long long int)label2.first;
								if (query_dis < min_dis)
								{
									min_dis = query_dis;
								}
							}
						}
					}

					if (min_dis > Lvi.distance)
					{
						Lv_final.push_back(Lvi);
						T[u].push_back({Lvi.distance, Lvi.hop});
					}
				}

				for (auto label : Lv_final){
					vector<pair<int, int>>().swap(T[label.hub_vertex]);
				}

				mtx_599[v].lock();
				L[v] = Lv_final;
				L[v].shrink_to_fit();
				mtx_599[v].unlock();
				label_size_after_canonical_repair_599 += Lv_final.size();

				mtx_599[max_N_ID_for_mtx_599 - 1].lock();
				Qid_599.push(used_id);
				mtx_599[max_N_ID_for_mtx_599 - 1].unlock();

				return 1; // return to results; the return type must be the same with
						  // results
			}));
	}

	for (auto &&result : results)
		result.get(); // all threads finish here
	results.clear();

}

/*canonical_repair*/
static void hop_constrained_clean_L (hop_constrained_case_info &case_info, vector<vector<hop_constrained_two_hop_label> >& LL, int thread_num, int N) {
	
	auto &L = LL;
	// int N = L.size();
	label_size_before_canonical_repair_599 = 0;
	label_size_after_canonical_repair_599 = 0;

	// ThreadPool pool(thread_num);
	// std::vector<std::future<int>> results;
	
	for (int v = 0; v < N; v++)
	{
		results.emplace_back(pool.enqueue(
			[v, &L] { // pass const type value j to thread; [] can be empty
				mtx_599[max_N_ID_for_mtx_599 - 1].lock();
				int used_id = Qid_599.front();
				Qid_599.pop();
				mtx_599[max_N_ID_for_mtx_599 - 1].unlock();

				vector<hop_constrained_two_hop_label> Lv_final;

				mtx_599[v].lock_shared();
				vector<hop_constrained_two_hop_label> Lv = L[v];
				mtx_599[v].unlock_shared();
				label_size_before_canonical_repair_599 += Lv.size();

				auto &T = Temp_L_vk_599[used_id];

				for (auto Lvi : Lv)
				{
					int u = Lvi.hub_vertex;
					int u_hop = Lvi.hop;

					mtx_599[u].lock_shared();
					auto Lu = L[u];
					mtx_599[u].unlock_shared();

					int min_dis = std::numeric_limits<int>::max();
					for (auto &label1 : Lu)
					{
						for (auto &label2 : T[label1.hub_vertex])
						{
							if (label1.hop + label2.second <= u_hop)
							{
								long long int query_dis =
									label1.distance + (long long int)label2.first;
								if (query_dis < min_dis)
								{
									min_dis = query_dis;
								}
							}
						}
					}

					if (min_dis > Lvi.distance)
					{
						Lv_final.push_back(Lvi);
						T[u].push_back({Lvi.distance, Lvi.hop});
					}
				}

				for (auto label : Lv_final)
				{
					vector<pair<int, int>>().swap(T[label.hub_vertex]);
				}

				mtx_599[v].lock();
				L[v] = Lv_final;
				L[v].shrink_to_fit();
				mtx_599[v].unlock();
				label_size_after_canonical_repair_599 += Lv_final.size();

				mtx_599[max_N_ID_for_mtx_599 - 1].lock();
				Qid_599.push(used_id);
				mtx_599[max_N_ID_for_mtx_599 - 1].unlock();

				return 1; // return to results; the return type must be the same with
						  // results
			}));
	}

	for (auto &&result : results)
		result.get(); // all threads finish here
	results.clear();

	case_info.label_size_before_canonical_repair = label_size_before_canonical_repair_599;
	case_info.label_size_after_canonical_repair = label_size_after_canonical_repair_599;
	case_info.canonical_repair_remove_label_ratio = (double)(label_size_before_canonical_repair_599 - label_size_after_canonical_repair_599) / label_size_before_canonical_repair_599;
}

static void hop_constrained_two_hop_labels_generation (graph_v_of_v<int> &input_graph, hop_constrained_case_info &case_info) {

	// ----------------------------------- step 1: initialization -----------------------------------
	auto begin = std::chrono::high_resolution_clock::now();

	labal_size_599 = 0;
	begin_time_599 = std::chrono::high_resolution_clock::now();
	max_run_time_nanoseconds_599 = case_info.max_run_time_seconds * 1e9;
	max_labal_size_599 = case_info.max_bit_size / sizeof(hop_constrained_two_hop_label);

	int N = input_graph.size();
	L_temp_599.resize(N);
	if (N > max_N_ID_for_mtx_599) {
		cout << "N > max_N_ID_for_mtx_599!" << endl;
		exit(1);
	}

	int num_of_threads = case_info.thread_num;
	// ThreadPool pool(num_of_threads);
	// std::vector<std::future<int>> results;

	ideal_graph_599 = input_graph;
	// global_use_rank_prune = case_info.use_rank_prune;

	auto end = std::chrono::high_resolution_clock::now();
	case_info.time_initialization =
		std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;

	//----------------------------------------------- step 2: generate labels
	//---------------------------------------------------------------
	begin = std::chrono::high_resolution_clock::now();

	global_upper_k = case_info.upper_k == 0 ? std::numeric_limits<int>::max() : case_info.upper_k;

	Temp_L_vk_599.resize(num_of_threads);
	dist_hop_599.resize(num_of_threads);
	Q_handle_priorities_599.resize(num_of_threads);
	Vh_599.resize(num_of_threads);
	hop_constrained_node_handle handle_x;
	for (int i = 0; i < num_of_threads; i++)
	{
		Temp_L_vk_599[i].resize(N);
		dist_hop_599[i].resize(N, {std::numeric_limits<int>::max(), 0});
		Q_handle_priorities_599[i].resize(N);
		for (int j = 0; j < N; j++)
		{
			Q_handle_priorities_599[i][j].resize(global_upper_k + 1, {handle_x, std::numeric_limits<int>::max()});
		}
		Vh_599[i].resize(global_upper_k + 2);
		Qid_599.push(i);
	}
	if (case_info.use_2023WWW_generation)
	{
		for (int v_k = 0; v_k < N; v_k++)
		{
			results.emplace_back(pool.enqueue([v_k] {_2023WWW_thread_function(v_k); return 1;}));
		}
	}
	else
	{
		int last_check_vID = N - 1;
		// if (global_use_2M_prune) {
		//   for (int v_k = N - 1; v_k >= 0; v_k--) {
		//     if (is_mock[v_k]) {
		//       last_check_vID = v_k;
		//       break;
		//     }
		//   }
		// }
		for (int v_k = 0; v_k <= last_check_vID; v_k++)
		{
			// if (global_use_2M_prune && is_mock[v_k]) {
			//	continue;
			// }
			results.emplace_back(pool.enqueue([v_k] {HSDL_thread_function(v_k); return 1;}));
		}
	}
	for (auto &&result : results)
		result.get();

	end = std::chrono::high_resolution_clock::now();
	case_info.time_generate_labels = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s

	//----------------------------------------------- step 3:
	// sortL---------------------------------------------------------------
	begin = std::chrono::high_resolution_clock::now();

	case_info.L = hop_constrained_sortL(num_of_threads);

	end = std::chrono::high_resolution_clock::now();
	case_info.time_sortL = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s

	//----------------------------------------------- step 4:
	// canonical_repair---------------------------------------------------------------
	begin = std::chrono::high_resolution_clock::now();

	// case_info.print_L();

	if (case_info.use_canonical_repair)
	{
		// hop_constrained_clean_L(case_info, num_of_threads);
	}

	// case_info.print_L();

	end = std::chrono::high_resolution_clock::now();
	case_info.time_canonical_repair = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
	//---------------------------------------------------------------------------------------------------------------------------------------

	case_info.time_total = case_info.time_initialization + case_info.time_generate_labels + case_info.time_sortL + case_info.time_canonical_repair;
	case_info.compute_label_size_per_node(N);
	// case_info.print_L();

	hop_constrained_clear_global_values();
}

static void hop_constrained_two_hop_labels_generation_init (graph_v_of_v<int> &input_graph, hop_constrained_case_info &case_info) {
	
	V = input_graph.size();
	global_upper_k = case_info.upper_k == 0 ? std::numeric_limits<int>::max() : case_info.upper_k;

	ideal_graph_599 = input_graph;

	labal_size_599 = 0;
	max_run_time_nanoseconds_599 = case_info.max_run_time_seconds * 1e9;
	max_labal_size_599 = case_info.max_bit_size / sizeof(hop_constrained_two_hop_label);
	num_of_threads = case_info.thread_num;
	
	if (V > max_N_ID_for_mtx_599) {
		cout << "V > max_N_ID_for_mtx_599!" << endl;
		exit(1);
	}
	if (case_info.upper_k == 0) {
		cout << "case_info.upper_k == 0!" << endl;
		exit(1);
	}

	T[0].resize(V);
	T[1].resize(V);
	Last_iter_label_size.resize(V);
	L_temp_599.resize(V);
	temp_D.resize(num_of_threads);
	dist_hop_599.resize(num_of_threads);
	Temp_L_vk_599.resize(num_of_threads);
	Temp_L_vk_599_v2.resize(num_of_threads);
	Vh_599.resize(num_of_threads);
	Q_handle_priorities_599.resize(num_of_threads);
	hop_constrained_node_handle handle_x;
	for (int i = 0; i < num_of_threads; i++) {
		temp_D[i].resize(V, {std::numeric_limits<int>::max(), 0});
		Temp_L_vk_599[i].resize(V);
		Temp_L_vk_599_v2[i].resize(V);
		dist_hop_599[i].resize(V, {std::numeric_limits<int>::max(), 0});
		Q_handle_priorities_599[i].resize(V);
		for (int j = 0; j < V; j++) {
			Q_handle_priorities_599[i][j].resize(global_upper_k + 1, {handle_x, std::numeric_limits<int>::max()});
			Temp_L_vk_599_v2[i][j].resize(global_upper_k + 1);
			for (int k = 0; k <= global_upper_k; ++k) {
				Temp_L_vk_599_v2[i][j][k] = std::numeric_limits<int>::max();
			}
		}
		Qid_599.push(i);
		Vh_599[i].resize(global_upper_k + 2);
	}
}

static void hop_constrained_two_hop_labels_generation (graph_v_of_v<int> &input_graph, hop_constrained_case_info &case_info
, vector<vector<hop_constrained_two_hop_label> >&L, vector<int> &nid_vec) {

	// ----------------------------------- step 1: initialization -----------------------------------
	auto begin = std::chrono::high_resolution_clock::now();
	
	int N = nid_vec.size();
	// ThreadPool pool(num_of_threads);
	// std::vector<std::future<int>> results;
	
	auto end = std::chrono::high_resolution_clock::now();
	case_info.time_initialization += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
	
	// ----------------------------------------------- step 2: generate labels ---------------------------------------------------------------
	begin = std::chrono::high_resolution_clock::now();

	// Temp_L_vk_599.resize(num_of_threads);
	// dist_hop_599.resize(num_of_threads);
	// Q_handle_priorities_599.resize(num_of_threads);
	// Vh_599.resize(num_of_threads);
	// for (int i = 0; i < num_of_threads; i++)
	// {
		// Temp_L_vk_599[i].resize(V);
		// dist_hop_599[i].resize(V, {std::numeric_limits<int>::max(), 0});
		// Vh_599[i].resize(global_upper_k + 2);
	// }

	if (case_info.use_2023WWW_generation) {
		for (int v_k = 0; v_k < N; ++ v_k) {
			int x = nid_vec[v_k];
			results.emplace_back(pool.enqueue([x]{_2023WWW_thread_function(x); return 1;}));
		}
		for (auto &&result : results) {
			result.get();
		}
		results.clear();

	} else if (case_info.use_2023WWW_generation_optimized) {
		for (int v_k = 0; v_k < N; ++ v_k) {
			int x = nid_vec[v_k];
			results.emplace_back(pool.enqueue([x]{_2023WWW_thread_function_optimized(x); return 1;}));
		}
		for (auto &&result : results) {
			result.get();
		}
		results.clear();

	} else if (case_info.use_GPU_version_generation) {
		boost::random::uniform_int_distribution<> qid_range{ static_cast<int>(0), static_cast<int>(num_of_threads_cpu - 1) };
		int tag = 0;
		for (int v_k = 0; v_k < N; ++ v_k) {
			T[tag][v_k].push_back(std::make_tuple(v_k, 0, v_k));
			// label_map[v_k][make_tuple(v_k, 0, 0)] = 0;
			L_temp_599[v_k].push_back(hop_constrained_two_hop_label(v_k, v_k, 0, 0));
			Last_iter_label_size[v_k] = 1;
		}
		
		int hop_cst = case_info.upper_k;
		for (int i = 1; i <= case_info.upper_k && global_label_generation_tag; ++ i, tag ^= 1) {
			
			// for (int v_k = 0; v_k < N; ++ v_k) {
			// 	Last_iter_label_size[v_k] = L_temp_599[v_k].size();
			// }
			// for (int v_k = 0; v_k < N; ++ v_k) {
			// 	results.emplace_back(pool.enqueue([v_k]{get_Last_iter_label_size(v_k); return 1;}));
			// }
			// for (auto &&result : results) result.get();
			// results.clear();
			global_label_generation_tag = 0;
			for (int v_k = 0; v_k < N; ++ v_k) {
				results.emplace_back(pool.enqueue([v_k, tag, i, hop_cst]
				{HSDL_with_large_queue(v_k, tag, i, hop_cst); return 1;}));
			}
			for (auto &&result : results){
				result.get();
			}
			results.clear();
		}
		
	} else if (case_info.use_GPU_version_generation_optimized) {
		boost::random::uniform_int_distribution<> qid_range{ static_cast<int>(0), static_cast<int>(num_of_threads_cpu - 1) };
		int tag = 0;
		for (int v_k = 0; v_k < N; ++ v_k) {
			T[tag][v_k].push_back(std::make_tuple(v_k, 0, v_k));
			// label_map[v_k][make_tuple(v_k, 0, 0)] = 0;
			L_temp_599[v_k].push_back(hop_constrained_two_hop_label(v_k, v_k, 0, 0));
			Last_iter_label_size[v_k] = 1;
		}
		
		int hop_cst = case_info.upper_k;
		for (int i = 1; i <= case_info.upper_k && global_label_generation_tag; ++ i, tag ^= 1) {
			
			// for (int v_k = 0; v_k < N; ++ v_k) {
			// 	Last_iter_label_size[v_k] = L_temp_599[v_k].size();
			// }
			// for (int v_k = 0; v_k < N; ++ v_k) {
			// 	results.emplace_back(pool.enqueue([v_k]{get_Last_iter_label_size(v_k); return 1;}));
			// }
			// for (auto &&result : results) result.get();
			// results.clear();
			global_label_generation_tag = 0;
			for (int v_k = 0; v_k < N; ++ v_k) {
				results.emplace_back(pool.enqueue([v_k, tag, i, hop_cst]
				{HSDL_with_large_queue_optimized(v_k, tag, i, hop_cst); return 1;}));
			}
			for (auto &&result : results){
				result.get();
			}
			results.clear();
		}
	} else {
		for (int v_k = 0; v_k < N; ++ v_k) {
			int x = nid_vec[v_k];
			results.emplace_back(pool.enqueue([x]{HSDL_thread_function(x); return 1;}));
		}
		for (auto &&result : results) {
			result.get();
		}
		results.clear();
	}

	end = std::chrono::high_resolution_clock::now();
	case_info.time_generate_labels += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s

	// Tranverse
	begin = std::chrono::high_resolution_clock::now();

	// for (int v_k = 0; v_k < V; ++ v_k) {
	// 	results.emplace_back(pool.enqueue([v_k, &L]{
	// 		// for (int j = 0; j < L_temp_599[v_k].size(); ++ j) {
	// 		// 	hop_constrained_two_hop_label x = L_temp_599[v_k][j];
	// 		// 	L[v_k].push_back({x.hub_vertex, 0, x.hop, x.distance});
	// 		// }
	// 		L[v_k].insert(L[v_k].end(), L_temp_599[v_k].begin(), L_temp_599[v_k].end());
	// 		L_temp_599[v_k].clear();
	// 		return 1;
	// 	}));
	// }
	// for (auto &&result : results) {
	// 	result.get();
	// }
	// results.clear();

	hop_constrained_two_hop_label x;
	for (int v_k = 0; v_k < V; ++ v_k) {
		// for (int j = L_temp_599[v_k].size() - 1; j >= 0; -- j) {
		// 	x = L_temp_599[v_k][j];
		// 	L[v_k].push_back({x.hub_vertex, 0, x.hop, x.distance});
		// }
		L[v_k].insert(L[v_k].end(), L_temp_599[v_k].begin(), L_temp_599[v_k].end());
		L_temp_599[v_k].clear();
	}

	end = std::chrono::high_resolution_clock::now();
	case_info.time_traverse += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s

}

// return the uncleaned labels
static void hop_constrained_two_hop_labels_generation (graph_v_of_v<int> &input_graph, hop_constrained_case_info &case_info, 
	vector<vector<hop_constrained_two_hop_label>>& uncleaned_L) {
	//----------------------------------- step 1: initialization
	//-----------------------------------
	auto begin = std::chrono::high_resolution_clock::now();

	labal_size_599 = 0;
	begin_time_599 = std::chrono::high_resolution_clock::now();
	max_run_time_nanoseconds_599 = case_info.max_run_time_seconds * 1e9;
	max_labal_size_599 = case_info.max_bit_size / sizeof(hop_constrained_two_hop_label);

	int N = input_graph.size();
	L_temp_599.resize(N);
	if (N > max_N_ID_for_mtx_599) {
		cout << "N > max_N_ID_for_mtx_599!" << endl;
		exit(1);
	}

	int num_of_threads = case_info.thread_num;
	// ThreadPool pool(num_of_threads);
	// std::vector<std::future<int>> results;

	ideal_graph_599 = input_graph;
	// global_use_rank_prune = case_info.use_rank_prune;

	auto end = std::chrono::high_resolution_clock::now();
	case_info.time_initialization =
		std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;

	//----------------------------------------------- step 2: generate labels
	//---------------------------------------------------------------
	begin = std::chrono::high_resolution_clock::now();

	global_upper_k = case_info.upper_k == 0 ? std::numeric_limits<int>::max() : case_info.upper_k;

	// Temp_L_vk_599.resize(num_of_threads);
	// dist_hop_599.resize(num_of_threads);
	// Q_handle_priorities_599.resize(num_of_threads);
	// Vh_599.resize(num_of_threads);
	// hop_constrained_node_handle handle_x;
	// for (int i = 0; i < num_of_threads; i++)
	// {
	// 	Temp_L_vk_599[i].resize(N);
	// 	dist_hop_599[i].resize(N, {std::numeric_limits<int>::max(), 0});
	// 	Q_handle_priorities_599[i].resize(N);
	// 	for (int j = 0; j < N; j++)
	// 	{
	// 		Q_handle_priorities_599[i][j].resize(global_upper_k + 1, {handle_x, std::numeric_limits<int>::max()});
	// 	}
	// 	Vh_599[i].resize(global_upper_k + 2);
	// 	Qid_599.push(i);
	// }
	if (case_info.use_2023WWW_generation) {
		for (int v_k = 0; v_k < N; v_k++) {
			results.emplace_back(pool.enqueue([v_k] {_2023WWW_thread_function(v_k); return 1;}));
		}
	} else {
		int last_check_vID = N - 1;
		for (int v_k = 0; v_k <= last_check_vID; v_k++) {
			results.emplace_back(pool.enqueue([v_k] {HSDL_thread_function(v_k); return 1;}));
		}
	}
	for (auto &&result : results)
		result.get();
	results.clear();
	end = std::chrono::high_resolution_clock::now();
	case_info.time_generate_labels = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s

	//----------------------------------------------- step 3:
	// sortL---------------------------------------------------------------
	begin = std::chrono::high_resolution_clock::now();
	case_info.L = hop_constrained_sortL (num_of_threads);

	end = std::chrono::high_resolution_clock::now();
	case_info.time_sortL = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s

	hop_constrained_two_hop_label x;
	for (int v_k = 0; v_k < V; ++ v_k) {
		for (int j = case_info.L[v_k].size() - 1; j >= 0; -- j) {
			x = case_info.L[v_k][j];
			uncleaned_L[v_k].push_back({x.hub_vertex, 0, x.hop, x.distance});
		}
		// L[v_k].insert(L[v_k].end(), L_temp_599[v_k].begin(), L_temp_599[v_k].end());
		case_info.L[v_k].clear();
	}
	// uncleaned_L = case_info.L;
	
	//----------------------------------------------- step 4:
	// canonical_repair---------------------------------------------------------------
	begin = std::chrono::high_resolution_clock::now();
	// case_info.print_L();

	if (case_info.use_canonical_repair) {
		// hop_constrained_clean_L(case_info, num_of_threads);
	}

	// case_info.print_L();

	end = std::chrono::high_resolution_clock::now();
	case_info.time_canonical_repair = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
	//---------------------------------------------------------------------------------------------------------------------------------------

	case_info.time_total = case_info.time_initialization + case_info.time_generate_labels + case_info.time_sortL + case_info.time_canonical_repair;

	// hop_constrained_clear_global_values();
}

#endif // HOP_CONSTRAINED_TWO_HOP_LABELS_GENERATION_H