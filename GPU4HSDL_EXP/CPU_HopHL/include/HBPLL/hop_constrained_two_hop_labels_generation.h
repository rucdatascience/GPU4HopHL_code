#pragma once
#include <HBPLL/hop_constrained_two_hop_labels.h>
// #include <HBPLL/two_hop_labels.h>
#include <boost/heap/fibonacci_heap.hpp>
#include <graph_v_of_v/graph_v_of_v.h>
#include <shared_mutex>
#include <tool_functions/ThreadPool.h>
#include <map>
struct Res {
  double index_time =0;
  long long before_clean_size=0; // MB
  long long size=0;// MB
  double query_time=0; // average query time
  double before_clean_query_time = 0;
  double clean_time=0;
};

/*unique code for this file: 599*/
long long int max_labal_size_599;
long long int labal_size_599;
int max_N_ID_for_mtx_599 = 1e7;
long long max_run_time_milliseconds_599;
int global_upper_k;
long long int label_size_before_canonical_repair_599, label_size_after_canonical_repair_599;

auto begin_time_599 = std::chrono::high_resolution_clock::now();

vector<std::shared_timed_mutex> mtx_599(max_N_ID_for_mtx_599);

graph_v_of_v<int> ideal_graph_599;

vector<vector<hop_constrained_two_hop_label>> L_temp_599;
vector<vector<vector<pair<int, int>>>> Temp_L_vk_599;
vector<vector<pair<int, int>>> dist_hop_599;
vector<vector<vector<int>>> Vh_599;

queue<int> Qid_599;

typedef typename boost::heap::fibonacci_heap<hop_constrained_two_hop_label>::handle_type hop_constrained_node_handle;

vector<vector<vector<pair<hop_constrained_node_handle, int>>>> Q_handle_priorities_599;

#include <atomic>
#include <iostream>
#include <mutex>

// Declare an atomic integer to track progress
std::atomic<int> progress_counter(0); 
std::mutex progress_mutex; // Mutex for synchronized progress output

std::atomic<bool> stop_flag(false);
std::atomic<int> last_printed_progress(0); // Track last printed progress percentage

void update_progress(int N) {
    int current_progress = progress_counter.load();
    int progress_percentage = (current_progress * 100) / N;
    
    // Check if current progress is a multiple of 10% of N and has not been printed yet
    if (progress_percentage >= last_printed_progress + 10) {
        std::lock_guard<std::mutex> lock(progress_mutex); // Ensure thread-safe output
        std::cout << "Progress: " << progress_percentage << "% (" 
                  << current_progress << "/" << N << " tasks completed).\n";
        last_printed_progress = progress_percentage;
    }
}

void watchdog(int N, std::chrono::milliseconds max_run_time) {
    auto start_time = std::chrono::high_resolution_clock::now();

    while (!stop_flag && progress_counter < N) {
        auto elapsed = std::chrono::high_resolution_clock::now() - start_time;
        if (std::chrono::duration_cast<std::chrono::milliseconds>(elapsed) > max_run_time) {
            std::cout << "Timeout reached. Stopping tasks...\n";
            stop_flag = true;
            break;
        }
        update_progress(N);
        std::this_thread::sleep_for(std::chrono::milliseconds(1000*60)); // Check every 100 ms
    }
}



void hop_constrained_clear_global_values()
{
	vector<vector<hop_constrained_two_hop_label>>().swap(L_temp_599);
	ideal_graph_599.clear();
	vector<vector<vector<pair<int, int>>>>().swap(Temp_L_vk_599);
	vector<vector<pair<int, int>>>().swap(dist_hop_599);
	vector<vector<vector<pair<hop_constrained_node_handle, int>>>>().swap(Q_handle_priorities_599);
	vector<vector<vector<int>>>().swap(Vh_599);
	queue<int>().swap(Qid_599);
}

void HSDL_thread_function(int v_k)
{

	if (labal_size_599 > max_labal_size_599)
	{
		// throw reach_limit_error_string_MB;
	}
	if (stop_flag)
	{
		//throw "HSDL reach_limit_error_string_time";
		//printf("HSDL over limit time\n\n");
		return;
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
	if (stop_flag)
	{
		//throw "HSDL reach_limit_error_string_time";
		//printf("HSDL over limit time\n\n");
		return;
	}

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

		if (P_u < query_v_k_u ||
			query_v_k_u == 0)
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
		Q_handle_priorities[xx.first][xx.second] = {
			handle_x, std::numeric_limits<int>::max()};
	}

	mtx_599[v_k].lock();
	vector<hop_constrained_two_hop_label>(L_temp_599[v_k]).swap(L_temp_599[v_k]);
	mtx_599[v_k].unlock();

	mtx_599[max_N_ID_for_mtx_599 - 1].lock();
	Qid_599.push(used_id);
	labal_size_599 = labal_size_599 + new_label_num;
	mtx_599[max_N_ID_for_mtx_599 - 1].unlock();
}

void _2023WWW_thread_function(int v_k)
{

	if (labal_size_599 > max_labal_size_599)
	{
		// throw reach_limit_error_string_MB;
	}
	if (stop_flag)
	{
		//throw reach_limit_error_string_time;
		//printf("2023 WWW 超过最长时间\n\n");
		return;
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
	mtx_599[v_k].unlock();

	Vh[0].push_back(v_k);

	dist_hop[v_k] = {0, v_k};
	dist_hop_changes.push_back(v_k);

	vector<tuple<int, int, int>> dh_updates;
	map<int, int> mp;
	for (int h = 0; h <= global_upper_k; h++)
	{
		if (stop_flag)
	{
		//throw reach_limit_error_string_time;
		//printf("2023 WWW 超过最长时间\n\n");
		return;
	}

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
			if(v_k > u )
			{
				continue;
			}
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
						if(P_u >= query_v_k_u)break;
					}
				}
			}
			mtx_599[u].unlock();

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
						if(mp.find(adj_v)==mp.end())
						{
							Vh[h + 1].push_back(adj_v);
							mp[adj_v]=1;
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

void query_vertex_pair(std::string query_path, vector<vector<hop_constrained_two_hop_label> >&LL, graph_v_of_v<int> &instance_graph, int upper_k, Res& result, int before_clean) {
    
    const int ITERATIONS = 10;  // 进行100次完整的查询操作

    long long total_time = 0;  // 累计所有查询的时间

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        std::ifstream in(query_path);  // 每次循环重新打开文件
        if (!in) {
            std::cerr << "Cannot open input file: " << query_path << "\n";
            return;
        }

        std::string header;
        std::getline(in, header); // 跳过标题行

        int source = 0, terminal = 0;
        long long time = 0;
        int lines = 0;
        long long match_count = 0;//实际计算次数
        volatile long long dis=0;
        // 执行一次完整的文件查询操作
        //auto begin = std::chrono::steady_clock::now();
        while (in >> source >> terminal) {
            lines++;
            
            auto begin = std::chrono::steady_clock::now();
            // 每对 source 和 terminal 执行一次查询
            dis+= hop_constrained_extract_distance(LL, source, terminal, upper_k);
            auto end = std::chrono::steady_clock::now();
            time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
            
            if(lines%10000==0)
            {
                //printf("size1: %d,size2: %d, total time now: %lld,match count: %lld\n\n",LL[source].size(),LL[terminal].size(),time,match_count);
            }
        }
        // auto end = std::chrono::steady_clock::now();
        // time += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();

        // 验证查询行数是否符合预期
        if (lines != 100000) {
            //std::cerr << "Query error: Expected 100000 lines, but got " << lines << "\n";
        }

        total_time += time;  // 将每次的查询时间累加
        
    }

    // 计算平均查询时间
    if (before_clean == 1) {
       
        result.before_clean_query_time = total_time / ITERATIONS / 1e6;  // 总时间除以ITERATIONS，转换为ms
    } else {
        printf("total time %ld\n",total_time);
        result.query_time = total_time / ITERATIONS / 1e6;  // 总时间除以ITERATIONS，转换为ms
    }
}



/*sortL*/
bool compare_hop_constrained_two_hop_label(hop_constrained_two_hop_label &i, hop_constrained_two_hop_label &j)
{
	if (i.hub_vertex != j.hub_vertex)
	{
		return i.hub_vertex < j.hub_vertex;
	}
	else if (i.hop != j.hop)
	{
		return i.hop < j.hop;
	}
	else
	{
		return i.distance < j.distance;
	}
}

vector<vector<hop_constrained_two_hop_label>> hop_constrained_sortL(int num_of_threads)
{

	/*time complexity: O(V*L*logL), where L is average number of labels per
	 * vertex*/

	int N = L_temp_599.size();
	vector<vector<hop_constrained_two_hop_label>> output_L(N);

	/*time complexity: O(V*L*logL), where L is average number of labels per
	 * vertex*/
	ThreadPool pool(num_of_threads);
	std::vector<std::future<int>> results; // return typename: xxx
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

/*canonical_repair*/
void hop_constrained_clean_L(hop_constrained_case_info &case_info, int thread_num)
{

	auto &L = case_info.L;
	int N = L.size();
	label_size_before_canonical_repair_599 = 0;
	label_size_after_canonical_repair_599 = 0;

	ThreadPool pool(thread_num);
	std::vector<std::future<int>> results;

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

void hop_constrained_two_hop_labels_generation(
	graph_v_of_v<int> &input_graph, hop_constrained_case_info &case_info,std::string query_path,Res& result)
{

	//----------------------------------- step 1: initialization
	//-----------------------------------
	auto begin = std::chrono::high_resolution_clock::now();

	labal_size_599 = 0;
	begin_time_599 = std::chrono::high_resolution_clock::now();
	max_run_time_milliseconds_599 = case_info.max_run_time_seconds * 1e3;
	max_labal_size_599 = case_info.max_bit_size / sizeof(hop_constrained_two_hop_label);

	int N = input_graph.size();
	L_temp_599.resize(N);
	if (N > max_N_ID_for_mtx_599)
	{
		cout << "N > max_N_ID_for_mtx_599!" << endl;
		exit(1);
	}

	int num_of_threads = case_info.thread_num;
	ThreadPool pool(num_of_threads);
	std::vector<std::future<int>> results;

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
	std::thread watchdog_thread(watchdog, N, std::chrono::milliseconds(static_cast<long long>(case_info.max_run_time_seconds * 1000)));
	if (case_info.use_2023WWW_generation) {
        for (int v_k = 0; v_k < N; v_k++) {
            results.emplace_back(pool.enqueue([v_k, N] {
				if (stop_flag) return 0;
                _2023WWW_thread_function(v_k);
                ++progress_counter; // Increment progress
                //update_progress(N); // Check and print progress update
                return 1;
            }));
        }
    } else {
        int last_check_vID = N - 1;
        for (int v_k = 0; v_k <= last_check_vID; v_k++) {
            results.emplace_back(pool.enqueue([v_k, N] {
				if (stop_flag) return 0;
                HSDL_thread_function(v_k);
                ++progress_counter; // Increment progress
                //update_progress(N); // Check and print progress update
                return 1;
            }));
        }
    }
	for (auto &&result : results)
		result.get();
	
	end = std::chrono::high_resolution_clock::now();
	case_info.time_generate_labels = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
	 // Final cleanup
    if (stop_flag) {
        std::cout << "Program terminated due to timeout.\n";
		case_info.time_generate_labels = 0;
		return;
    } 
	stop_flag = true;
	watchdog_thread.join();

	//----------------------------------------------- step 3:
	// sortL---------------------------------------------------------------
	begin = std::chrono::high_resolution_clock::now();

	case_info.L = hop_constrained_sortL(num_of_threads);

	end = std::chrono::high_resolution_clock::now();
	case_info.time_sortL =
		std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
			.count() /
		1e9; // s

	//----------------------------------------------- step 4:
	// canonical_repair---------------------------------------------------------------
	

	// case_info.print_L();
	
	
	long long int index_size = 0;
    
	if (case_info.use_canonical_repair)
	{
		for (auto it = case_info.L.begin(); it != case_info.L.end(); it++) {
      		index_size = index_size + (*it).size();
    	}	
    	result.before_clean_size = index_size;
		query_vertex_pair(query_path, case_info.L, input_graph, case_info.upper_k,result,1);
		begin = std::chrono::high_resolution_clock::now();
		hop_constrained_clean_L(case_info, num_of_threads);
		end = std::chrono::high_resolution_clock::now();
		case_info.time_canonical_repair = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9; // s
	}
	//end = std::chrono::high_resolution_clock::now();
	else{
		case_info.time_canonical_repair = 0; // s
	}

	index_size = 0;
	for (auto it = case_info.L.begin(); it != case_info.L.end(); it++) {
      		index_size = index_size + (*it).size();
    }
	result.size = index_size;
	query_vertex_pair(query_path, case_info.L, input_graph, case_info.upper_k,result,0);
	


	// case_info.print_L();

	
	//---------------------------------------------------------------------------------------------------------------------------------------

	case_info.time_total = case_info.time_initialization + case_info.time_generate_labels + case_info.time_sortL + case_info.time_canonical_repair;

	hop_constrained_clear_global_values();
}