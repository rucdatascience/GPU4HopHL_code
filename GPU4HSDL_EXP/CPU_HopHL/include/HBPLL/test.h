#pragma once

/*the following codes are for testing

---------------------------------------------------
a cpp file (try.cpp) for running the following test code:
----------------------------------------

#include <fstream>
#include <iostream>
using namespace std;

// header files in the Boost library: https://www.boost.org/
#include <boost/random.hpp>
boost::random::mt19937 boost_random_time_seed{
static_cast<std::uint32_t>(std::time(0)) };

#include <HBPLL/test.h>


int main()
{
		test_PLL();
}

------------------------------------------------------------------------------------------
Commends for running the above cpp file on Linux:

g++ -std=c++17 -I/home/boost_1_75_0 -I/root/rucgraph try.cpp -lpthread -Ofast -o
A
./A
rm A

(optional to put the above commends in run.sh, and then use the comment: sh
run.sh)


*/
// #include <HBPLL/PLL.h>
// #include <HBPLL/two_hop_labels.h>
#include <chrono>
#include <graph_v_of_v/graph_v_of_v.h>
#include <graph_v_of_v/graph_v_of_v_generate_random_graph.h>
#include <graph_v_of_v/graph_v_of_v_shortest_paths.h>
#include <graph_v_of_v/graph_v_of_v_update_vertexIDs_by_degrees_large_to_small.h>

#include <HBPLL/hop_constrained_two_hop_labels_generation.h>
#include <graph_v_of_v/graph_v_of_v_hop_constrained_shortest_distance.h>
#include <limits>
#include <text_mining/binary_save_read_vector.h>
#include <text_mining/print_items.h>
#include <future>
#include <thread>

string reach_limit_error_string_MB = "reach limit error MB";
string reach_limit_error_string_time = "reach limit error time";

boost::random::mt19937 boost_random_time_seed{
    static_cast<std::uint32_t>(std::time(0))};





void hop_constrained_check_correctness(hop_constrained_case_info &case_info, graph_v_of_v<int> &instance_graph, int iteration_source_times, int iteration_terminal_times, int upper_k,Res& result)
{

	boost::random::uniform_int_distribution<> vertex_range{
		static_cast<int>(0), static_cast<int>(instance_graph.size() - 1)};
        boost::random::uniform_int_distribution<> hop_range{
            static_cast<int>(1), static_cast<int>(upper_k)};

    
	double time = 0.0;
	for (int yy = 0; yy < iteration_source_times; yy++)
	{
		int source = vertex_range(boost_random_time_seed);

		// while (!is_mock[source]) {
		//   source = vertex_range(boost_random_time_seed);
		// }

		std::vector<int> distances(instance_graph.size());

		int hop_cst = hop_range(boost_random_time_seed);

		graph_v_of_v_hop_constrained_shortest_distance(instance_graph, source, hop_cst, distances);

		for (int xx = 0; xx < iteration_terminal_times; xx++)
		{
			int terminal = vertex_range(boost_random_time_seed);

			// while (is_mock[terminal]) {
			//   terminal = vertex_range(boost_random_time_seed);
			// }
			std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();// 
                        int query_dis = hop_constrained_extract_distance(
                            case_info.L, source, terminal, hop_cst);
			std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
			time += std::chrono::duration_cast<std::chrono::seconds>(end - begin).count(); //s
                        
			if (abs(query_dis - distances[terminal]) > 1e-4)
			{
				instance_graph.print();
				case_info.print_L();
				cout << "source = " << source << endl;
				cout << "terminal = " << terminal << endl;
				cout << "hop_cst = " << hop_cst << endl;
				cout << "query_dis = " << query_dis << endl;
				cout << "distances[terminal] = " << distances[terminal] << endl;
				cout << "abs(dis - distances[terminal]) > 1e-5!" << endl;
				getchar();
			}

			vector<pair<int, int>> path = hop_constrained_extract_shortest_path(case_info.L, source, terminal, hop_cst);
			int path_dis = 0;
			if (path.size() == 0 && source != terminal)
			{
				path_dis = std::numeric_limits<int>::max();
			}
			for (auto xx : path)
			{
				path_dis += instance_graph.edge_weight(xx.first, xx.second);
			}
			if (abs(query_dis - path_dis) > 1e-4)
			{
				instance_graph.print();
				case_info.print_L();
				cout << "source = " << source << endl;
				cout << "terminal = " << terminal << endl;
				cout << "hop_cst = " << hop_cst << endl;
				std::cout << "print_vector_pair_int:" << std::endl;
				for (int i = 0; i < path.size(); i++)
				{
					std::cout << "item: |" << path[i].first << "," << path[i].second << "|" << std::endl;
				}
				cout << "query_dis = " << query_dis << endl;
				cout << "path_dis = " << path_dis << endl;
				cout << "abs(dis - path_dis) > 1e-5!" << endl;
				getchar();
			}
		}
        }
        result.query_time = time / iteration_source_times / iteration_terminal_times;
}



// Function to perform the Dijkstra with hop constraint
void dijkstra_hopconstrained(const std::string dataset, const std::string query_path, int upper_k, Res& result) {

    ThreadPool pool(144);
    graph_v_of_v<int> instance_graph;
    std::cout << "read start" << std::endl;

    instance_graph.txt_read(dataset);
    int V = instance_graph.size();
    std::cout << "read success" << std::endl;

    std::ifstream in(query_path);
    if (!in) {
        std::cout << "Cannot open input file.\n";
        return;
    }

    std::string line;
    int source, terminal;
    double time = 0.0;

    std::getline(in, line); // Skip the first line
    // std::vector<int> distances(instance_graph.size());

    // Process each query line asynchronously
    std::vector<std::future<void>> results;
    
    int query_index = 0; // To track the current query index
    auto begin = std::chrono::steady_clock::now();
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        if (!(iss >> source >> terminal)) {
            break;
        }
        if(V>1000000)
        {
            results.emplace_back(pool.enqueue([&instance_graph, source, upper_k] {
                std::vector<int> distances(instance_graph.size());
                graph_v_of_v_hop_constrained_shortest_distance<int>(instance_graph, source, upper_k, distances);
            }));
        }
        else{
            results.emplace_back(pool.enqueue([&instance_graph, source,terminal, upper_k] {
                int distances;
                graph_v_of_v_hop_constrained_shortest_distance_speed_up<int>(instance_graph, source, terminal,upper_k, distances);
            }));
        }

        query_index++; // Increment query index
    }

    // Collect results
    for (auto &&result : results)
		  result.get();
    auto end = std::chrono::steady_clock::now();

    result.query_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count(); // Convert to milliseconds
    result.index_time = 0;
    result.size = 0;
}

void dijkstra_hopconstrained_speed_up(const std::string dataset, const std::string query_path, int upper_k, Res& result) {

    ThreadPool pool(144);
    graph_v_of_v<int> instance_graph;
    std::cout << "read start" << std::endl;

    instance_graph.txt_read(dataset);
    std::cout << "read success" << std::endl;

    std::ifstream in(query_path);
    if (!in) {
        std::cout << "Cannot open input file.\n";
        return;
    }

    std::string line;
    int source, terminal;
    double time = 0.0;

    std::getline(in, line); // Skip the first line
    // std::vector<int> distances(instance_graph.size());

    // Process each query line asynchronously
    std::vector<std::future<void>> results;
    
    int query_index = 0; // To track the current query index
    auto begin = std::chrono::steady_clock::now();
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        if (!(iss >> source >> terminal)) {
            break;
        }
        results.emplace_back(pool.enqueue([&instance_graph, source,terminal, upper_k] {
                int distances;
                graph_v_of_v_hop_constrained_shortest_distance_speed_up<int>(instance_graph, source, terminal,upper_k, distances);
            }));

        query_index++; // Increment query index
    }

    // Collect results
    for (auto &&result : results)
		  result.get();
    auto end = std::chrono::steady_clock::now();

    result.query_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count(); // Convert to milliseconds
    result.index_time = 0;
    result.size = 0;
}


void dijkstra_hopconstrained_correctness(const std::string dataset, const std::string query_path, int upper_k, Res& result) {

    ThreadPool pool(144);
    graph_v_of_v<int> instance_graph;
    std::cout << "read start" << std::endl;

    instance_graph.txt_read(dataset);
    std::cout << "read success" << std::endl;

    std::ifstream in(query_path);
    if (!in) {
        std::cout << "Cannot open input file.\n";
        return;
    }

    std::string line;
    int source, terminal;
    double time = 0.0;

    std::getline(in, line); // Skip the first line
    // std::vector<int> distances(instance_graph.size());

    // Process each query line asynchronously
    std::vector<std::future<void>> results;
    
    int query_index = 0; // To track the current query index
    //auto begin = std::chrono::steady_clock::now();
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        if (!(iss >> source >> terminal)) {
            break;
        }
        std::vector<int> distances(instance_graph.size());
        int dis = -1;
        graph_v_of_v_hop_constrained_shortest_distance_speed_up<int>(instance_graph, source, terminal,upper_k, dis);
        graph_v_of_v_hop_constrained_shortest_distance<int>(instance_graph, source, upper_k, distances);
        if(dis!=distances[terminal])
        {
            printf("dis(speed up)= %d, dis(origin)=%d\n",dis,distances[terminal]);
            return;
        }
        if(query_index%1000 == 0)
            {
            printf("%d\n",query_index);
            }
        
        query_index++; // Increment query index
    }

    // Collect results

    
    
    //auto end = std::chrono::steady_clock::now();

    //result.query_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count(); // Convert to milliseconds
    //result.index_time = 0;
    //result.size = 0;
}

void test_HSDL(std::string dataset, std::string query_path,int upper_k, int algo,Res &result,int is_clean) {

  /* problem parameters */
  int iteration_graph_times = 1, iteration_source_times = 10,
      iteration_terminal_times = 10;
  int V = 10, E = 10;
  int ec_min = 1, ec_max = 10;

  bool generate_new_random_graph = 0;
  bool load_new_graph = 1;

  /* hop bounded info */
  hop_constrained_case_info mm;
  mm.upper_k = upper_k;
  mm.use_rank_prune = 1; //HSDL
  mm.use_2023WWW_generation = algo; //0: HSDL, 1: 2023www
  mm.use_canonical_repair = is_clean; //clean
  mm.max_run_time_seconds = 3600*10;
  mm.thread_num = 144;

  /* result info */
  double avg_index_time = 0;
  double avg_time_initialization = 0, avg_time_generate_labels = 0,
         avg_time_sortL = 0, avg_time_canonical_repair = 0;
  double avg_canonical_repair_remove_label_ratio = 0, avg_index_size_per_v = 0;
  long long total_size = 0;

  /* iteration */
  for (int i = 0; i < iteration_graph_times; i++) {
    //cout << ">>>iteration_graph_times: " << i << endl;

    graph_v_of_v<int> instance_graph;
	  cout<<"read start"<<endl;

    instance_graph.txt_read(dataset);
    instance_graph = graph_v_of_v_update_vertexIDs_by_degrees_large_to_small<int>(instance_graph);
    int E = 0;
    for (int i = 0; i < instance_graph.size(); ++i) E += instance_graph[i].size();
    printf("E: %d\n\n",E);
    cout<<"read success"<<endl;

    // instance_graph.print();

    auto begin = std::chrono::high_resolution_clock::now();
    try {
      hop_constrained_two_hop_labels_generation(instance_graph, mm,query_path,result);
    } catch (string s) {
      cout << s << endl;
      hop_constrained_clear_global_values();
      continue;
    }
    auto end = std::chrono::high_resolution_clock::now();
    double runningtime =
        std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin)
            .count() /
        1e9; // s
    avg_index_time = avg_index_time + runningtime / iteration_graph_times;
    avg_time_initialization += mm.time_initialization / iteration_graph_times;
    avg_time_generate_labels += mm.time_generate_labels / iteration_graph_times;
    avg_time_sortL += mm.time_sortL / iteration_graph_times;
    avg_time_canonical_repair +=
        mm.time_canonical_repair / iteration_graph_times;
    avg_canonical_repair_remove_label_ratio +=
        mm.canonical_repair_remove_label_ratio / iteration_graph_times;

    // long long int index_size = 0;
    // for (auto it = mm.L.begin(); it != mm.L.end(); it++) {
    //   index_size = index_size + (*it).size();
    // }
    // total_size += index_size;
    // avg_index_size_per_v =
    //     avg_index_size_per_v + (double)index_size / V / iteration_graph_times;

    // hop_constrained_check_correctness(mm, instance_graph,
    //                                   iteration_source_times,
    //                                   iteration_terminal_times,
    //                                   mm.upper_k,result);
    // mm.print_L();
    //query_vertex_pair(query_path, mm, instance_graph, upper_k,result);
    mm.clear_labels();
  }
  result.index_time = avg_time_generate_labels;
  result.clean_time = avg_time_canonical_repair;
  //result.size = total_size * 4 / 1024 / 1024;
  


  cout << "avg_canonical_repair_remove_label_ratio: "
       << avg_canonical_repair_remove_label_ratio << endl;
  cout << "avg_index_time: " << avg_index_time << "s" << endl;
  cout << "\t avg_time_initialization: " << avg_time_initialization << endl;
  cout << "\t avg_time_generate_labels: " << avg_time_generate_labels << endl;
  cout << "\t avg_time_sortL: " << avg_time_sortL << endl;
  cout << "\t avg_time_canonical_repair: " << avg_time_canonical_repair << endl;
  cout << "\t avg_index_size_per_v: " << avg_index_size_per_v << endl;
}

