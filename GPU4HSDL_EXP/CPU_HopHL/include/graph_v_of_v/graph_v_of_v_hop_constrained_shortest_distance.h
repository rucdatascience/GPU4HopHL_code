#pragma once


/* this func get all distances from source with hop constraint */

template<typename T>
void graph_v_of_v_hop_constrained_shortest_distance(graph_v_of_v<T>& instance_graph, int source, int hop_cst, vector<T>& distance) {

	int N = instance_graph.size();

	vector<std::vector<pair<int, T>>> Q(hop_cst + 2);
	Q[0].push_back({ source, 0 });
	//vector<T>distance;
	//distance.resize(N); // distance.resize(N, std::numeric_limits<T>::max()) does not work here, since the type of the second parametter of resize should be specified
	distance.assign(N, std::numeric_limits<T>::max());
	distance[source] = 0;

	int h = 0;

	/* BFS */
	while (h <= hop_cst) {
		for (auto& xx : Q[h]) {
			int v = xx.first;
			T distance_v = xx.second;

			if (v == source || distance[v] > distance_v) {
				distance[v] = distance_v;
				for (auto& yy : instance_graph[v]) {
					if (distance_v + yy.second < distance[yy.first]) {
						Q[h + 1].emplace_back(yy.first, distance_v + yy.second);
					}
				}
			}
		}
		h++;
	}
}

template<typename T>
void graph_v_of_v_hop_constrained_shortest_distance_local(graph_v_of_v<T>& instance_graph, int source, int hop_cst) {

	int N = instance_graph.size();
    vector<T>distance(N);

	vector<std::vector<pair<int, T>>> Q(hop_cst + 2);
	Q[0].push_back({ source, 0 });
	//vector<T>distance;
	//distance.resize(N); // distance.resize(N, std::numeric_limits<T>::max()) does not work here, since the type of the second parametter of resize should be specified
	distance.assign(N, std::numeric_limits<T>::max());
	distance[source] = 0;

	int h = 0;

	/* BFS */
	while (h <= hop_cst) {
		for (auto& xx : Q[h]) {
			int v = xx.first;
			T distance_v = xx.second;

			if (v == source || distance[v] > distance_v) {
				distance[v] = distance_v;
				for (auto& yy : instance_graph[v]) {
					if (distance_v + yy.second < distance[yy.first]) {
						Q[h + 1].emplace_back(yy.first, distance_v + yy.second);
					}
				}
			}
		}
		h++;
	}
}






//动态规划，小图特别好用
template<typename T>
void graph_v_of_v_hop_constrained_shortest_distance_speed_up(
    graph_v_of_v<T>& instance_graph,
    int source,
    int target,
    int hop_cst,
    T& distance)
{
    if(source==target){
        distance = 0;
        return;
    }
    int N = instance_graph.size();
    vector<vector<T>> dp(hop_cst + 1, vector<T>(N, std::numeric_limits<T>::max()));

    dp[0][source] = 0;

    for (int k = 1; k <= hop_cst; ++k) {
        for (int u = 0; u < N; ++u) {
            if (dp[k - 1][u] != std::numeric_limits<T>::max()) {
                for (auto& edge : instance_graph[u]) {
                    int v = edge.first;
                    T weight = edge.second;
                    if (dp[k][v] > dp[k - 1][u] + weight) {
                        dp[k][v] = dp[k - 1][u] + weight;
                    }
                }
            }
        }
    }

    distance = std::numeric_limits<T>::max();
    for (int k = 1; k <= hop_cst; ++k) {
        if (dp[k][target] < distance) {
            distance = dp[k][target];
        }
    }
}





