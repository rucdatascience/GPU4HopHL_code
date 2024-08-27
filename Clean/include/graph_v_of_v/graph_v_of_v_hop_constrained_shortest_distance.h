#pragma once


/* this func get all distances from source with hop constraint */

template<typename T>
void graph_v_of_v_hop_constrained_shortest_distance(graph_v_of_v<T>& instance_graph, int source, int hop_cst, vector<T>& distance) {

	int N = instance_graph.size();

	vector<vector<pair<int, T>>> Q(hop_cst + 2);
	Q[0].push_back({ source, 0 });

	distance.resize(N); // distance.resize(N, std::numeric_limits<T>::max()) does not work here, since the type of the second parametter of resize should be specified
	for (int i = 0; i < N; i++) {
		distance[i] = std::numeric_limits<T>::max();
	}
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
						Q[h + 1].push_back({ yy.first, distance_v + yy.second });
					}
				}
			}
		}
		h++;
	}
}