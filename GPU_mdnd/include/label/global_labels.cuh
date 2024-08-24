#ifndef GLOBAL_LABELS_CUH
#define GLOBAL_LABELS_CUH

#include "definition/hub_def.h"
#include "label/hop_constrained_two_hop_labels.cuh"
#include "memoryManagement/cuda_hashTable.cuh"
#include "memoryManagement/cuda_vector.cuh"
#include "memoryManagement/mmpool.cuh"
#include "unordered_map"
#include <cstddef>
#include <cuda_runtime.h>
#include <graph/graph_v_of_v.h>
#include <memoryManagement/cuda_hashtable.cuh>
#include <unordered_set>
#include <vector>

#include <queue> // Add this line to include the <queue> header file
using namespace std;

class hop_constrained_case_info {
public:
    /*labels*/
    mmpool<hub_type> *mmpool_labels;
    cuda_vector<hub_type> *L_cuda;           // gpu res
    int *group;                              //存放对应group的顶点
    cuda_hashTable<int, int> *reflect_group; //存放group中的顶点在group中的位置
    vector<unordered_set<hub_type>> *final_label;
    // vector<vector<hub_type>> L_cpu; // cpu res
    size_t L_size;

    __host__ void init();

    __host__ void destroy_L_cuda();

    inline size_t cuda_vector_size() { return L_size; };

    ~hop_constrained_case_info() {
        mmpool_labels->~mmpool();
        cudaFree(mmpool_labels);
    };
    void init_group(std::vector<int> &group, graph_v_of_v<disType> &G, int hop_cst) {
        //将group中的顶点，及顶点的邻点存到group中

        unordered_set<int> temp;
        queue<pair<int, int>> Q;

        for (auto it : group) {
            Q.emplace(it, 0);
            while (!Q.empty()) {
                int vertex = Q.front().first;
                int hop = Q.front().second;
                Q.pop();
                if (temp.find(vertex) != temp.end()) {
                    continue;
                }
                temp.insert(vertex);
                if (hop < hop_cst) {
                    for (auto it2 : G.ADJs[vertex]) {
                        Q.emplace(it2.first, hop + 1);
                    }
                }
            }
        }

        L_size = temp.size();

        cudaMallocManaged(&this->group, L_size * sizeof(int));
        int j = 0;
        for (auto it : temp) {
            this->group[j++] = it;
        }
        printf("group size:%d\n",temp.size());

        cudaMallocManaged(&reflect_group, sizeof(cuda_hashTable<int, int>));
        new (reflect_group) cuda_hashTable<int, int>(L_size);
        //将group中的顶点在group中的位置存到reflect_group中
        for (int i = 0; i < this->L_size; i++) {
            reflect_group->insert(this->group[i], i);
        }
    }

    void merge_instance(hop_constrained_case_info &instance2) {
        for (int i = 0; i < instance2.L_size; i++) {
            for (int j = 0; j < instance2.L_cuda[i].size(); j++) {
                auto &temp = final_label->at(instance2.group[i]);
                temp.insert(*(instance2.L_cuda[i].get(j)));
            }
        }
    }

    disType query_distance(int start, int end, int hop_cst) {
        unordered_set<hub_type> &start_set = final_label->at(start);
        unordered_set<hub_type> &end_set = final_label->at(end);
        disType min_dis = (int)(1e9);
        for (auto &it1 : start_set) {
            for (auto &it2 : end_set) {
            if (it1.hub_vertex == it2.hub_vertex && it1.hop + it2.hop <= hop_cst) {
                disType dis = it1.distance + it2.distance;
                if (dis < min_dis) {
                min_dis = dis;
                }
            }
            }
        }
        return min_dis;
    }

    void print_L() {
        for (int i = 0; i < L_size; i++) {
            printf("Vertex %d: ", group[i]);
            for (int j = 0; j < L_cuda[i].size(); j++) {
                printf("%d ", L_cuda[i].get(j)->hub_vertex);
            }
            printf("\n");
        }
        printf("\n");
    }

    void print_final_label(int i) {
        printf("Vertex %d: ", i);
        unordered_set<hub_type> &temp = final_label->at(i);
        for (const auto &it : temp) {
            printf("{%d,%d,%d,%d}, ", it.hub_vertex, it.parent_vertex, it.hop, it.distance);
        }
        printf("\n");
    }
    
};

#endif