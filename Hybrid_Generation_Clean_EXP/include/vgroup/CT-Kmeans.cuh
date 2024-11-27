#ifndef CT_KMEANS_H
#define CT_KMEANS_H

#include "graph/graph_v_of_v.h"
#include "utilities/dijkstra.cuh"
#include "vgroup/CT/CT.hpp"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <omp.h>
#include <atomic>
#include <mutex>
using namespace std;

__global__ void assign_clusters(int N, int num_centers, const int* centers, const disType* distances, int* labels, int* group_size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int nearest_center = -1;
        disType min_distance = 1e10;

        for (int j = 0; j < num_centers; ++j) {
            int center = centers[j];
            disType distance = distances[j * N + i]; // Assuming distances are stored in row-major order

            if (distance < min_distance) {
                nearest_center = center;
                min_distance = distance;
            }
        }

        if (nearest_center != -1 && atomicAdd(&group_size[nearest_center], 1) <= MAX_GROUP_SIZE) {
            int old_label = atomicExch(&labels[i], nearest_center);
            if (old_label != -1) {
                atomicSub(&group_size[old_label], 1);
            }
        }
    }
}

void perform_clustering(int N, const std::vector<int>& centers, disType* d_distances, std::vector<int>& labels) {
    int* d_centers;
    int* d_labels;
    int* d_group_size;

    cudaMalloc(&d_centers, centers.size() * sizeof(int));
    cudaMalloc(&d_labels, N * sizeof(int));
    cudaMalloc(&d_group_size, N * sizeof(int));

    cudaMemcpy(d_centers, centers.data(), centers.size() * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_labels, labels.data(), N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_group_size, 0, N * sizeof(int));

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    assign_clusters<<<blocksPerGrid, threadsPerBlock>>>(N, centers.size(), d_centers, d_distances, d_labels, d_group_size);

    // Copy results back to host
    cudaMemcpy(labels.data(), d_labels, N * sizeof(int), cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_centers);
    cudaFree(d_labels);
    cudaFree(d_group_size);
}


static void generate_Group_CT_cores(graph_v_of_v<disType> &instance_graph, int hop_cst, std::vector<std::vector<int>> &groups) {

  int N = instance_graph.size();
  vector<int> labels(N, -1);
  std::unordered_map<int, std::atomic<int>> group_size;  // 使用std::atomic
  vector<int> centers;
  // generate CT cores
  CT_case_info mm;
  dijkstra_table dt(instance_graph, false, hop_cst);
  dt.is_gpu = true;
  mm.d = 10;
  mm.use_P2H_pruning = 1;
  mm.two_hop_info.use_2M_prune = 1;
  mm.two_hop_info.use_canonical_repair = 1;
  mm.thread_num = 64;
  auto start = std::chrono::high_resolution_clock::now();
  CT_cores(instance_graph, mm);
  auto end = std::chrono::high_resolution_clock::now();
  printf("CT_cores finished\n");
  std::chrono::duration<double> duration = end - start;
  printf("CT-cores took %f seconds\n", duration.count());

  int max_group_nums = N/MAX_GROUP_SIZE+1;
  int j = 0;
  for (int i = 0; i < N && j<max_group_nums; i++) {
    if (mm.isIntree[i] == 0) {
      j++;
      centers.push_back(i);
    }
  }
  groups.resize(N);
  printf("centers size %d\n", centers.size());
  start = std::chrono::high_resolution_clock::now();
  dt.runDijkstra_gpu(centers);
  end = std::chrono::high_resolution_clock::now();
  duration = end - start;
  printf("runDijkstra_gpu took %f seconds\n", duration.count());

  
    // 2.
    // 对于每个点，计算它到每个聚类中心的距离，将它划分到距离最近的聚类中心所在的类中
  #pragma omp parallel for
  for (int i = 0; i < N; ++i) {
      int nearest_center = -1;
      auto min_distance = std::numeric_limits<disType>::max();

      for (auto j : centers) {
        if(group_size[j] > MAX_GROUP_SIZE)
        {
          continue;
        }
          auto distance = dt.query_distance(j, i);
          if (distance < min_distance) {
              nearest_center = j;
              min_distance = distance;
          }
      }

      if (labels[i] != nearest_center) {
          if (labels[i] != -1) {
              group_size[labels[i]]--;  // 递减旧的中心的计数
          }
          group_size[nearest_center]++;  // 递增新的中心的计数
          labels[i] = nearest_center;
      }
  }
  

  for (int i = 0; i < N; ++i) {
      groups[labels[i]].emplace_back(i);
  }
}

void generate_Group_CT_cores_new(graph_v_of_v<disType> &instance_graph, int hop_cst, std::vector<std::vector<int>> &groups) {

    int N = instance_graph.size();
    std::vector<int> labels(N, -1);
    std::unordered_map<int, std::atomic<int>> group_size;
    std::vector<int> centers;

    // Generate CT cores and centers
    CT_case_info mm;
    dijkstra_table dt(instance_graph, false, hop_cst);
    dt.is_gpu = true;
    mm.d = 10;
    mm.use_P2H_pruning = 1;
    mm.two_hop_info.use_2M_prune = 1;
    mm.two_hop_info.use_canonical_repair = 1;
    mm.thread_num = 64;
    auto start = std::chrono::high_resolution_clock::now();
    CT_cores(instance_graph, mm);
    auto end = std::chrono::high_resolution_clock::now();
    printf("CT_cores finished\n");
    std::chrono::duration<double> duration = end - start;
    printf("CT-cores took %f seconds\n", duration.count());

    int max_group_nums = N / MAX_GROUP_SIZE + 1;
    for (int i = 0; i < N && centers.size() < max_group_nums; ++i) {
        if (mm.isIntree[i] == 0) {
            centers.push_back(i);
        }
    }

    // Prepare data for GPU
    dt.runDijkstra_gpu(centers);
    disType* d_distances = dt.get_distances_gpu(); // Assuming the distances are stored on the GPU
    printf("%d\n",centers[0]);
    printf("%lf\n",d_distances[0]);

    cudaEvent_t startt, stop;
    float milliseconds = 0;

    // 创建 CUDA 事件
    cudaEventCreate(&startt);
    cudaEventCreate(&stop);

    // 记录开始时间
    cudaEventRecord(startt);

    // 执行聚类操作
    perform_clustering(N, centers, d_distances, labels);

    // 记录结束时间
    cudaEventRecord(stop);

    // 等待事件完成
    cudaEventSynchronize(stop);

    // 计算时间差（毫秒）
    cudaEventElapsedTime(&milliseconds, startt, stop);

    // 输出执行时间
    printf("Clustering took %f milliseconds\n", milliseconds);

    // 销毁事件
    cudaEventDestroy(startt);
    cudaEventDestroy(stop);

    // Group the vertices based on labels
    groups.resize(N);
    for (int i = 0; i < N; ++i) {
      if(labels[i]==-1)
      {
        printf("error %d\n",i);
        assert(false);
      }
        groups[labels[i]].emplace_back(i);
    }
}

#endif