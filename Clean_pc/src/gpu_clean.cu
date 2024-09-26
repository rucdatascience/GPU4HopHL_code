#include <HBPLL/gpu_clean.cuh>
#include <climits>
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <unordered_map>
using namespace std;

#define THREADS_PER_BLOCK 256
#define THREADS_TC 256

struct NodeInfo {
  int node_id;        //在图中的节点id
  long long workload; //该节点的工作量
  __host__ __device__ NodeInfo() : node_id(0), workload(0) {}
};

// 定义排序谓词
struct NodeInfoCompare {
  __host__ __device__ bool operator()(const NodeInfo &a,
                                      const NodeInfo &b) const {
    return a.workload > b.workload; // 按照工作量从大到小排序
  }
};

__global__ void clean_kernel_v2(int V, int K, int start_id, int end_id,
                                int *__restrict__ labelIdx2nodeId,
                                label *__restrict__ L,
                                const long long *__restrict__ L_start,
                                int *__restrict__ hash_array, int *mark,
                                int *nodeId2idx, int total_threads) {

  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (L_start[start_id] + tid >= L_start[end_id + 1]) {
    return;
  }

  long long label_idx = L_start[start_id] + tid;

  int nid = labelIdx2nodeId[label_idx];
  int v = L[label_idx].hub_vertex;
  if (nid == v) {
    return;
  }
  int h_v = L[label_idx].hop, d_v = L[label_idx].distance;

 // int tid_nid = nid - start_id; // 当前线程处理的节点相对于 start_id 的偏移

  for (long long label_id = L_start[v];
       label_id < L_start[v + 1]; ++label_id) {
    int vx = L[label_id].hub_vertex;
    
    int h_vx = L[label_id].hop;
    
    if(h_v > h_vx && vx!=v)
    {

      int d_vx = L[label_id].distance;

      // 计算 hash_array 的偏移量
      long long offset = (K + 1) * ((long long)vx * total_threads + (long long)nid - start_id);

      //int idx = offset + h_v - h_vx;
      int new_dis = hash_array[offset + h_v - h_vx];

      //if (new_dis != INT_MAX) {
        if (new_dis <= d_v - d_vx) {
          mark[label_idx] = 1;
          return;
        }
     //}
    }
  }
}

__global__ void get_hash_optimized(int V, int K, int start_id, int end_id,
                                   label *__restrict__ L,
                                   const long long *__restrict__ L_start,
                                   int *__restrict__ hash_array,
                                   int total_threads, int *idx2nodeID) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (L_start[start_id] + tid >= L_start[end_id + 1]) {
    return;
  }

  // 计算标签索引
  long long label_idx = L_start[start_id] + tid;

  // 获取当前标签
  //label current_label = L[label_idx];
  int v = L[label_idx].hub_vertex;
  int h_v = L[label_idx].hop;
  int d_v =L[label_idx].distance;

  // 计算 hash_array 的索引
  // int offset =
  //     v * (K + 1) * total_threads +
  //     (idx2nodeID[L_start[start_id] + tid] - L_start[start_id]) * (K + 1);

  long long offset = (K + 1) * ((long long)v * total_threads + idx2nodeID[label_idx] - start_id);
  for (int x = h_v; x <= K; x++) {
    // atomicMin(&hash_array[offset + x], d_v);
    hash_array[offset+x] = min(hash_array[offset+x],d_v);
  }
}


__global__ void clear_hash_optimized(int V, int K, int start_id, int end_id,
                                     label *__restrict__ L,
                                     const long long *__restrict__ L_start,
                                     int *__restrict__ hash_array,
                                     int total_threads, int *idx2nodeID) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (L_start[start_id] + tid >= L_start[end_id + 1]) {
    return;
  }

  // 计算标签索引
  long long label_idx = L_start[start_id] + tid;

  // 获取当前标签
  int v = L[label_idx].hub_vertex;
  int h_v = L[label_idx].hop;
  int d_v =L[label_idx].distance;

  // 计算 hash_array 的索引
  // int offset =
  //     v * (K + 1) * total_threads +
  //     (idx2nodeID[L_start[start_id] + tid] - L_start[start_id]) * (K + 1);

  long long offset = (K + 1) * ((long long)v * total_threads + (long long)(idx2nodeID[label_idx] - start_id));
  for (int x = h_v; x <= K; x++) {
    hash_array[offset + x] = INT_MAX;
  }
}

double gpu_clean(graph_v_of_v<int> &input_graph, vector<vector<label>> &input_L,
                 vector<vector<hop_constrained_two_hop_label>> &res, int tc,
                 int K) {

  int V = input_graph.size();
  // //打印input_L
  // for(int i = 0; i < V; i++) {
  //   for(int j = 0; j < input_L[i].size(); j++) {
  //     printf("input_L[%d][%d]: {hub_vertex: %d, hop: %d,dis: %d}\n", i, j,
  //     input_L[i][j].hub_vertex,input_L[i][j].hop,input_L[i][j].distance);
  //   }
  //   printf("\n");
  // }

  // 创建 NodeInfo 的主机向量
  vector<NodeInfo> host_node_info(V);

  for (int i = 0; i < V; ++i) {
    NodeInfo info;
    info.node_id = i;
    long long workload = 0;
    for (size_t j = 0; j < input_L[i].size(); ++j) {
      int neighbor = input_L[i][j].hub_vertex;
      workload += input_L[neighbor].size();
    }
    info.workload = workload;
    host_node_info[i] = info;
  }
  // 将 NodeInfo 拷贝到设备端
  //sort(host_node_info.begin(), host_node_info.end(), NodeInfoCompare());
  // {
  // thrust::device_vector<NodeInfo> node_info_vec = host_node_info;

  // // 使用 Thrust 进行排序
  // thrust::sort(node_info_vec.begin(), node_info_vec.end(),
  // NodeInfoCompare());
  // // 将排序后的 NodeInfo 拷贝回主机端
  // thrust::copy(node_info_vec.begin(), node_info_vec.end(),
  //              host_node_info.begin());
  // }

  int *nodeId2idx = new int[V];
  int *nodeId2idx_device = nullptr;
  cudaMalloc(&nodeId2idx_device, V * sizeof(int));
  for (int i = 0; i < V; i++) {
    nodeId2idx[host_node_info[i].node_id] = i;
  }
  cudaMemcpyAsync(nodeId2idx_device, nodeId2idx, V * sizeof(int),
                  cudaMemcpyHostToDevice, 0);

  long long total_memsize = 0; // B

  // vector<label> L_flat;
  long long *L_start =
      new long long[V + 1]; //代表排序后i点的node_id的label的起始位置
  long long *L_start_device = nullptr;

  cudaMalloc(&L_start_device, (V + 1) * sizeof(long long));
  total_memsize += (V + 1) * 8;
  // cudaDeviceSynchronize();

  long long point = 0;
  long long total_size = 0;
  for (int i = 0; i < V; i++) {
    L_start[i] = point;
    int _size = input_L[host_node_info[i].node_id].size();
    // printf("point[%d]: %ld, ", i,point);
    point += _size;
  }
  total_size = point;
  L_start[V] = point;
  // printf("\n\n");

  cudaMemcpyAsync(L_start_device, L_start, sizeof(long long) * (V + 1),
                  cudaMemcpyHostToDevice, 0);

  // printf("total_size: %ld\n\n", L_start[V]);

  int *node_id = nullptr; //这是用来标识属于第几个节点的label，值是排序后的
  int cnt = 0;
  cudaMallocManaged(&node_id, total_size * sizeof(int));
  total_memsize += total_size * 4;

  label *L = nullptr;
  cudaMallocManaged(&L, total_size * sizeof(label));
  total_memsize += total_size * sizeof(label);

  for (int i = 0; i < V; i++) {
    int _size = input_L[host_node_info[i].node_id].size();
    // printf("host_node_info[%d].node_id= %d\n", i, host_node_info[i].node_id);
    for (int j = 0; j < _size; j++) {
      node_id[cnt] = i;
      L[cnt] = input_L[host_node_info[i].node_id][j];
      // printf("node_id[%d]: %d, ", cnt, node_id[cnt]);
      // printf("L[%d].hub_vertex: %d, \n\n", cnt, L[cnt].hub_vertex);
      ++cnt;
    }
  }

  int *mark;
  cudaMalloc(&mark, total_size * sizeof(int));
  total_memsize += total_size * 4;
  cudaMemset(mark, 0, sizeof(int) * total_size);

  int *host_mark;
  host_mark = new int[total_size];
  memset(host_mark, 0, total_size * sizeof(int));

  int *hash_array = nullptr; // first dim size is V * (K+1)

  cudaMallocManaged(&hash_array, sizeof(int) * tc * V * (K + 1));
  total_memsize += sizeof(int) * tc * V * (K + 1);
  // printf("total memory: %ld\n\n", total_memsize / 1024);
  //  cudaDeviceSynchronize();

  for (long long i = 0; i < (long long)tc * V * (K + 1); i++)
    hash_array[i] = INT_MAX;

  // cudaDeviceSynchronize();
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  unordered_map<int, cudaStream_t> streams; // 创建 tc 个 stream
  for (int i = 0; i < V; i += tc) {
    cudaStream_t temp;
    cudaStreamCreate(&temp); // 为每个循环迭代创建一个流
    streams[i] = temp;
  }

  int start_id, end_id;
  start_id = 0;

  cudaStreamSynchronize(0);

  
  // 创建事件，在循环外部创建，以避免重复创建销毁的开销
cudaEvent_t start1, stop1;
cudaEventCreate(&start1);
cudaEventCreate(&stop1);

while (start_id < V) {
    end_id = min(start_id + tc - 1, V - 1);
    long long num_labels = L_start[end_id + 1] - L_start[start_id];

    printf("start_id, end_id: %d %d\n", start_id, end_id);

    // 记录开始时间，在对应的流上
    cudaEventRecord(start1, streams[start_id]);

    // 调用 get_hash_optimized 内核，注意确保所有用于索引计算的变量都进行适当的类型转换，防止整数溢出
    get_hash_optimized<<<(num_labels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                         THREADS_PER_BLOCK, 0, streams[start_id]>>>(
        V, K, start_id, end_id, L, L_start_device, hash_array, tc, node_id);

    // 记录结束时间，在同一个流上
    cudaEventRecord(stop1, streams[start_id]);

    // 等待事件完成，确保内核执行完毕
    cudaEventSynchronize(stop1);

    // 计算耗时
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start1, stop1);
    printf("get hash: %f s\n\n", milliseconds / 1000);

    // 同样的方式处理 clean_kernel_v2 内核
    cudaEventRecord(start1, streams[start_id]);

    clean_kernel_v2<<<(num_labels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                      THREADS_PER_BLOCK, 0, streams[start_id]>>>(
        V, K, start_id, end_id, node_id, L, L_start_device, hash_array, mark,
        nodeId2idx_device, tc);

    cudaEventRecord(stop1, streams[start_id]);
    cudaEventSynchronize(stop1);

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start1, stop1);
    printf("clean kernel: %f s\n\n", milliseconds / 1000);

    // 处理 clear_hash_optimized 内核
    cudaEventRecord(start1, streams[start_id]);

    clear_hash_optimized<<<(num_labels + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK,
                           THREADS_PER_BLOCK, 0, streams[start_id]>>>(
        V, K, start_id, end_id, L, L_start_device, hash_array, tc, node_id);

    cudaEventRecord(stop1, streams[start_id]);
    cudaEventSynchronize(stop1);

    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start1, stop1);
    printf("clear hash: %f s\n\n", milliseconds / 1000);

    // 同步流，确保所有操作完成
    cudaStreamSynchronize(streams[start_id]);

    // 异步复制 mark 数据到主机，确保在正确的流上进行
    long long start = L_start[start_id];
    long long end = L_start[end_id + 1];

    cudaMemcpyAsync(host_mark + start, mark + start,
                    sizeof(int) * (end - start), cudaMemcpyDeviceToHost,
                    streams[start_id]);

    // 继续下一个批次
    start_id += tc;
}

// 在循环结束后，销毁事件
cudaEventDestroy(start1);
cudaEventDestroy(stop1);


  // cudaMemcpy(host_mark,mark,total_size*sizeof(int),cudaMemcpyDeviceToHost);

  // 结束后同步所有 streams
  for (int i = 0; i < V; i += tc) {
    cudaStreamSynchronize(streams[i]); // 等待所有流的任务完成
  }

  // 最后销毁所有 streams
  for (int i = 0; i < V; i += tc) {
    cudaStreamDestroy(streams[i]);
  }

  // 将L(csr)转为res(vector<vector>)

  //#pragma omp parallel for
  for (int i = 0; i < V; ++i) {
    int start = L_start[i];
    int end = L_start[i + 1];
    int s1 = 0;
    int e1 = 0;
    for (int j = start; j < end; ++j) {
      if (host_mark[j] == 0) {
        e1++;
      } else {
        res[host_node_info[i].node_id].insert(
            res[host_node_info[i].node_id].end(),
            input_L[host_node_info[i].node_id].begin() + s1,
            input_L[host_node_info[i].node_id].begin() + e1);
        e1 += 1;
        s1 = e1;
      }
    }
    if (e1 == input_L[host_node_info[i].node_id].size()) {
      res[host_node_info[i].node_id].insert(
          res[host_node_info[i].node_id].end(),
          input_L[host_node_info[i].node_id].begin() + s1,
          input_L[host_node_info[i].node_id].begin() + e1);
    }
  }

  // L_start[V] = point;
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  double GPU_clean_time = milliseconds / 1e3;

  // printf("total: %ld\n", total);

  cudaFree(L_start_device);
  cudaFree(L);
  cudaFree(node_id);
  cudaFree(mark);
  cudaFree(hash_array);
  cudaFree(nodeId2idx_device);

  free(L_start);
  delete[] host_mark;
  delete[] nodeId2idx;

  return GPU_clean_time;
}
