#include <iostream>
#include <cuda_runtime.h>
#include <HBPLL/gpu_clean.cuh>

#define THREADS_PER_BLOCK 32

// 64bits, hub_vertex 24bits, parent_vertex 24bits, hop 3bits, distance 10bits
inline __host__ __device__ int get_hub_vertex (long long x) {
    return (x >> 37);
}
inline __host__ __device__ int get_parent_vertex (long long x) {
    return (x >> 13) & ((1 << 24) - 1);
}
inline __host__ __device__ int get_hop (long long x) {
    return (x >> 10) & ((1 << 3) - 1);
}
inline __host__ __device__ int get_distance (long long x) {
    return (x) & ((1 << 10) - 1);
}
inline __host__ __device__ long long get_label (int hub_vertex, int parent_vertex, int hop, int distance) {
    return ((long long)hub_vertex << 37) | ((long long)parent_vertex << 13) | ((long long)hop << 10) | ((long long)distance);
}

__global__ void clean_kernel (int V, int K, int tc, long long *L, long long *L_start, int *hash_array, int *mark) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 0 || i >= tc)
        return;

    int offset = i * V * (K + 1);

    for (int u = i; u < V; u += tc) {
        for (long long label_idx = L_start[u]; label_idx < L_start[u + 1]; label_idx++) {
            int v = get_hub_vertex(L[label_idx]);
            if (v == u) continue;

            int h_v = get_distance(L[label_idx]);
            int check_duv = get_distance(L[label_idx]);

            int offset2 = offset + v * (K + 1);

            int update_dis = INT_MAX;
            for (long long label_id = L_start[v]; label_id < L_start[v + 1]; ++label_id) {
                int v_x = get_hub_vertex(L[label_id]);
                if (v_x == v) continue;

                int h_x = get_hop(L[label_id]);
                int d_vvx = get_distance(L[label_id]);

                int h_y = h_v - h_x;
                if (h_y >= 0) {
                    int new_dis = (hash_array + offset)[v_x * (K + 1) + h_y];
                    // printf("new_dis: %d\n", new_dis);
                    if (new_dis != INT_MAX) {
                        new_dis = new_dis + d_vvx;
                        update_dis = min(update_dis, new_dis);
                        if (update_dis <= check_duv) { // new pruning, no effect
                            break;
                        }
                    }
                }
            }

            if (update_dis > check_duv) {
                for (int x = h_v; x <= K; x++) {
                    int z = offset2 + x;
                    hash_array[z] = min(check_duv, hash_array[z]);
                }
            } else {
                mark[label_idx] = 1;
            }
        }

        // 恢复全局 shared_hash_array 数据
        for (long long label_idx = L_start[u]; label_idx < L_start[u + 1]; label_idx++) {
            int v = get_hub_vertex(L[label_idx]);
            int h_v = get_hop(L[label_idx]);
            int offset2 = offset + v * (K + 1);
            if (label_idx - 1 < L_start[u] || get_hub_vertex(L[label_idx - 1]) != v) {
                for (int x = h_v; x <= K; x++) {
                    hash_array[offset2 + x] = INT_MAX;
                }
            }
        }
    }
}

__global__ void clean_kernel_v2 (int V, int K, int tc, int start_id, int end_id, int *node_id, long long *L, 
long long *L_start, long long *L_end, int *nid_to_tid, int *hash_array, int *mark, int *nidd) {

    long long tid = blockIdx.x * blockDim.x + threadIdx.x;

    start_id = nidd[start_id], end_id = nidd[end_id];

    if (tid < 0 || L_start[start_id] + tid >= L_end[end_id]) {
        return;
    }

    long long label_idx = L_start[start_id] + tid;
    int nid = node_id[label_idx];
    int v = get_hub_vertex(L[label_idx]), h_v = get_hop(L[label_idx]), d_v = get_distance(L[label_idx]);
    long long offset = (long long) nid_to_tid[nid] * V * (K + 1);
    
    if (nid == v) {
        return;
    }

    long long LL;
    for (long long label_id = L_start[v]; label_id < L_end[v]; ++ label_id) {
        LL = L[label_id];
        int vx = get_hub_vertex(LL);
        if (vx == v) {
            continue;
        }
        int h_vx = get_hop(LL);
        int d_vx = get_distance(LL);
        if (h_v <= h_vx) {
            continue;
        }

        // long long offset2 = offset + vx * (K + 1);
        int new_dis = (hash_array + offset) [vx * (K + 1) + h_v - h_vx];
        if (new_dis != INT_MAX) {
            new_dis = new_dis + d_vx;
            if (new_dis <= d_v) {
                // if (new_dis != d_v) {
                //     printf("start_id, v1, v2, dis1, dis2, dis3, hop1, hop2, : %d, %d, %d, %d, %d, %d, %d, %d\n", nid, v, vx, new_dis - d_vx, d_vx, d_v, h_v, h_vx);
                // }
                mark[label_idx] = 1;
                break;
            }
        }
    }
}

__global__ void get_hash (int V, int K, int tc, int start_id, int end_id, long long *L, 
long long *L_start, long long *L_end, int *nid_to_tid, int *hash_array, int *mark, int *nid) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 0 || start_id + tid > end_id) {
        return;
    }
    
    int node_id = nid[start_id + tid];
    nid_to_tid[node_id] = tid;
    long long offset = tid * V * (K + 1);
    long long LL;
    for (long long label_idx = L_start[node_id]; label_idx < L_end[node_id]; ++label_idx) {
        LL = L[label_idx];
        int v = get_hub_vertex(LL);
        int h_v = get_hop(LL);
        int d_v = get_distance(LL);
        
        long long offset2 = offset + v * (K + 1);

        for (int x = h_v; x <= K; x++) {
            hash_array[offset2 + x] = min(d_v, hash_array[offset2 + x]);
        }
    }

    return;
}

__global__ void clear_hash (int V, int K, int tc, int start_id, int end_id, long long *L, long long *L_start, long long *L_end, int *hash_array, int *mark, int *nid) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 0 || start_id + tid > end_id) {
        return;
    }

    int node_id = nid[start_id + tid];
    long long offset = tid * V * (K + 1);
    long long LL;
    for (long long label_idx = L_start[node_id]; label_idx < L_end[node_id]; ++label_idx) {
        LL = L[label_idx];
        int v = get_hub_vertex(LL);
        int h_v = get_hop(LL);
        int d_v = get_distance(LL);
        long long offset2 = offset + v * (K + 1);

        for (int x = h_v; x <= K; x++) {
            hash_array[offset2 + x] = INT_MAX;
        }
    }

    return;
}

void gpu_clean_init (graph_v_of_v<int> &input_graph, vector<vector<hop_constrained_two_hop_label>> &input_L, hop_constrained_case_info_v2 * info_gpu, Graph_pool<int>& graph_pool, int tc, int K) {
    
    int V = input_graph.size();
    vector<long long> L_flat;

    cudaMallocManaged(&info_gpu->L_start, (long long) (V + 1) * sizeof(long long));
    cudaMallocManaged(&info_gpu->L_end, (long long) (V + 1) * sizeof(long long));
    cudaDeviceSynchronize();

    long long point = 0;
    int x;
    for (int i = 0; i < graph_pool.size(); ++i) {
        for (int k = 0; k < graph_pool.graph_group[i].size(); ++k) {
            x = graph_pool.graph_group[i][k];
            info_gpu->L_start[x] = point;
            int _size = input_L[x].size();
            for (int j = 0; j < _size; j++) {
                L_flat.push_back(get_label(input_L[x][j].hub_vertex, input_L[x][j].parent_vertex, 
                                           input_L[x][j].hop, input_L[x][j].distance));
            }
            point += _size;
            info_gpu->L_end[x] = point;
        }
    }

    int cnt = 0;
    cudaMallocManaged(&info_gpu->node_id, (long long) point * sizeof(int));
    cudaDeviceSynchronize();
    for (int i = 0; i < graph_pool.size(); ++i) {
        for (int k = 0; k < graph_pool.graph_group[i].size(); ++k) {
            x = graph_pool.graph_group[i][k];
            int _size = input_L[x].size();
            for (int j = 0; j < _size; j++) {
                info_gpu->node_id[cnt ++] = x;
            }
        }
    }

    long long *L = nullptr;
    cudaMallocManaged(&info_gpu->L, (long long) L_flat.size() * sizeof(long long));
    cudaDeviceSynchronize();

    cudaMemcpy(info_gpu->L, L_flat.data(), (long long) L_flat.size() * sizeof(long long), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaMallocManaged(&info_gpu->mark, (long long) L_flat.size() * sizeof(int));
    cudaMemset(info_gpu->mark, 0, (long long) sizeof(int) * L_flat.size());

    cudaMallocManaged(&info_gpu->hash_array, (long long) sizeof(int) * tc * V * (K + 1));
    cudaDeviceSynchronize();

    for (long long i = 0; i < (long long) tc * V * (K + 1); i++){
        info_gpu->hash_array[i] = INT_MAX;
    }
    
    cudaMallocManaged(&info_gpu->nid, (long long) sizeof(int*) * graph_pool.graph_group.size());
    cudaMallocManaged(&info_gpu->nid_size, (long long) sizeof(int) * graph_pool.graph_group.size());
    cudaDeviceSynchronize();
    for (int j = 0; j < graph_pool.graph_group.size(); ++ j) {
        cudaMallocManaged(&info_gpu->nid[j], (long long) sizeof(int) * graph_pool.graph_group[j].size());
        cudaDeviceSynchronize();
        info_gpu->nid_size[j] = graph_pool.graph_group[j].size();
        for (int k = 0; k < graph_pool.graph_group[j].size(); ++k) {
            info_gpu->nid[j][k] = graph_pool.graph_group[j][k];
        }
    }
    cudaDeviceSynchronize();

    cudaMallocManaged(&info_gpu->nid_to_tid, (long long) V * sizeof(int));
    cudaDeviceSynchronize();
}

void gpu_clean(graph_v_of_v<int> &input_graph, hop_constrained_case_info_v2 * info_gpu, 
vector<vector<hop_constrained_two_hop_label>> &res, int thread_num, int nid_vec_id) {

    int V = input_graph.size();
    int K = info_gpu->hop_cst;

    long long *L_start = info_gpu->L_start;
    long long *L_end = info_gpu->L_end;

    int *nid_to_tid = info_gpu->nid_to_tid;
    int *node_id = info_gpu->node_id;
    long long *L = info_gpu->L;
    int *mark = info_gpu->mark;
    int *hash_array = info_gpu->hash_array; // first dim size is V * (K + 1)

    int *nid = info_gpu->nid[nid_vec_id];
    int nid_size = info_gpu->nid_size[nid_vec_id];

    // cudaEvent_t start, stop;
    // cudaEventCreate(&start);
    // cudaEventCreate(&stop);

    // cudaEventRecord(start);
    int start_id, end_id, start_node_id, end_node_id;
    start_id = nid_size;
    // 每一�? while 会清洗一块标签�?
    auto begin = std::chrono::high_resolution_clock::now();
    while (start_id > 0) {
        end_id = start_id - 1;
        start_id = max(0, start_id - thread_num);
        
        end_node_id = nid[end_id];
        start_node_id = nid[start_id];

        // printf("start_id, end_id: %d %d\n", start_id, end_id);
        get_hash <<< (thread_num + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>
        (V, K, thread_num, start_id, end_id, L, L_start, L_end, nid_to_tid, hash_array, mark, nid);
        cudaDeviceSynchronize();
        
        clean_kernel_v2 <<< (L_end[end_node_id] - L_start[start_node_id] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>
        (V, K, thread_num, start_id, end_id, node_id, L, L_start, L_end, nid_to_tid, hash_array, mark, nid);
        cudaDeviceSynchronize();
        
        clear_hash <<< (thread_num + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>
        (V, K, thread_num, start_id, end_id, L, L_start, L_end, hash_array, mark, nid);
        cudaDeviceSynchronize();
        
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
    printf("Duration: %lf\n", duration);
    // clean_kernel <<< (tc + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>> (V, K, tc, L, L_start, hash_array, mark);
    // cudaDeviceSynchronize();

    // 将L(csr)转为res(vector<vector>)
    for (int i = 0; i < nid_size; ++i) {
        results_gpu.emplace_back(pool_gpu.enqueue(
            [i, nid_vec_id, &res, &info_gpu] { // pass const type value j to thread; [] can be empty
                int node_id = info_gpu->nid[nid_vec_id][i];
                long long start = info_gpu->L_start[node_id];
                long long end = info_gpu->L_end[node_id];

                res[node_id].clear();
                long long LL;
                for (long long j = start; j < end; ++j) {
                    if (info_gpu->mark[j] == 0) {
                        hop_constrained_two_hop_label temp;
                        LL = info_gpu->L[j];
                        temp.hub_vertex = get_hub_vertex(LL);
                        temp.hop = get_hop(LL);
                        temp.distance = get_distance(LL);
                        temp.parent_vertex = get_parent_vertex(LL);
                        res[node_id].push_back(temp);
                    } else {
                        // printf("clean label: %d %d %d %d\n !", get_hub_vertex(info_gpu->L[j]), get_hop(info_gpu->L[j]), 
                        //                                        get_distance(info_gpu->L[j]), get_parent_vertex(info_gpu->L[j]));
                    }
                }
                return 1;
            }));
    }
    for (auto &&result : results_gpu) {
	    result.get(); // all threads finish here
    }
	results_gpu.clear();
    
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    
    // float milliseconds = 0;
    // cudaEventElapsedTime(&milliseconds, start, stop);
    // double GPU_clean_time = milliseconds / 1e3;

    return;
}

/*

clean label: 7054 5 183 9757
clean label: 12804 5 253 23639

clean label: 13404 5 275 13522
clean label: 7827 5 293 11799

clean label: 8032 5 331 14727
clean label: 7054 5 183 9757

clean label: 13189 5 185 18523
clean label: 7827 5 293 11799

*/