#include <iostream>
#include <cuda_runtime.h>
#include <HBPLL/gpu_clean.cuh>

#define THREADS_PER_BLOCK 1024
#define clean_thread_num 1000

int *L2_pos;
int *L_size;
long long L_tot = 0;

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

bool cmp_L(long long x, long long y) {
    int vx1 = get_hub_vertex(x);
    int vx2 = get_parent_vertex(x);
    int vy1 = get_hub_vertex(y);
    int vy2 = get_parent_vertex(y);
    return min(vx1, vx2) > min(vy1, vy2);
}

// get hash_table
__global__ void get_hash_v2 (int V, int hop_cst, int vid, int *in_L, long long *L, long long *L_start, long long *L_end, int *hash_array) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 0 || tid >= L_end[vid] - L_start[vid]) {
        return;
    }
    long long LL = L[L_start[vid] + tid]; // get the label need to get hash
    int hub_vertex = get_hub_vertex(LL);
    int hop = get_hop(LL);
    int dis = get_distance(LL);
    int offset = hub_vertex * (hop_cst + 1) + hop;
    for (int x = hop; x <= hop_cst; ++ x) {
        if (hash_array[offset] > dis) {
            atomicMin(&hash_array[offset ++], dis);
        } else {
            break;
        }
    }
    in_L[hub_vertex] = 1;
    return;
}

// clean hash_table
__global__ void clear_hash_v2 (int V, int hop_cst, int vid, int *in_L, long long *L, long long *L_start, long long *L_end, int *hash_array) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 0 || tid >= L_end[vid] - L_start[vid]) {
        return;
    }
    long long LL = L[L_start[vid] + tid]; // get the label need to get hash
    int hub_vertex = get_hub_vertex(LL);
    int hop = get_hop(LL);
    int offset = hub_vertex * (hop_cst + 1) + hop;
    for (int x = hop; x <= hop_cst; ++ x) {
        if (hash_array[offset] != (1 << 14)) {
            hash_array[offset ++] = (1 << 14);
        } else {
            break;
        }
        
    }
    in_L[hub_vertex] = 0;
    return;
}

__global__ void clean_check (int hop_cst, int vid, long long L_tot, int *in_L, long long *L, int *hash_array, int *mark) {
    long long tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < 0 || tid >= L_tot) {
        return;
    }

    if (mark[tid]) return;

    long long LL = L[tid];
    
    int st_vertex = (LL >> 37);
    int ed_vertex = (LL >> 13) & ((1 << 24) - 1);
    
    if (!in_L[st_vertex] || !in_L[ed_vertex]) {
        return;
    }

    int hop_now = (LL >> 10) & ((1 << 3) - 1);
    int dis = (LL) & ((1 << 10) - 1);
    st_vertex = st_vertex * (hop_cst + 1) + hop_now;
    ed_vertex = ed_vertex * (hop_cst + 1);

    for (int i = hop_now; i >= 0; -- i) {
        if (hash_array[st_vertex --] + hash_array[ed_vertex ++] <= dis) {
            mark[tid] = 1;
            return;
        }
    }

    return;
}

void gpu_clean_init_v2 (graph_v_of_v<int> &input_graph, vector<vector<hop_constrained_two_hop_label>> &input_L, hop_constrained_case_info_v2 * info_gpu, Graph_pool<int>& graph_pool, int tc, int K) {
    int V = input_graph.size();

    vector<vector<hop_constrained_two_hop_label>> transfer_L;
    transfer_L.resize(V);

    for (int i = 0; i < V; ++ i) {
        for (int j = 0; j < input_L[i].size(); ++ j) {
            hop_constrained_two_hop_label temp;
            temp.hub_vertex = i;
            temp.hop = input_L[i][j].hop;
            temp.distance = input_L[i][j].distance;
            temp.parent_vertex = input_L[i][j].hub_vertex;
            transfer_L[input_L[i][j].hub_vertex].push_back(temp);
        }
    }

    vector<long long> L_flat;

    cudaMallocManaged(&info_gpu->L_start, (long long) (V + 1) * sizeof(long long));
    cudaMallocManaged(&info_gpu->L_end, (long long) (V + 1) * sizeof(long long));
    cudaDeviceSynchronize();

    L_size = (int*) malloc(sizeof(int) * V);

    long long point = 0;
    int x;
    for (int i = 0; i < V; ++i) {
        info_gpu->L_start[i] = point;
        int _size = transfer_L[i].size();
        for (int j = 0; j < _size; j++) {
            L_flat.push_back(get_label(transfer_L[i][j].hub_vertex, transfer_L[i][j].parent_vertex, 
                                       transfer_L[i][j].hop, transfer_L[i][j].distance));
        }
        point += _size;
        info_gpu->L_end[i] = point;
        L_size[i] = _size;
        L_tot += _size;
    }

    // get L
    cudaMallocManaged(&info_gpu->L, (long long) L_flat.size() * sizeof(long long));
    cudaDeviceSynchronize();
    cudaMemcpy(info_gpu->L, L_flat.data(), (long long) L_flat.size() * sizeof(long long), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // get L2
    sort(L_flat.begin(), L_flat.end(), cmp_L);
    // int pos = 0;
    L2_pos = (int*) malloc(sizeof(int) * L_flat.size());
    for(int i = 0; i <= V; ++ i) L2_pos[i] = -1;
    for(int i = 0; i < L_flat.size(); ++ i){
        int now_mn = min(get_hub_vertex(L_flat[i]), get_parent_vertex(L_flat[i]));
        L2_pos[now_mn] = i;
    }
    for(int i = 0; i < V; ++ i) L2_pos[i] = L2_pos[i + 1];

    // get L2
    cudaMallocManaged(&info_gpu->L2, (long long) L_flat.size() * sizeof(long long));
    cudaDeviceSynchronize();
    cudaMemcpy(info_gpu->L2, L_flat.data(), (long long) L_flat.size() * sizeof(long long), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaMallocManaged(&info_gpu->mark, (long long) L_flat.size() * sizeof(int));
    cudaMemset(info_gpu->mark, 0, (long long) sizeof(int) * L_flat.size());

    cudaMallocManaged(&info_gpu->hash_array, (long long) V * sizeof(int) * (K + 1));
    cudaDeviceSynchronize();

    for (long long i = 0; i < (long long) V * (K + 1); i++){
        info_gpu->hash_array[i] = (1 << 14);
    }
    cudaDeviceSynchronize();

    cudaMallocManaged(&info_gpu->in_L, (long long) V * sizeof(int));
    cudaDeviceSynchronize();
}

void gpu_clean_v2 (graph_v_of_v<int> &input_graph, hop_constrained_case_info_v2 * info_gpu, 
vector<vector<hop_constrained_two_hop_label>> &res, int thread_num) {

    int V = input_graph.size();
    int K = info_gpu->hop_cst;

    long long *L_start = info_gpu->L_start;
    long long *L_end = info_gpu->L_end;

    long long *L = info_gpu->L;
    long long *L2 = info_gpu->L2;
    int *in_L = info_gpu->in_L;

    int *mark = info_gpu->mark;
    int *hash_array = info_gpu->hash_array; // first dim size is V * (K + 1)

    int start_id = V, end_id, start_node_id, end_node_id;

    double tot_duration = 0.0, tot_duration_check = 0.0;
    
    auto begin_for = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < V; ++ i) {
        auto begin = std::chrono::high_resolution_clock::now();

        get_hash_v2 <<< (L_size[i] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>
        (V, K, i, in_L, L, L_start, L_end, hash_array);
        cudaDeviceSynchronize();
        
        auto begin_check = std::chrono::high_resolution_clock::now();
        clean_check <<< (L2_pos[i] + 1 + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>> 
        (K, i, L2_pos[i] + 1, in_L, L2, hash_array, mark);
        cudaDeviceSynchronize();
        auto end_check = std::chrono::high_resolution_clock::now();
        auto duration_check = std::chrono::duration_cast<std::chrono::nanoseconds>(end_check - begin_check).count() / 1e9;
        tot_duration_check += duration_check;

        clear_hash_v2 <<< (L_size[i] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>> 
        (V, K, i, in_L, L, L_start, L_end, hash_array);
        cudaDeviceSynchronize();

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
        tot_duration += duration;
        if (i % 1000 == 0) {
            printf("%lf, %lf\n", tot_duration, tot_duration_check);
            tot_duration = 0;
            tot_duration_check = 0;
        }
    }
    auto end_for = std::chrono::high_resolution_clock::now();
    auto duration_for = std::chrono::duration_cast<std::chrono::nanoseconds>(end_for - begin_for).count() / 1e9;
    
    printf("\n%lf\n", duration_for);

    long long LL;
    for (int i = 0; i < V; ++ i) {
        res[i].clear();
    }

    for (int i = 0; i < L_tot; ++ i) {
        
    }
    for (int i = 0; i < L_tot; ++ i) {
        if (info_gpu->mark[i] == 0) {
            hop_constrained_two_hop_label temp;
            LL = info_gpu->L2[i];
            temp.hub_vertex = get_parent_vertex(LL);
            temp.hop = get_hop(LL);
            temp.distance = get_distance(LL);
            temp.parent_vertex = get_hub_vertex(LL);
            res[temp.parent_vertex].push_back(temp);
        }
    }

    return;
}