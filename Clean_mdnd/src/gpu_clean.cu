#include <iostream>
#include <cuda_runtime.h>
#include <HBPLL/gpu_clean.cuh>

#define THREADS_PER_BLOCK 32

__global__ void clean_kernel (int V, int K, int tc, label *L, long long *L_start, int *hash_array, int *mark)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 0 || i >= tc)
        return;

    int offset = i * V * (K + 1);

    for (int u = i; u < V; u += tc)
    {
        for (long long label_idx = L_start[u]; label_idx < L_start[u + 1]; label_idx++)
        {
            int v = L[label_idx].v;
            if (v == u) continue;

            int h_v = L[label_idx].h;
            int check_duv = L[label_idx].d;

            int offset2 = offset + v * (K + 1);

            int update_dis = INT_MAX;
            for (long long label_id = L_start[v]; label_id < L_start[v + 1]; ++label_id)
            {
                int v_x = L[label_id].v;
                if (v_x == v) continue;

                int h_x = L[label_id].h;
                int d_vvx = L[label_id].d;

                int h_y = h_v - h_x;
                if (h_y >= 0)
                {
                    int new_dis = (hash_array + offset)[v_x * (K + 1) + h_y];
                    // printf("new_dis: %d\n", new_dis);
                    if (new_dis != INT_MAX)
                    {
                        new_dis = new_dis + d_vvx;
                        update_dis = min(update_dis, new_dis);
                        if (update_dis <= check_duv)
                        { // new pruning, no effect
                            break;
                        }
                    }
                }
            }

            if (update_dis > check_duv)
            {
                // printf("shit\n");
                for (int x = h_v; x <= K; x++)
                {
                    int z = offset2 + x;
                    hash_array[z] = min(check_duv, hash_array[z]);
                }
            }
            else
            {
                mark[label_idx] = 1;
            }
        }

        // 恢复全局 shared_hash_array 数据
        for (long long label_idx = L_start[u]; label_idx < L_start[u + 1]; label_idx++)
        {
            int v = L[label_idx].v;
            int h_v = L[label_idx].h;

            int offset2 = offset + v * (K + 1);

            if (label_idx - 1 < L_start[u] || L[label_idx - 1].v != v)
            {
                for (int x = h_v; x <= K; x++)
                {
                    hash_array[offset2 + x] = INT_MAX;
                }
            }
        }
    }
}

__global__ void clean_kernel_v2 (int V, int K, int tc, int start_id, int end_id, int *node_id, label *L, long long *L_start, int *hash_array, int *mark) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < 0 || L_start[start_id] + tid >= L_start[end_id + 1]) {
        return;
    }

    long long label_idx = L_start[start_id] + tid;

    int nid = node_id[label_idx];
    int v = L[label_idx].v, h_v = L[label_idx].h, d_v = L[label_idx].d;

    long long offset = (nid - start_id) * V * (K + 1);
    
    if (nid == v) {
        return;
    }

    for (long long label_id = L_start[v]; label_id < L_start[v + 1]; ++ label_id) {
        int vx = L[label_id].v;
        if (vx == v) {
            continue;
        }
        int h_vx = L[label_id].h;
        int d_vx = L[label_id].d;
        if (h_v <= h_vx) {
            continue;
        }

        // long long offset2 = offset + vx * (K + 1);
        int new_dis = (hash_array + offset)[vx * (K + 1) + h_v - h_vx];
        if (new_dis != INT_MAX) {
            new_dis = new_dis + d_vx;
            // update_dis = min(update_dis, new_dis);
            if (new_dis <= d_v) {
                mark[label_idx] = 1;
                break;
            }
        }
    }
}

__global__ void get_hash (int V, int K, int tc, int start_id, int end_id, label *L, long long *L_start, int *hash_array, int *mark) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("gethash\n");
    if (tid < 0 || start_id + tid > end_id) {
        return;
    }
    
    int node_id = start_id + tid;
    long long offset = tid * V * (K + 1);
    for (long long label_idx = L_start[node_id]; label_idx < L_start[node_id + 1]; label_idx++) {
        int v = L[label_idx].v;
        int h_v = L[label_idx].h;
        int d_v = L[label_idx].d;
        // printf("d_v: %d\n", d_v);
        long long offset2 = offset + v * (K + 1);

        for (int x = h_v; x <= K; x++) {
            hash_array[offset2 + x] = min(d_v, hash_array[offset2 + x]);
        }
    }

    return;
}

__global__ void clear_hash (int V, int K, int tc, int start_id, int end_id, label *L, long long *L_start, int *hash_array, int *mark) {
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < 0 || start_id + tid > end_id) {
        return;
    }

    int node_id = start_id + tid;
    long long offset = tid * V * (K + 1);
    for (long long label_idx = L_start[node_id]; label_idx < L_start[node_id + 1]; label_idx++) {
        int v = L[label_idx].v;
        int h_v = L[label_idx].h;
        int d_v = L[label_idx].d;
        long long offset2 = offset + v * (K + 1);

        for (int x = h_v; x <= K; x++) {
            hash_array[offset2 + x] = INT_MAX;
        }
    }

    return;
}

double gpu_clean(graph_v_of_v<int> &input_graph, vector<vector<label>> &input_L, vector<vector<hop_constrained_two_hop_label>> &res, int tc, int K)
{

    int V = input_graph.size();

    vector<label> L_flat;
    long long *L_start = nullptr;

    cudaMallocManaged(&L_start, (V + 1) * sizeof(long long));
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "Error in cudaMallocManaged 1" << std::endl;
        std::cerr << cudaGetErrorString(error) << std::endl;
        // return nullptr;
    }

    long long point = 0;
    for (int i = 0; i < V; i++) {
        L_start[i] = point;
        int _size = input_L[i].size();
        for (int j = 0; j < _size; j++)
            L_flat.push_back(input_L[i][j]);
        point += _size;
    }
    L_start[V] = point;

    int *node_id = nullptr, cnt = 0;
    cudaMallocManaged(&node_id, L_start[V] * sizeof(int));
    cudaDeviceSynchronize();
    for (int i = 0; i < V; i++) {
        int _size = input_L[i].size();
        for (int j = 0; j < _size; j++) {
            node_id[cnt ++] = i;
        }
    }

    label *L = nullptr;
    cudaMallocManaged(&L, L_flat.size() * sizeof(label));
    // cudaDeviceSynchronize();

    cudaMemcpy(L, L_flat.data(), L_flat.size() * sizeof(label), cudaMemcpyHostToDevice);
    // cudaDeviceSynchronize();

    int *mark;
    cudaMallocManaged(&mark, L_flat.size() * sizeof(int));
    cudaMemset(mark, 0, sizeof(int) * L_flat.size());

    int *hash_array = nullptr; // first dim size is V * (K + 1)

    cudaMallocManaged(&hash_array, sizeof(int) * tc * V * (K + 1));
    // cudaDeviceSynchronize();

    for (long long i = 0; i < (long long)tc * V * (K + 1); i++)
        hash_array[i] = INT_MAX;

    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "Error in cudaMallocManaged 2" << std::endl;
        std::cerr << cudaGetErrorString(error) << std::endl;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    int start_id, end_id;
    start_id = V;
    // 每一个 while 会清洗一块标签。
    while (start_id > 0) {
        end_id = start_id - 1;
        start_id = max(0, start_id - tc);
        
        printf("start_id, end_id: %d %d\n", start_id, end_id);

        get_hash <<< (tc + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>
        (V, K, tc, start_id, end_id, L, L_start, hash_array, mark);
        cudaDeviceSynchronize();

        clean_kernel_v2 <<< (L_start[end_id + 1] - L_start[start_id] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>
        (V, K, tc, start_id, end_id, node_id, L, L_start, hash_array, mark);
        cudaDeviceSynchronize();
        
        clear_hash <<< (tc + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>>
        (V, K, tc, start_id, end_id, L, L_start, hash_array, mark);
        cudaDeviceSynchronize();
    }
    // cudaDeviceSynchronize();
    // clean_kernel <<< (tc + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK, THREADS_PER_BLOCK >>> (V, K, tc, L, L_start, hash_array, mark);
    // cudaDeviceSynchronize();
    
    // 将L(csr)转为res(vector<vector>)
    for (int i = 0; i < V; ++i)
    {
        int start = L_start[i];
        int end = L_start[i + 1];
        for (int j = start; j < end; ++j)
        {
            if (mark[j] == 0)
            {
                hop_constrained_two_hop_label temp;
                temp.hub_vertex = L[j].v;
                temp.hop = L[j].h;
                temp.distance = L[j].d;
                res[i].emplace_back(temp);
            }
        }
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    double GPU_clean_time = milliseconds / 1e3;

    error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        // return nullptr;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Max shared memory per block: %d bytes\n", deviceProp.sharedMemPerBlock);

    cudaFree(L_start);
    cudaFree(L);
    cudaFree(hash_array);

    return GPU_clean_time;
}
