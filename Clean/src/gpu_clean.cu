#include <iostream>

#include <HBPLL/gpu_clean.cuh>

__global__ void query_label(label* L, long long start, long long end, int i, int h_v, int* Lc_hashed, int* d_uv, int V, int K) {
    long long label_id = blockIdx.x * blockDim.x + threadIdx.x;
    label_id += start;

    if (label_id < start || label_id >= end)
        return;
    int v_x = L[label_id].v;
    int h_x = L[label_id].h;
    int d_vvx = L[label_id].d;
    for (int h_y = 0; h_y <= h_v - h_x; h_y++) {
        int update_dis = Lc_hashed[(long long)i * V * (K + 1) + v_x * (K + 1) + h_y];
        if (update_dis == INT_MAX)
            continue;
        update_dis += d_vvx;
        atomicMin(d_uv, update_dis);
    }
}

__global__ void clean_kernel(int V, int K, int tc, label* L, long long* L_start, cuda_vector<label>** Lc, int* hash_array, int* d_uv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 0 || i >= tc)
        return;

    for (int u = i; u < V; u += tc) {
        // Hashing Lc[u] using the ith hash array
        for (long long label_idx = L_start[u]; label_idx < L_start[u + 1]; label_idx++) {
            d_uv[u] = INT_MAX;
            int v = L[label_idx].v;
            int h_v = L[label_idx].h;

            query_label<<<(L_start[v + 1] - L_start[v] + 255) / 256, 256>>>(L, L_start[v], L_start[v + 1], i, h_v, hash_array, &d_uv[u], V, K);
            cudaDeviceSynchronize();
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                printf("CUDA error: %s\n", cudaGetErrorString(error));
                return;
            }

            if (d_uv[u] > L[label_idx].d) {
                Lc[u]->push_back(L[label_idx]);
                hash_array[(long long)i * V * (K + 1) + v * (K + 1) + h_v] = L[label_idx].d;
            }
        }
        // Restore the hash array
        for (int it = 0; it < Lc[u]->current_size; it++) {
            int v = (*Lc[u])[it].v;
            int h_v = (*Lc[u])[it].h;
            hash_array[(long long)i * V * (K + 1) + v * (K + 1) + h_v] = INT_MAX;
        }
    }
}

cuda_vector<label>** gpu_clean(graph_v_of_v<int>& input_graph, vector<vector<label>>& input_L, int tc, int K) {
    int V = input_graph.size();

    vector<label> L_flat;
    long long* L_start = nullptr;

    cudaMallocManaged(&L_start, (V + 1) * sizeof(long long));
    cudaDeviceSynchronize();

    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "Error in cudaMallocManaged" << std::endl;
        return nullptr;
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

    label* L = nullptr;
    cudaMallocManaged(&L, L_flat.size() * sizeof(label));
    cudaDeviceSynchronize();

    cudaMemcpy(L, L_flat.data(), L_flat.size() * sizeof(label), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    base_memory<label>* pool = nullptr;
    cudaMallocManaged(&pool, sizeof(base_memory<label>));
    new (pool) base_memory<label>(1024, (size_t)V * (K + 1) * 1024);

    cuda_vector<label>** Lc = nullptr;
    cudaMallocManaged(&Lc, sizeof(cuda_vector<label>*) * V);
    cudaDeviceSynchronize();

    for (int i = 0; i < V; i++) {
        cudaMallocManaged(&Lc[i], sizeof(cuda_vector<label>));
        new (Lc[i]) cuda_vector<label>(pool, 1024);
    }

    int* hash_array = nullptr; // first dim size is V * (K + 1)

    cudaMallocManaged(&hash_array, sizeof(int) * tc * V * (K + 1));
    cudaDeviceSynchronize();


    for (long long i = 0; i < (long long)tc * V * (K + 1); i++)
        hash_array[i] = INT_MAX;

    int* d_uv = nullptr;
    cudaMallocManaged(&d_uv, sizeof(int) * tc);
    cudaDeviceSynchronize();

    if (cudaGetLastError() != cudaSuccess) {
        std::cerr << "Error in cudaMallocManaged" << std::endl;
        return nullptr;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    clean_kernel<<<(tc + 255) / 256, 256>>>(V, K, tc, L, L_start, Lc, hash_array, d_uv);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "GPU Clean Time: " << milliseconds << " ms" << std::endl;

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        return nullptr;
    }

    cudaFree(L_start);
    cudaFree(L);
    cudaFree(hash_array);
    cudaFree(d_uv);

    return Lc;
}