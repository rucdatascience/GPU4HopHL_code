#include <iostream>

#include <HBPLL/gpu_clean.cuh>

#define THREADS_PER_BLOCK 1024

__global__ void clean_kernel(int V, int K, int tc, label* L, long long* L_start, cuda_vector<label>** Lc, int* hash_array, int* d_uv) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 0 || i >= tc)
        return;

    for (int u = i; u < V; u += tc) {
        // Hashing Lc[u] using the ith hash array
        for (long long label_idx = L_start[u]; label_idx < L_start[u + 1]; label_idx++) {
            d_uv[i] = INT_MAX;
            int v = L[label_idx].v;
            int h_v = L[label_idx].h;
            int check_duv = L[label_idx].d;

            for(int label_id= L_start[v]; label_id< L_start[v + 1]; label_id++){

                int v_x = L[label_id].v;
                int h_x = L[label_id].h;
                int d_vvx = L[label_id].d;
                int update_dis = d_uv[i];
                for (int h_y = 0; h_y <= h_v - h_x; h_y++) {
                    int new_dis = hash_array[(long long)i * V * (K + 1) + v_x * (K + 1) + h_y];
                    if (new_dis == INT_MAX)
                        continue;
                    new_dis += d_vvx;
                    update_dis = update_dis > new_dis ? new_dis : update_dis;
                }
                if (update_dis < d_uv[i]){
                    d_uv[i] = update_dis;
                }
            }

Lc[u]->push_back(L[label_idx]);

            // if (d_uv[i] > check_duv) {
            //     Lc[u]->push_back(L[label_idx]);
            //     hash_array[(long long)i * V * (K + 1) + v * (K + 1) + h_v] = check_duv;
            // }


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

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Error in cudaMallocManaged 1" << std::endl;
        std::cerr << cudaGetErrorString(error) << std::endl;
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

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Error in cudaMallocManaged 2" << std::endl;
        std::cerr << cudaGetErrorString(error) << std::endl;
        return nullptr;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    clean_kernel<<<(tc + THREADS_PER_BLOCK) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(V, K, tc, L, L_start, Lc, hash_array, d_uv);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "GPU Clean Time: " << milliseconds << " ms" << std::endl;

    error = cudaGetLastError();
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