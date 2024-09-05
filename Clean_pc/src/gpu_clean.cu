#include <iostream>
#include <cuda_runtime.h>
#include <HBPLL/gpu_clean.cuh>

#define THREADS_PER_BLOCK 1024

__global__ void clean_kernel(int V, int K, int tc, label* L, long long* L_start, int* hash_array, int* mark) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 0 || i >= tc)
        return;
    int offset = i*V*(K+1);
    for (int u = i; u < V; u += tc) {
        for (long long label_idx = L_start[u]; label_idx < L_start[u + 1]; label_idx++) {
            int v = L[label_idx].v;
            int h_v = L[label_idx].h;
            int check_duv = L[label_idx].d;

            //d_uv = query_label(L, L_start[v], L_start[v + 1], i, h_v, hash_array + offset, V, K);
            int update_dis = INT_MAX;
    for(long long label_id = L_start[v]; label_id < L_start[v + 1]; ++label_id) {
        int v_x = L[label_id].v;
        int h_x = L[label_id].h;
        int d_vvx = L[label_id].d;

        for (int h_y = 0; h_y <= h_v - h_x; h_y++) {
            int new_dis = (hash_array + offset)[v_x * (K + 1) + h_y];
            new_dis = (new_dis == INT_MAX) ? INT_MAX : new_dis + d_vvx;
            update_dis = (update_dis > new_dis) ? new_dis : update_dis;
        }
        // for (int h_y = h_v - h_x; h_y <= h_v - h_x; h_y++) {
        //     int new_dis = (hash_array + offset)[v_x * (K + 1) + h_y];
        //     new_dis = (new_dis == INT_MAX) ? INT_MAX : new_dis + d_vvx;
        //     update_dis = (update_dis > new_dis) ? new_dis : update_dis;
        // }

        if(update_dis <= check_duv){  // new pruning, no effect
            break;
        }
    }

            if (update_dis > check_duv) {
                hash_array[offset + v * (K + 1) + h_v] = check_duv;
// for(int x = h_v; x<= K; x++){
//     hash_array[offset + v * (K + 1) + x] = min(check_duv, hash_array[offset + v * (K + 1) + x]);
// }

            } else {
                mark[label_idx] = 1;
            }
        }

        // 恢复全局 shared_hash_array 数据
        for (long long label_idx = L_start[u]; label_idx < L_start[u + 1]; label_idx++) {
            int v = L[label_idx].v;
            int h_v = L[label_idx].h;

            hash_array[offset + v * (K + 1) + h_v] = INT_MAX;
// for(int x = h_v; x<= K; x++){
//     hash_array[offset + v * (K + 1) + x] = INT_MAX;
// }

        }
    }
}


double gpu_clean(graph_v_of_v<int>& input_graph, vector<vector<label>>& input_L, vector<vector<hop_constrained_two_hop_label>>& res, int tc, int K) {

    int V = input_graph.size();

    vector<label> L_flat;
    long long* L_start = nullptr;

    cudaMallocManaged(&L_start, (V + 1) * sizeof(long long));
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Error in cudaMallocManaged 1" << std::endl;
        std::cerr << cudaGetErrorString(error) << std::endl;
        //return nullptr;
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
    //cudaDeviceSynchronize();

    cudaMemcpy(L, L_flat.data(), L_flat.size() * sizeof(label), cudaMemcpyHostToDevice);
    //cudaDeviceSynchronize();

    int *mark;
    cudaMallocManaged(&mark,L_flat.size() * sizeof(int));
    cudaMemset(mark,0,sizeof(int)*L_flat.size());

    int* hash_array = nullptr; // first dim size is V * (K + 1)

    cudaMallocManaged(&hash_array, sizeof(int) * tc * V * (K + 1));
    //cudaDeviceSynchronize();

    for (long long i = 0; i < (long long)tc * V * (K + 1); i++)
        hash_array[i] = INT_MAX;

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Error in cudaMallocManaged 2" << std::endl;
        std::cerr << cudaGetErrorString(error) << std::endl;
    }

    

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    clean_kernel<<<(tc + THREADS_PER_BLOCK) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(V, K, tc, L, L_start, hash_array, mark);
    cudaDeviceSynchronize();


    //将L(csr)转为res(vector<vector>)
    for(int i = 0; i < V; ++i)
{
    int start = L_start[i];
    int end = L_start[i+1];
    for(int j = start; j < end; ++j)
    {
        if(mark[j]==0)
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
    double GPU_clean_time = milliseconds/1e3;

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        //return nullptr;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Max shared memory per block: %d bytes\n", deviceProp.sharedMemPerBlock);

    cudaFree(L_start);
    cudaFree(L);
    cudaFree(hash_array);

    return GPU_clean_time;
}

