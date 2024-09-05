#include <iostream>
#include <cuda_runtime.h>
#include <HBPLL/gpu_clean.cuh>

//extern __shared__ int shared_hash_array[];

#define THREADS_PER_BLOCK 512

__device__ int query_label(label* L, long long start, long long end, int i, int h_v, int* Lc_hashed, int V, int K) {
    int update_dis = INT_MAX;
    for(long long label_id = start; label_id < end; ++label_id) {
        int v_x = L[label_id].v;
        int h_x = L[label_id].h;
        int d_vvx = L[label_id].d;

        for (int h_y = 0; h_y <= h_v - h_x; h_y++) {
            int new_dis = Lc_hashed[v_x * (K + 1) + h_y];
            new_dis = (new_dis == INT_MAX) ? INT_MAX : new_dis + d_vvx;
            update_dis = (update_dis > new_dis) ? new_dis : update_dis;
        }
    }
    return update_dis;
}

__global__ void clean_kernel(int V, int K, int tc, label* L, long long* L_start, int* hash_array, int* mark) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < 0 || i >= tc)
        return;
    int offset = i*V*(K+1);
    for (int u = i; u < V; u += tc) {
        for (long long label_idx = L_start[u]; label_idx < L_start[u + 1]; label_idx++) {
            int d_uv = INT_MAX;
            int v = L[label_idx].v;
            int h_v = L[label_idx].h;
            int check_duv = L[label_idx].d;
            d_uv = query_label(L, L_start[v], L_start[v + 1], i, h_v, hash_array + offset, V, K);
            
            cudaError_t error = cudaGetLastError();
            if (error != cudaSuccess) {
                printf("CUDA error: %s\n", cudaGetErrorString(error));
                return;
            }

            if (d_uv > check_duv) {
                hash_array[offset + v * (K + 1) + h_v] = check_duv;
            } else {
                mark[label_idx] = 1;
            }
        }

        // 恢复全局 shared_hash_array 数据
        for (long long label_idx = L_start[u]; label_idx < L_start[u + 1]; label_idx++) {
            int v = L[label_idx].v;
            int h_v = L[label_idx].h;
            hash_array[offset + v * (K + 1) + h_v] = INT_MAX;
        }
    }
}


void gpu_clean(graph_v_of_v<int>& input_graph, vector<vector<label>>& input_L,vector<vector<hop_constrained_two_hop_label>>& res, int tc, int K) {
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

    // base_memory<label>* pool = nullptr;
    // cudaMallocManaged(&pool, sizeof(base_memory<label>));
    // new (pool) base_memory<label>(1024, (size_t)V * (K + 1) * 1024);

    // cuda_vector<label>** Lc = nullptr;
    // cudaMallocManaged(&Lc, sizeof(cuda_vector<label>*) * V);
    // cudaDeviceSynchronize();
    // label **Lc = nullptr;
    // cudaMallocManaged(&Lc, sizeof(label*) * V);
    // for (int i = 0; i < V; i++) {
    //     cudaMallocManaged(&Lc[i], sizeof(label)*);
    //     new (Lc[i]) cuda_vector<label>(pool, 1024);
    // }

    int* hash_array = nullptr; // first dim size is V * (K + 1)

    cudaMallocManaged(&hash_array, sizeof(int) * tc * V * (K + 1));
    //cudaDeviceSynchronize();

    for (long long i = 0; i < (long long)tc * V * (K + 1); i++)
        hash_array[i] = INT_MAX;

    // int* d_uv = nullptr;
    // cudaMallocManaged(&d_uv, sizeof(int) * tc);
    // cudaDeviceSynchronize();

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Error in cudaMallocManaged 2" << std::endl;
        std::cerr << cudaGetErrorString(error) << std::endl;
        //return nullptr;
    }

    

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    clean_kernel<<<(tc + THREADS_PER_BLOCK) / THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(V, K, tc, L, L_start, hash_array, mark);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    std::cout << "GPU Clean Time: " << milliseconds << " ms" << std::endl;

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(error) << std::endl;
        //return nullptr;
    }

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("Max shared memory per block: %d bytes\n", deviceProp.sharedMemPerBlock);


    //将L(csr)转为res(vector<vector>)
    for(int i = 0; i < V; ++i)
{
    int start = L_start[i];
    int end = L_start[i+1];
    // int ts = 0;
    // int te = 0;
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



        // if(L[j] != INT_MAX)
        // {
        //     te++;
        // }
        // else
        // {
        //     // 插入范围从 L[j+ts] 开始，插入 te-ts 个元素
        //     res[i].insert(res[i].end(), L + j + ts, L + j + te);
        //     ts = j + 1;
        //     te = j + 1;
        // }
    }
}

    cudaFree(L_start);
    cudaFree(L);
    cudaFree(hash_array);
    //cudaFree(d_uv);

    //return Lc;
}

