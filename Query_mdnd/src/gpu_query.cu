#include <iostream>
#include <cuda_runtime.h>
#include <HBPLL/gpu_query.cuh>

#define THREADS_PER_BLOCK 32

__global__ void get_dis (int s, int t, int h, int label_size, int *dis, label *L, long long *L_start) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= label_size) {
        return;
    }
    int x = tid;
    int v = L[L_start[s] + x].v;
    int l = L_start[t], r = L_start[t + 1] - 1, mid;
    while (l < r) {
        mid = (l + r) >> 1;
        if (L[mid].v >= v) {
            r = mid;
        } else {
            l = mid + 1;
        }
    }
    dis[tid] = INT_MAX;
    for (int i = r; i < L_start[t + 1]; i++) {
        if (L[i].v == v) {
            if (L[x].h + L[i].h <= h) {
                dis[tid] = min(dis[tid], L[x].d + L[i].d);
            }
        }else{
            break;
        }
    }

    return;
}

__global__ void cuda_query (int s, int t, int h, int label_size, int *dis, int *part_dis, label *L, long long *L_start) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_thread_num = gridDim.x * blockDim.x;

    if (tid >= label_size) {
        return;
    }

    int mn_dis = INT_MAX;
    for (int index = tid; index < label_size; index += total_thread_num) {
        mn_dis = min(mn_dis, dis[tid + index]);  //规约求和
    }

    extern __shared__ int shm[];
    shm[threadIdx.x] = mn_dis;
    __syncthreads();

    for (int  active_thread_num = blockDim.x / 2; active_thread_num > 32; active_thread_num /= 2) {
        if (threadIdx.x < active_thread_num) {
            shm[threadIdx.x] = min(shm[threadIdx.x], shm[threadIdx.x + active_thread_num]);
        }
        __syncthreads();
    }

    if (threadIdx.x < 32) {
        volatile int* vshm = shm;
        if (blockDim.x >= 64) {
            vshm[threadIdx.x] = min(vshm[threadIdx.x], vshm[threadIdx.x + 32]);
        }
        vshm[threadIdx.x] = min(vshm[threadIdx.x], vshm[threadIdx.x + 16]);
        vshm[threadIdx.x] = min(vshm[threadIdx.x], vshm[threadIdx.x + 8]);
        vshm[threadIdx.x] = min(vshm[threadIdx.x], vshm[threadIdx.x + 4]);
        vshm[threadIdx.x] = min(vshm[threadIdx.x], vshm[threadIdx.x + 2]);
        vshm[threadIdx.x] = min(vshm[threadIdx.x], vshm[threadIdx.x + 1]);
        if (threadIdx.x == 0) {
            part_dis[blockIdx.x] = vshm[0];
        }
    }
}

double gpu_query(graph_v_of_v<int> &input_graph, vector<vector<label>> &input_L, int query_num, query_info* que, int *ans_gpu, int K)
{

    int V = input_graph.size();

    vector<label> L_flat;
    long long *L_start = nullptr;

    cudaMallocManaged(&L_start, (V + 1) * sizeof(long long));
    cudaDeviceSynchronize();

    long long point = 0;
    long long max_label_size = 0;
    for (int i = 0; i < V; i++) {
        L_start[i] = point;
        long long _size = input_L[i].size();
        max_label_size = max(max_label_size, _size);
        for (int j = 0; j < _size; j++) {
            L_flat.push_back(input_L[i][j]);
        }
        point += _size;
    }
    L_start[V] = point;

    // cout << max_label_size << endl;
    // int *node_id = nullptr, cnt = 0;
    // cudaMallocManaged(&node_id, L_start[V] * sizeof(int));
    // cudaDeviceSynchronize();
    // for (int i = 0; i < V; i++) {
    //     int _size = input_L[i].size();
    //     for (int j = 0; j < _size; j++) {
    //         node_id[cnt ++] = i;
    //     }
    // }
    int *dis;
    cudaMallocManaged(&dis, max_label_size * max_label_size * sizeof(int));
    cudaDeviceSynchronize();

    int *part_dis;
    cudaMallocManaged(&part_dis, max_label_size * max_label_size * sizeof(int));
    cudaDeviceSynchronize();
    
    label *L = nullptr;
    cudaMallocManaged(&L, L_flat.size() * sizeof(label));
    cudaDeviceSynchronize();

    cudaMemcpy(L, L_flat.data(), L_flat.size() * sizeof(label), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Error in cudaMallocManaged" << std::endl;
        std::cerr << cudaGetErrorString(error) << std::endl;
    }

    int dimGrid, dimBlock = 32;
    dimGrid = (max_label_size * max_label_size + dimBlock - 1) / dimBlock;

    cudaEvent_t start, stop;
    float milliseconds = 0;
    double GPU_query_time = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < query_num; i++) {
        // cout << i << endl;
        int qs = que[i].s, qt = que[i].t, qh = que[i].h;
        
        int label_size = (L_start[qs + 1] - L_start[qs]);
        dimGrid = (label_size + dimBlock - 1) / dimBlock;

        get_dis <<< dimGrid, dimBlock >>> (qs, qt, qh, label_size, dis, L, L_start);
        cudaDeviceSynchronize();

        // cuda_query <<< dimGrid, dimBlock >>> (qs, qt, qh, label_size, dis, part_dis, L, L_start);
        // 
        
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    GPU_query_time = (milliseconds / 1000.0);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Error in query" << std::endl;
        std::cerr << cudaGetErrorString(error) << std::endl;
    }

    return GPU_query_time;
}
