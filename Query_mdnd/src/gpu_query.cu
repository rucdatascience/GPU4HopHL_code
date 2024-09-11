#include <iostream>
#include <cuda_runtime.h>
#include <HBPLL/gpu_query.cuh>

__global__ void get_dis (int s, int t, int h, int label_size, int *dis, label *L, long long *L_start) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= label_size) {
        return;
    }

    int x = L_start[s] + tid;
    int v = L[x].v;
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
    for (int i = l; i < L_start[t + 1]; ++i) {
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

__global__ void reduction_kernel1 (int *out, int *in, size_t N) {
    extern __shared__ int sPartials[];
    int mn = INT_MAX;
    const int tid = threadIdx.x;
    for (size_t i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x) {
        mn = min(mn, in[i]);
    }
    sPartials[tid] = mn;
    __syncthreads();
    for (int activeTrheads = blockDim.x >> 1; activeTrheads > 0; activeTrheads >>= 1) {
        if (tid < activeTrheads) {
            sPartials[tid] = min(sPartials[tid], sPartials[tid + activeTrheads]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        out[blockIdx.x] = sPartials[0];
    }
}

__global__ void reduction_kernel2 (int *in, size_t N, int *ans, int qid) {
    extern __shared__ int sPartials[];
    int mn = INT_MAX;
    const int tid = threadIdx.x;
    for (size_t i = blockIdx.x * blockDim.x + tid; i < N; i += blockDim.x * gridDim.x) {
        mn = min(mn, in[i]);
    }
    sPartials[tid] = mn;
    __syncthreads();
    for (int activeTrheads = blockDim.x >> 1; activeTrheads > 0; activeTrheads >>= 1) {
        if (tid < activeTrheads) {
            sPartials[tid] = min(sPartials[tid], sPartials[tid + activeTrheads]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        ans[qid] = sPartials[0];
    }
}

inline void cuda_dis_reduction (int *dis, int *dis_out, int label_size, int qid, int *gpu_ans, const int numBlocks, int numThreads) {
    unsigned int sharedSize = numThreads * sizeof(int);
    reduction_kernel1 <<< numBlocks, numThreads, sharedSize >>> (dis_out, dis, label_size);
    reduction_kernel2 <<< 1, numBlocks, sharedSize >>> (dis_out, numBlocks, gpu_ans, qid);
}

double gpu_query(graph_v_of_v<int> &input_graph, vector<vector<label>> &input_L, int query_num, query_info* que, int *ans, int K) {

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

    int *ans_gpu;
    cudaMallocManaged(&ans_gpu, query_num * sizeof(int));

    int *dis;
    cudaMallocManaged(&dis, max_label_size * max_label_size * sizeof(int));

    int *dis_out;
    cudaMallocManaged(&dis_out, max_label_size * max_label_size * sizeof(int));
    
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

    int dimGrid, dimBlock = 1024;
    dimGrid = (max_label_size * max_label_size + dimBlock - 1) / dimBlock;

    cudaEvent_t start, stop;
    float milliseconds = 0;
    double GPU_query_time = 0;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for (int i = 0; i < query_num; ++i) {
        // cout << i << endl;
        int qs = que[i].s, qt = que[i].t, qh = que[i].h;
        
        int label_size = (L_start[qs + 1] - L_start[qs]);
        dimGrid = (label_size + dimBlock - 1) / dimBlock;

        get_dis <<< dimGrid, dimBlock >>> (qs, qt, qh, label_size, dis, L, L_start);
        cudaDeviceSynchronize();

        cuda_dis_reduction (dis, dis_out, label_size, i, ans_gpu, dimGrid, dimBlock);
        // cudaDeviceSynchronize();
        // ans_gpu[i] = INT_MAX;
        // for (int j = 0; j < label_size; ++j) {
        //     ans_gpu[i] = min(ans_gpu[i], dis[j]);
        // }
    }
    // cudaDeviceSynchronize();
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    GPU_query_time = (milliseconds / 1000.0);

    cudaMemcpy(ans, ans_gpu, query_num * sizeof(int), cudaMemcpyDeviceToHost);

    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Error in query" << std::endl;
        std::cerr << cudaGetErrorString(error) << std::endl;
    }

    cudaFree(ans_gpu);
    cudaFree(dis);
    cudaFree(dis_out);
    cudaFree(L);
    cudaFree(L_start);
    
    return GPU_query_time;
}