// #ifndef CDLPGPU
// #define CDLPGPU

// #include "cuda_runtime.h"
// #include <cuda_runtime_api.h>
// #include "device_launch_parameters.h"
// #include <graph/csr_graph.hpp>
// #include <thrust/device_vector.h>
// #include <thrust/host_vector.h>
// #include <cub/cub.cuh>
// #include <vector>
// #include <string.h>
// #include <definition/hub_def.h>

// using namespace std;
// #define CD_THREAD_PER_BLOCK 512

// __global__ void Label_init(int *labels, int *all_pointer, int N);
// __global__ void LabelPropagation(int *all_pointer, int *prop_labels, int *labels, int *all_edge, int N);
// __global__ void Get_New_Label(int *all_pointer, int *prop_labels, int *new_labels,int* community_size,  int N);
// void checkCudaError(cudaError_t err, const char* msg);
// void checkDeviceProperties();

// //void CDLP_GPU(graph_structure<double>& graph, CSR_graph<double>& input_graph, std::vector<string>& res, int max_iterations);

// //std::vector<std::pair<std::string, std::string>> Cuda_CDLP(graph_structure<double>& graph, CSR_graph<double>& input_graph, int max_iterations);

// // propagate the label, the label of the neighbor vertex is propagated in parallel
// __global__ void LabelPropagation(int *all_pointer, int *prop_labels, int *labels, int *all_edge, int N)
// {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x; // tid decides process which vertex

//     if (tid >= 0 && tid < N)
//     {
//         for (int c = all_pointer[tid]; c < all_pointer[tid + 1]; c++) // traverse the neighbor of the tid vertex
//         {
//             prop_labels[c] = labels[all_edge[c]]; // record the label of the neighbor vertex
//         }
//     }
// }

// // Initialize all labels at once with GPU.Initially
// // each vertex v is assigned a unique label which matches its identifier.
// __global__ void Label_init(int *labels, int *all_pointer, int N)
// {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x; // tid decides process which vertex

//     if (tid >= 0 && tid < N) // tid decides process which vertex
//     {
//         labels[tid] = tid; // each vertex is initially labeled by itself
//     }
// }

// // each thread is responsible for one vertex
// // every segmentation are sorted
// // count Frequency from the start in the global_space_for_label to the end in the global_space_for_label
// // the new labels are stroed in the new_labels
// __global__ void Get_New_Label(int *all_pointer, int *prop_labels, int *new_labels, int* community_size, int N, int MAX_GROUP_SIZE)
// {
//     // Use GPU to propagate all labels at the same time.
//     int tid = blockDim.x * blockIdx.x + threadIdx.x; // tid decides process which vertex
//     if (tid >= 0 && tid < N) {
//         int maxlabel = prop_labels[all_pointer[tid]], maxcount = 0; // the label that appears the most times and its number of occurrences
//         atomicAdd(&community_size[maxlabel], 1);
//         for (int c = all_pointer[tid], last_label = prop_labels[all_pointer[tid]], last_count = 0; c < all_pointer[tid + 1]; c++) // traverse the neighbor vertex label data in order
//         {
//             if (prop_labels[c] == last_label)
//             {
//                 last_count ++; // add up the number of label occurrences
//                 if (last_count > maxcount && atomicAdd(&community_size[last_label], 1) < MAX_GROUP_SIZE) // the number of label occurrences currently traversed is greater than the recorded value
//                 {
//                     atomicAdd(&community_size[maxlabel], -1);
//                     maxcount = last_count; // update maxcount and maxlabel
//                     maxlabel = last_label;
//                 }
//             }
//             else
//             {
//                 last_label = prop_labels[c]; // a new label appears, updates the label and number of occurrences
//                 last_count = 1;
//             }
//         }
//         // 检查选择的标签对应的社区大小是否已达最大值
//         //atomicAdd(&community_size[maxlabel], 1); // 如果没有超过限制，则增加该标签对应社区的大小
//         new_labels[tid] = maxlabel; // 记录maxlabel
        
//     }
// }


// // Community Detection Using Label Propagation on GPU
// // Returns label of the graph based on the graph and number of iterations.
// void CDLP_GPU(int N, CSR_graph<int>& input_graph, std::vector<int>& res, int MAX_GROUP_SIZE, int max_iterations = 100000)
// {
//     //int N = graph.size(); // number of vertices in the graph
//     dim3 init_label_block((N + CD_THREAD_PER_BLOCK - 1) / CD_THREAD_PER_BLOCK, 1, 1); // the number of blocks used in the gpu
//     dim3 init_label_thread(CD_THREAD_PER_BLOCK, 1, 1); // the number of threads used in the gpu

//     int* all_edge = input_graph.all_edge; // graph stored in csr format
//     int* all_pointer = input_graph.all_pointer;

//     int* prop_labels = nullptr;
//     int* new_prop_labels = nullptr;
//     int* new_labels = nullptr;
//     int* labels = nullptr;

//     int CD_ITERATION = max_iterations; // fixed number of iterations
//     long long E = input_graph.E_all; // number of edges in the graph

//     printf("N, E: %d, %d\n", N, E);

//     int *community_size;
//     cudaMallocManaged((void**)&community_size, N * sizeof(int));
//     cudaMemset(community_size, 0, N * sizeof(int));

//     cudaMallocManaged((void**)&new_labels, N * sizeof(int));
//     cudaMallocManaged((void**)&labels, N * sizeof(int));
//     cudaMallocManaged((void**)&prop_labels, E * sizeof(int));
//     cudaMallocManaged((void**)&new_prop_labels, E * sizeof(int));
//     cudaDeviceSynchronize(); // synchronize, ensure the cudaMalloc is complete
    
//     cudaError_t cuda_status = cudaGetLastError();
//     if (cuda_status != cudaSuccess) { // use the cudaGetLastError to check for possible cudaMalloc errors
//         fprintf(stderr, "Cuda malloc failed: %s\n", cudaGetErrorString(cuda_status));
//         return;
//     }

//     Label_init <<<init_label_block, init_label_thread>>> (labels, all_pointer, N); // initialize all labels at once with GPU

//     cudaDeviceSynchronize(); // synchronize, ensure the label initialization is complete
//     cuda_status = cudaGetLastError();
//     if (cuda_status != cudaSuccess) // use the cudaGetLastError to check for possible label initialization errors
//     {
//         fprintf(stderr, "Label init failed: %s\n", cudaGetErrorString(cuda_status));
//         return;
//     }

//     int it = 0; // number of iterations
//     // Determine temporary device storage requirements
//     void *d_temp_storage = NULL;
//     size_t temp_storage_bytes = 0;
//     cub::DeviceSegmentedSort::SortKeys(
//         d_temp_storage, temp_storage_bytes, prop_labels, new_prop_labels,
//         E, N, all_pointer, all_pointer + 1); // sort the labels of each vertex's neighbors

//     cudaDeviceSynchronize();
//     cuda_status = cudaGetLastError();
//     if (cuda_status != cudaSuccess)
//     {
//         fprintf(stderr, "Sort failed: %s\n", cudaGetErrorString(cuda_status));
//         return;
//     }

//     cudaError_t err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
//     if (err != cudaSuccess)
//     {
//         cerr << "Error: " << "Malloc failed" << " (" << cudaGetErrorString(err) << ")" << endl;
//         return;
//     }

//     while (it < CD_ITERATION) // continue for a fixed number of iterations
//     {
//         LabelPropagation<<<init_label_block, init_label_thread>>>(all_pointer, prop_labels, labels, all_edge, N); // calculate the neighbor label array for each vertex
//         cudaDeviceSynchronize();  // synchronize, ensure the label propagation is complete

//         cuda_status = cudaGetLastError(); // check for errors
//         if (cuda_status != cudaSuccess) {
//             fprintf(stderr, "LabelPropagation failed: %s\n", cudaGetErrorString(cuda_status));
//             return;
//         }

//         // Run sorting operation
//         cub::DeviceSegmentedSort::SortKeys(
//             d_temp_storage, temp_storage_bytes, prop_labels, new_prop_labels,
//             E, N, all_pointer, all_pointer + 1); // sort the labels of each vertex's neighbors
//         cudaDeviceSynchronize();

//         cuda_status = cudaGetLastError(); // check for errors
//         if (cuda_status != cudaSuccess) {
//             fprintf(stderr, "Sort failed: %s\n", cudaGetErrorString(cuda_status));
//             return;
//         }

//         cudaMemset(community_size, 0, N * sizeof(int));  // 每次迭代重置社区大小
//         Get_New_Label<<<init_label_block, init_label_thread>>>(all_pointer, new_prop_labels, new_labels, community_size, N, MAX_GROUP_SIZE); // generate a new vertex label by label propagation information
//         cudaDeviceSynchronize();

//         cuda_status = cudaGetLastError(); // check for errors
//         if (cuda_status != cudaSuccess) {
//             fprintf(stderr, "Get_New_Label failed: %s\n", cudaGetErrorString(cuda_status));
//             return;
//         }

//         it++; // record number of iterations
//         std::swap(labels, new_labels); // store the updated label in the labels
//     }
//     cudaFree(prop_labels); // free memory
//     cudaFree(new_prop_labels);
//     cudaFree(new_labels);
//     cudaFree(d_temp_storage);

//     res.resize(N);

//     for (int i = 0; i < N; i++)
//     {
//         res[i] = labels[i]; // convert the label to string and store it in res
//     }

//     cudaFree(labels);
//     cudaFree(community_size);

// }

// // check whether cuda errors occur and output error information
// void checkCudaError(cudaError_t err, const char *msg)
// {
//     if (err != cudaSuccess)
//     {
//         cerr << "Error: " << msg << " (" << cudaGetErrorString(err) << ")" << endl; // output error message
//         exit(EXIT_FAILURE);
//     }
// }

// // Community Detection Using Label Propagation on GPU
// // Returns label of the graph based on the graph and number of iterations.
// // the type of the vertex and label are string
// // std::vector<std::pair<std::string, std::string>> Cuda_CDLP(graph_structure<double>& graph, CSR_graph<double>& input_graph, int max_iterations) {
// //     std::vector<std::string> result;
// //     CDLP_GPU(graph, input_graph, result, max_iterations); // get the labels of each vertex. vector index is the id of vertex

// //     std::vector<std::pair<std::string, std::string>> res;
// //     int size = result.size();
// //     for (int i = 0; i < size; i++)
// //         res.push_back(std::make_pair(graph.vertex_id_to_str[i].first, result[i])); // for each vertex, get its string number and store it in res
    
// //     return res; // return the results
// // }

// #endif

#ifndef CDLPGPU
#define CDLPGPU

#include "cuda_runtime.h"
#include <cuda_runtime_api.h>
#include "device_launch_parameters.h"
#include <graph/csr_graph.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cub/cub.cuh>
#include <vector>
#include <string.h>
#include <definition/hub_def.h>

using namespace std;
#define CD_THREAD_PER_BLOCK 512

__global__ void Label_init(int *labels, int *all_pointer, int N);
__global__ void LabelPropagation(int *all_pointer, int *prop_labels, int *labels, int *all_edge, int N);
__global__ void Get_New_Label(int *all_pointer, int *prop_labels, int *new_labels,int* community_size,  int N);
void checkCudaError(cudaError_t err, const char* msg);
void checkDeviceProperties();

//void CDLP_GPU(graph_structure<double>& graph, CSR_graph<double>& input_graph, std::vector<string>& res, int max_iterations);

//std::vector<std::pair<std::string, std::string>> Cuda_CDLP(graph_structure<double>& graph, CSR_graph<double>& input_graph, int max_iterations);

// propagate the label, the label of the neighbor vertex is propagated in parallel
__global__ void LabelPropagation(int *all_pointer, int *prop_labels, int *labels, int *all_edge, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // tid decides process which vertex

    if (tid >= 0 && tid < N)
    {
        for (int c = all_pointer[tid]; c < all_pointer[tid + 1]; c++) // traverse the neighbor of the tid vertex
        {
            prop_labels[c] = labels[all_edge[c]]; // record the label of the neighbor vertex
        }
    }
}

// Initialize all labels at once with GPU.Initially
// each vertex v is assigned a unique label which matches its identifier.
__global__ void Label_init(int *labels, int *all_pointer, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x; // tid decides process which vertex

    if (tid >= 0 && tid < N) // tid decides process which vertex
    {
        labels[tid] = tid; // each vertex is initially labeled by itself
    }
}

// each thread is responsible for one vertex
// every segmentation are sorted
// count Frequency from the start in the global_space_for_label to the end in the global_space_for_label
// the new labels are stroed in the new_labels
// __global__ void Get_New_Label(int *all_pointer, int *prop_labels, int *new_labels, int* community_size, int N, int MAX_GROUP_SIZE) {
//     // Use GPU to propagate all labels at the same time.
//     int tid = blockDim.x * blockIdx.x + threadIdx.x; // tid decides process which vertex
//     if (tid >= 0 && tid < N) {
//         int maxlabel = prop_labels[all_pointer[tid]], maxcount = 0; // the label that appears the most times and its number of occurrences
//         atomicAdd(&community_size[maxlabel], 1);
//         for (int c = all_pointer[tid], last_label = prop_labels[all_pointer[tid]], last_count = 0; c < all_pointer[tid + 1]; c++) // traverse the neighbor vertex label data in order
//         {
//             if (prop_labels[c] == last_label) {
//                 last_count ++; // add up the number of label occurrences
//                 int x = atomicAdd(&community_size[last_label], 1);
//                 x ++;
//                 if (last_count > maxcount && x < MAX_GROUP_SIZE) { // the number of label occurrences currently traversed is greater than the recorded value
//                     atomicAdd(&community_size[maxlabel], -1);
//                     maxcount = last_count; // update maxcount and maxlabel
//                     maxlabel = last_label;
//                 } else {
//                     atomicAdd(&community_size[last_label], -1);
//                 }
//             } else {
//                 last_label = prop_labels[c]; // a new label appears, updates the label and number of occurrences
//                 last_count = 1;
//             }
//         }
//         // 检查选择的标签对应的社区大小是否已达最大值
//         //atomicAdd(&community_size[maxlabel], 1); // 如果没有超过限制，则增加该标签对应社区的大小
//         new_labels[tid] = maxlabel; // 记录maxlabel
        
//     }
// }
__global__ void Get_New_Label(int *all_pointer, int *prop_labels, int *new_labels, int* community_size, int N, int MAX_GROUP_SIZE)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x; // 当前线程处理的顶点索引
    if (tid < N) {
        int start = all_pointer[tid];
        int end = all_pointer[tid + 1];
        int assigned_label = -1;

        for (int c = start; c < end; ) {
            int label = prop_labels[c];
            int count = 1;
            c++;
            // 统计连续相同的标签出现次数
            while (c < end && prop_labels[c] == label) {
                count++;
                c++;
            }

            // 尝试原子性地增加社区大小
            bool assigned = false;
            while (!assigned) {
                int current_size = atomicAdd(&community_size[label], 0);
                if (current_size >= MAX_GROUP_SIZE) {
                    // 社区已满，无法分配该标签
                    break;
                }
                int result = atomicCAS(&community_size[label], current_size, current_size + 1);
                if (result == current_size) {
                    // 成功增加社区大小，分配该标签
                    assigned_label = label;
                    assigned = true;
                    break;
                } else {
                    // 其他线程更新了社区大小，重试
                }
            }
            if (assigned) {
                break; // 已成功分配标签，退出循环
            }
            // 否则，继续尝试下一个标签
        }

        if (assigned_label == -1) {
            // 无法分配任何邻居标签，尝试分配自己的标签
            assigned_label = tid;
            bool assigned = false;
            while (!assigned) {
                int current_size = atomicAdd(&community_size[assigned_label], 0);
                if (current_size >= MAX_GROUP_SIZE) {
                    // 自己的社区也已满，可以选择特殊处理方式
                    // 这里选择忽略社区大小限制，或者您可以选择其他策略
                    assigned = true;
                } else {
                    int result = atomicCAS(&community_size[assigned_label], current_size, current_size + 1);
                    if (result == current_size) {
                        assigned = true;
                    } else {
                        // 其他线程更新了社区大小，重试
                    }
                }
            }
        }
        // 记录最终分配的标签
        new_labels[tid] = assigned_label;
    }
}

// Community Detection Using Label Propagation on GPU
// Returns label of the graph based on the graph and number of iterations.
void CDLP_GPU(int N, CSR_graph<int>& input_graph, std::vector<int>& res, int MAX_GROUP_SIZE, int max_iterations = 100000)
{
    //int N = graph.size(); // number of vertices in the graph
    dim3 init_label_block((N + CD_THREAD_PER_BLOCK - 1) / CD_THREAD_PER_BLOCK, 1, 1); // the number of blocks used in the gpu
    dim3 init_label_thread(CD_THREAD_PER_BLOCK, 1, 1); // the number of threads used in the gpu

    int* all_edge = input_graph.all_edge; // graph stored in csr format
    int* all_pointer = input_graph.all_pointer;

    int* prop_labels = nullptr;
    int* new_prop_labels = nullptr;
    int* new_labels = nullptr;
    int* labels = nullptr;

    int CD_ITERATION = max_iterations; // fixed number of iterations
    long long E = input_graph.E_all; // number of edges in the graph

    int *community_size;
    cudaMallocManaged((void**)&community_size, N * sizeof(int));
    cudaMemset(community_size, 0, N * sizeof(int));

    cudaMallocManaged((void**)&new_labels, N * sizeof(int));
    cudaMallocManaged((void**)&labels, N * sizeof(int));
    cudaMallocManaged((void**)&prop_labels, E * sizeof(int));
    cudaMallocManaged((void**)&new_prop_labels, E * sizeof(int));

    cudaDeviceSynchronize(); // synchronize, ensure the cudaMalloc is complete
    cudaError_t cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) // use the cudaGetLastError to check for possible cudaMalloc errors
    {
        fprintf(stderr, "Cuda malloc failed: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    Label_init<<<init_label_block, init_label_thread>>>(labels, all_pointer, N); // initialize all labels at once with GPU

    cudaDeviceSynchronize(); // synchronize, ensure the label initialization is complete
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess) // use the cudaGetLastError to check for possible label initialization errors
    {
        fprintf(stderr, "Label init failed: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    int it = 0; // number of iterations
    // Determine temporary device storage requirements
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
    cub::DeviceSegmentedSort::SortKeys(
        d_temp_storage, temp_storage_bytes, prop_labels, new_prop_labels,
        E, N, all_pointer, all_pointer + 1); // sort the labels of each vertex's neighbors

    cudaDeviceSynchronize();
    cuda_status = cudaGetLastError();
    if (cuda_status != cudaSuccess)
    {
        fprintf(stderr, "Sort failed: %s\n", cudaGetErrorString(cuda_status));
        return;
    }

    cudaError_t err = cudaMalloc(&d_temp_storage, temp_storage_bytes);
    if (err != cudaSuccess)
    {
        cerr << "Error: " << "Malloc failed" << " (" << cudaGetErrorString(err) << ")" << endl;
        return;
    }

    while (it < CD_ITERATION) // continue for a fixed number of iterations
    {
        LabelPropagation<<<init_label_block, init_label_thread>>>(all_pointer, prop_labels, labels, all_edge, N); // calculate the neighbor label array for each vertex
        cudaDeviceSynchronize();  // synchronize, ensure the label propagation is complete

        cuda_status = cudaGetLastError(); // check for errors
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "LabelPropagation failed: %s\n", cudaGetErrorString(cuda_status));
            return;
        }

        // Run sorting operation
        cub::DeviceSegmentedSort::SortKeys(
            d_temp_storage, temp_storage_bytes, prop_labels, new_prop_labels,
            E, N, all_pointer, all_pointer + 1); // sort the labels of each vertex's neighbors
        cudaDeviceSynchronize();

        cuda_status = cudaGetLastError(); // check for errors
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Sort failed: %s\n", cudaGetErrorString(cuda_status));
            return;
        }

        cudaMemset(community_size, 0, N * sizeof(int));  // 每次迭代重置社区大小
        Get_New_Label<<<init_label_block, init_label_thread>>>(all_pointer, new_prop_labels, new_labels, community_size, N, MAX_GROUP_SIZE); // generate a new vertex label by label propagation information
        cudaDeviceSynchronize();

        cuda_status = cudaGetLastError(); // check for errors
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Get_New_Label failed: %s\n", cudaGetErrorString(cuda_status));
            return;
        }

        it++; // record number of iterations
        std::swap(labels, new_labels); // store the updated label in the labels
    }
    cudaFree(prop_labels); // free memory
    cudaFree(new_prop_labels);
    cudaFree(new_labels);
    cudaFree(d_temp_storage);

    res.resize(N);

    for (int i = 0; i < N; i++)
    {
        res[i] = labels[i]; // convert the label to string and store it in res
    }

    cudaFree(labels);
    cudaFree(community_size);

}

// check whether cuda errors occur and output error information
void checkCudaError(cudaError_t err, const char *msg)
{
    if (err != cudaSuccess)
    {
        cerr << "Error: " << msg << " (" << cudaGetErrorString(err) << ")" << endl; // output error message
        exit(EXIT_FAILURE);
    }
}

// Community Detection Using Label Propagation on GPU
// Returns label of the graph based on the graph and number of iterations.
// the type of the vertex and label are string
// std::vector<std::pair<std::string, std::string>> Cuda_CDLP(graph_structure<double>& graph, CSR_graph<double>& input_graph, int max_iterations) {
//     std::vector<std::string> result;
//     CDLP_GPU(graph, input_graph, result, max_iterations); // get the labels of each vertex. vector index is the id of vertex

//     std::vector<std::pair<std::string, std::string>> res;
//     int size = result.size();
//     for (int i = 0; i < size; i++)
//         res.push_back(std::make_pair(graph.vertex_id_to_str[i].first, result[i])); // for each vertex, get its string number and store it in res
    
//     return res; // return the results
// }

#endif