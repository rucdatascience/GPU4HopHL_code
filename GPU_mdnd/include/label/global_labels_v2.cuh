#ifndef GLOBAL_LABELS_V2_CUH
#define GLOBAL_LABELS_V2_CUH

#include "definition/hub_def.h"
#include "label/hop_constrained_two_hop_labels_v2.cuh"
#include "memoryManagement/cuda_vector_v2.cuh"
#include "memoryManagement/mmpool_v2.cuh"
#include <cuda_runtime.h>

class hop_constrained_case_info_v2 {
public:
    /*labels*/
    mmpool_v2<hub_type> *mmpool_labels = NULL;
    mmpool_v2<T_item> *mmpool_T0 = NULL;
    mmpool_v2<T_item> *mmpool_T1 = NULL;

    cuda_vector_v2<hub_type> *L_cuda = NULL; // gpu res
    cuda_vector_v2<T_item> *T0 = NULL; // T0
    cuda_vector_v2<T_item> *T1 = NULL; // T1

    /*hop bounded*/
    int thread_num = 1;
    int upper_k = 0;
    int use_d_optimization = 0;

    /*running time records*/
	double time_initialization = 0;
	double time_generate_labels = 0;
	double time_sortL = 0;
	double time_canonical_repair = 0;
	double time_total = 0;
    double label_size = 0;

    size_t L_size;

    // 构造函数
    // mmpool_size_block 就是一共要存的元素个数，/ nodes_per_block 即为需要的 block 数
    __host__ void init (int vertex_nums, int mmpool_size_block, int hop_cst) {
        L_size = vertex_nums;

        // 创建三个内存池
        // 第一个内存池用来存 label
        cudaMallocManaged(&mmpool_labels, sizeof(mmpool_v2<hub_type>));
        cudaDeviceSynchronize();
        new (mmpool_labels) mmpool_v2<hub_type> (vertex_nums, mmpool_size_block / nodes_per_block);
        // 第二个内存池用来存 T0
        cudaMallocManaged(&mmpool_T0, sizeof(mmpool_v2<T_item>));
        cudaDeviceSynchronize();
        new (mmpool_T0) mmpool_v2<T_item> (vertex_nums, mmpool_size_block / nodes_per_block);
        // 第三个内存池用来存 T1
        cudaMallocManaged(&mmpool_T1, sizeof(mmpool_v2<T_item>));
        cudaDeviceSynchronize();
        new (mmpool_T1) mmpool_v2<T_item> (vertex_nums, mmpool_size_block / nodes_per_block);
        cudaDeviceSynchronize();
        
        // 分配 L_cuda 内存池
        cudaMallocManaged(&L_cuda, vertex_nums * sizeof(cuda_vector_v2<hub_type>)); // 分配 n 个cuda_vector指针
        cudaDeviceSynchronize();
        for (int i = 0; i < vertex_nums; i++) {
            new (L_cuda + i) cuda_vector_v2<hub_type>(mmpool_labels, i, (hop_cst * vertex_nums) / nodes_per_block + 1); // 调用构造函数
        }
        // 分配 T0 内存池
        cudaMallocManaged(&T0, vertex_nums * sizeof(cuda_vector_v2<T_item>)); // 分配 n 个cuda_vector指针
        cudaDeviceSynchronize();
        for (int i = 0; i < vertex_nums; i++) {
            new (T0 + i) cuda_vector_v2<T_item>(mmpool_T0, i, (hop_cst * vertex_nums) / nodes_per_block + 1); // 调用构造函数
        }
        // 分配 T1 内存池
        cudaMallocManaged(&T1, vertex_nums * sizeof(cuda_vector_v2<T_item>)); // 分配 n 个cuda_vector指针
        cudaDeviceSynchronize();
        for (int i = 0; i < vertex_nums; i++) {
            new (T1 + i) cuda_vector_v2<T_item>(mmpool_T1, i, (hop_cst * vertex_nums) / nodes_per_block + 1); // 调用构造函数
        }
    }

    // label 中的点数
    inline size_t cuda_vector_size() {
        return L_size;
    }
    
    // ~hop_constrained_case_info_v2() {
    //     for (int i = 0; i < L_size; i++) {
    //         L_cuda[i].~cuda_vector_v2<hub_type>();
    //         T0[i].~cuda_vector_v2<hub_type>();
    //         T1[i].~cuda_vector_v2<hub_type>();
    //     }
    //     cudaFree(L_cuda);
    //     cudaFree(T0);
    //     cudaFree(T1);

    //     mmpool_labels->~mmpool_v2();
    //     mmpool_T0->~mmpool_v2();
    //     mmpool_T1->~mmpool_v2();
    //     cudaFree(mmpool_labels);
    //     cudaFree(mmpool_T0);
    //     cudaFree(mmpool_T1);
    // }

    // 析构函数
    __host__ void destroy_L_cuda() {
        for (int i = 0; i < L_size; i++) {
            L_cuda[i].~cuda_vector_v2<hub_type>();
            T0[i].~cuda_vector_v2<T_item>();
            T1[i].~cuda_vector_v2<T_item>();
        }
        cudaFree(L_cuda);
        cudaFree(T0);
        cudaFree(T1);

        mmpool_labels->~mmpool_v2();
        mmpool_T0->~mmpool_v2();
        mmpool_T1->~mmpool_v2();
        cudaFree(mmpool_labels);
        cudaFree(mmpool_T0);
        cudaFree(mmpool_T1);
    }

};

#endif