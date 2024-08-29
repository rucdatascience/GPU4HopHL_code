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

    // ���캯��
    // mmpool_size_block ����һ��Ҫ���Ԫ�ظ�����/ nodes_per_block ��Ϊ��Ҫ�� block ��
    __host__ void init (int vertex_nums, int mmpool_size_block, int hop_cst) {
        L_size = vertex_nums;

        // ���������ڴ��
        // ��һ���ڴ�������� label
        cudaMallocManaged(&mmpool_labels, sizeof(mmpool_v2<hub_type>));
        cudaDeviceSynchronize();
        new (mmpool_labels) mmpool_v2<hub_type> (vertex_nums, mmpool_size_block / nodes_per_block);
        // �ڶ����ڴ�������� T0
        cudaMallocManaged(&mmpool_T0, sizeof(mmpool_v2<T_item>));
        cudaDeviceSynchronize();
        new (mmpool_T0) mmpool_v2<T_item> (vertex_nums, mmpool_size_block / nodes_per_block);
        // �������ڴ�������� T1
        cudaMallocManaged(&mmpool_T1, sizeof(mmpool_v2<T_item>));
        cudaDeviceSynchronize();
        new (mmpool_T1) mmpool_v2<T_item> (vertex_nums, mmpool_size_block / nodes_per_block);
        cudaDeviceSynchronize();
        
        // ���� L_cuda �ڴ��
        cudaMallocManaged(&L_cuda, vertex_nums * sizeof(cuda_vector_v2<hub_type>)); // ���� n ��cuda_vectorָ��
        cudaDeviceSynchronize();
        for (int i = 0; i < vertex_nums; i++) {
            new (L_cuda + i) cuda_vector_v2<hub_type>(mmpool_labels, i, (hop_cst * vertex_nums) / nodes_per_block + 1); // ���ù��캯��
        }
        // ���� T0 �ڴ��
        cudaMallocManaged(&T0, vertex_nums * sizeof(cuda_vector_v2<T_item>)); // ���� n ��cuda_vectorָ��
        cudaDeviceSynchronize();
        for (int i = 0; i < vertex_nums; i++) {
            new (T0 + i) cuda_vector_v2<T_item>(mmpool_T0, i, (hop_cst * vertex_nums) / nodes_per_block + 1); // ���ù��캯��
        }
        // ���� T1 �ڴ��
        cudaMallocManaged(&T1, vertex_nums * sizeof(cuda_vector_v2<T_item>)); // ���� n ��cuda_vectorָ��
        cudaDeviceSynchronize();
        for (int i = 0; i < vertex_nums; i++) {
            new (T1 + i) cuda_vector_v2<T_item>(mmpool_T1, i, (hop_cst * vertex_nums) / nodes_per_block + 1); // ���ù��캯��
        }
    }

    // label �еĵ���
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

    // ��������
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