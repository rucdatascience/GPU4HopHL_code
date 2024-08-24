#ifndef GLOBAL_LABELS_V2_CUH
#define GLOBAL_LABELS_V2_CUH

#include "definition/hub_def.h"
#include "label/hop_constrained_two_hop_labels.cuh"
#include "memoryManagement/cuda_vector_v2.cuh"
#include "memoryManagement/mmpool_v2.cuh"
#include <cuda_runtime.h>

class hop_constrained_case_info_v2 {
public:
    /*labels*/
    mmpool_v2<hub_type> *mmpool_labels;
    mmpool_v2<hub_type> *mmpool_T0;
    mmpool_v2<hub_type> *mmpool_T1;

    cuda_vector_v2<hub_type> *L_cuda; // gpu res
    cuda_vector_v2<hub_type> *T0; // T0
    cuda_vector_v2<hub_type> *T1; // T1

    vector<vector<hub_type>> L_cpu; // cpu res

    size_t L_size;

    // ���캯��
    // mmpool_size_block ����һ��Ҫ���Ԫ�ظ�����/ nodes_per_block ��Ϊ��Ҫ�� block ��
    __host__ void init (const int vertex_nums, const int mmpool_size_block, const int hop_cst) {
        L_size = vertex_nums;

        // ���������ڴ��
        // ��һ���ڴ�������� label
        cudaMallocManaged(&mmpool_labels, sizeof(mmpool_v2<hub_type>));
        new (mmpool_labels) mmpool_v2<hub_type>(vertex_nums, mmpool_size_block / nodes_per_block);
        // �ڶ����ڴ�������� T0
        cudaMallocManaged(&mmpool_T0, sizeof(mmpool_v2<hub_type>));
        new (mmpool_T0) mmpool_v2<hub_type>(vertex_nums, mmpool_size_block / nodes_per_block);
        // �������ڴ�������� T1
        cudaMallocManaged(&mmpool_T1, sizeof(mmpool_v2<hub_type>));
        new (mmpool_T1) mmpool_v2<hub_type>(vertex_nums, mmpool_size_block / nodes_per_block);
        cudaDeviceSynchronize();
        
        // ���� L_cuda �ڴ��
        cudaMallocManaged(&L_cuda, vertex_nums * sizeof(cuda_vector_v2<hub_type>)); // ����n��cuda_vectorָ��
        for (int i = 0; i < vertex_nums; i++) {
            new (L_cuda + i) cuda_vector_v2<hub_type>(mmpool_labels, i, (hop_cst * vertex_nums) / nodes_per_block + 1); // ���ù��캯��
        }
        // ���� T0 �ڴ��
        cudaMallocManaged(&T0, vertex_nums * sizeof(cuda_vector_v2<hub_type>)); // ����n��cuda_vectorָ��
        for (int i = 0; i < vertex_nums; i++) {
            new (T0 + i) cuda_vector_v2<hub_type>(mmpool_T0, i, (hop_cst * vertex_nums) / nodes_per_block + 1); // ���ù��캯��
        }
        // ���� T1 �ڴ��
        cudaMallocManaged(&T1, vertex_nums * sizeof(cuda_vector_v2<hub_type>)); // ����n��cuda_vectorָ��
        for (int i = 0; i < vertex_nums; i++) {
            new (T1 + i) cuda_vector_v2<hub_type>(mmpool_T1, i, (hop_cst * vertex_nums) / nodes_per_block + 1); // ���ù��캯��
        }
        cudaDeviceSynchronize();
    }

    // label �еĵ���
    inline size_t cuda_vector_size() {
        return L_size;
    }
    
    // ��������
    __host__ void destroy_L_cuda() {
        for (int i = 0; i < L_size; i++) {
            L_cuda[i].~cuda_vector_v2<hub_type>();
            T0[i].~cuda_vector_v2<hub_type>();
            T1[i].~cuda_vector_v2<hub_type>();
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