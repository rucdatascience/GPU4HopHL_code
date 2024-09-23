#ifndef GLOBAL_LABELS_V2_CUH
#define GLOBAL_LABELS_V2_CUH

#include "definition/hub_def.h"
#include "label/hop_constrained_two_hop_labels_v2.cuh"
#include "memoryManagement/cuda_hashtable_v2.cuh"
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

    cuda_hashTable_v2<weight_type> *L_hash;
    cuda_hashTable_v2<weight_type> *D_hash;

    int *D_vector;
    int *LT_push_back;
    // int *T_push_back;

    /*hop bounded*/
    int thread_num = 1;
    int hop_cst = 0;
    int use_d_optimization = 0;

    /*running time records*/
	double time_initialization = 0;
	double time_generate_labels = 0;
    double time_traverse_labels = 0;
	double time_total = 0;
    double label_size = 0;

    size_t L_size;

    hop_constrained_case_info_v2() {}

    // ���캯��
    // mmpool_size_block ����һ��Ҫ���Ԫ�ظ�����/ nodes_per_block ��Ϊ��Ҫ�� block ��
    __host__ void init (int V, long long mmpool_size_block, int hop_cst, int G_max, int thread_num) {
        cudaError_t err;
        L_size = V;

        // ���������ڴ��
        // ��һ���ڴ�������� label
        cudaMallocManaged(&mmpool_labels, sizeof(mmpool_v2<hub_type>));
        cudaDeviceSynchronize();
        new (mmpool_labels) mmpool_v2<hub_type> (V, (long long) G_max * V * (hop_cst + 1) / 2 / nodes_per_block);
        printf("init 1!\n");

        // �ڶ����ڴ�������� T0
        cudaMallocManaged(&mmpool_T0, sizeof(mmpool_v2<T_item>));
        cudaDeviceSynchronize();
        new (mmpool_T0) mmpool_v2<T_item> (V, (long long) G_max * V * (hop_cst + 1) / 2 / nodes_per_block);
        printf("init 2!\n");

        // �������ڴ�������� T1
        cudaMallocManaged(&mmpool_T1, sizeof(mmpool_v2<T_item>));
        cudaDeviceSynchronize();
        new (mmpool_T1) mmpool_v2<T_item> (V, (long long) G_max * V * (hop_cst + 1) / 2 / nodes_per_block);
        cudaDeviceSynchronize();
        printf("init 3!\n");

        // ���� L_cuda �ڴ��
        cudaMallocManaged(&L_cuda, (long long) V * sizeof(cuda_vector_v2<hub_type>)); // ���� n ��cuda_vectorָ��
        cudaDeviceSynchronize();
        for (int i = 0; i < V; i++) {
            new (L_cuda + i) cuda_vector_v2<hub_type> (mmpool_labels, i, V / nodes_per_block + 1); // ���ù��캯��
        }
        printf("init 4!\n");

        // ���� T0 �ڴ��
        cudaMallocManaged(&T0, (long long) V * sizeof(cuda_vector_v2<T_item>)); // ���� n ��cuda_vectorָ��
        cudaDeviceSynchronize();
        for (int i = 0; i < V; i++) {
            new (T0 + i) cuda_vector_v2<T_item> (mmpool_T0, i, V / nodes_per_block + 1); // ���ù��캯��
        }
        printf("init 5!\n");

        // ���� T1 �ڴ��
        cudaMallocManaged(&T1, (long long) V * sizeof(cuda_vector_v2<T_item>)); // ���� n ��cuda_vectorָ��
        cudaDeviceSynchronize();
        for (int i = 0; i < V; i++) {
            new (T1 + i) cuda_vector_v2<T_item> (mmpool_T1, i, V / nodes_per_block + 1); // ���ù��캯��
        }
        printf("init 6!\n");

        // ׼�� L_hash
        cudaMallocManaged(&L_hash, (long long) thread_num * sizeof(cuda_hashTable_v2<weight_type>));
        cudaDeviceSynchronize();
        for (int i = 0; i < thread_num; i++) {
            new (L_hash + i) cuda_hashTable_v2 <weight_type> (G_max * (hop_cst + 1));
        }
        cudaDeviceSynchronize();
        printf("init 7!\n");

        // ׼�� D_hashTable
        cudaMallocManaged(&D_hash, (long long) thread_num * sizeof(cuda_hashTable_v2<weight_type>));
        cudaDeviceSynchronize();
        for (int i = 0; i < thread_num; i++) {
            new (D_hash + i) cuda_hashTable_v2 <int> (V);
        }
        cudaDeviceSynchronize();
        printf("init 8!\n");
        
        // ׼�� D_vector
        cudaMallocManaged(&D_vector, (long long) thread_num * V * sizeof(int));

        // ׼�� LT_push_back_table
        cudaMallocManaged(&LT_push_back, (long long) thread_num * V * sizeof(int));
        printf("init 9!\n");
        // ׼�� T_push_back_table
        // cudaMallocManaged(&T_push_back, thread_num * V * sizeof(int));
        
        // ͬ������֤���� malloc ��ɡ�
        cudaDeviceSynchronize();

        err = cudaGetLastError(); // ����ں��ڴ��������
        if (err != cudaSuccess) {
            printf("init cuda error !: %s\n", cudaGetErrorString(err));
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