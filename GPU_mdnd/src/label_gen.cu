#include "label/gen_label.cuh"

// ͨ�� hashtable �Ŀ��ٲ�ѯ
__device__ int query_dis_by_hash_table (int u, int v, cuda_hashTable_v2<weight_type> *H, cuda_vector_v2<hub_type> *L, int hop_now, int hop_cst) {
    int min_dis = 1e9;

    for (int i = 0; i < L->blocks_num; ++i) {
        int block_id = L->block_idx_array[i];
        int block_siz = L->pool->get_block_size(block_id);
        for (int j = 0; j < block_siz; ++j) {
            hub_type* x = L->pool->get_node(block_id, j);
            for (int k = hop_now - x->hop; k >= 0; --k) {
                min_dis = min(min_dis, x->distance + H->get(x->hub_vertex, k, hop_cst));
            }
        }
    }
    return min_dis;

}

// ��ʼ�� T
__global__ void init_T (int V, cuda_vector_v2<T_item> *T, cuda_vector_v2<hub_type> *L_gpu) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < V) {
        // target_vertex, distance
        T[tid].push_back({tid, 0});
        L_gpu[tid].push_back({tid, 0, 0});
    }
}

// ��� T
__global__ void clear_T (int V, cuda_vector_v2<T_item> *T, cuda_vector_v2<T_item> *D) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < V) {
        T[tid].init(V, tid);
        D[tid].init(V, tid);
    }
}

// �������ɹ���
__global__ void gen_label_hsdl (int V, int thread_num, int hop_cst, int hop_now, int* out_pointer, int* out_edge, int* out_edge_weight,
            cuda_vector_v2<hub_type> *L_gpu, cuda_hashTable_v2<weight_type> *Has, cuda_vector_v2<T_item> *T0, cuda_vector_v2<T_item> *T1) {
    
    // �߳�id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // hash table
    cuda_hashTable_v2<weight_type> *has = (Has + tid);

    for (int node_id = tid; node_id < V; node_id += thread_num) {

        // node_id �� T ����
        cuda_vector_v2<T_item> *t0 = (T0 + node_id);
        cuda_vector_v2<T_item> *t1 = (T1 + node_id);

        // node_id �� label
        cuda_vector_v2<hub_type> *L = (L_gpu + node_id);

        // ��ʼ�� hashtable�����Ǳ��� label ���ϲ�һһ�޸��� hashtable �е�ֵ
        for (int i = 0; i < L->blocks_num; ++i) {
            int block_id = L->block_idx_array[i];
            int block_siz = L->pool->get_block_size(block_id);
            for (int j = 0; j < block_siz; ++j) {
                hub_type* x = L->pool->get_node(block_id, j);
                has->modify(x->hub_vertex, x->hop, hop_cst, x->distance);
            }
        }

        // ���� T ����
        for (int i = 0; i < t0->blocks_num; ++i) {
            int block_id = t0->block_idx_array[i];
            int block_siz = t0->pool->get_block_size(block_id);
            for (int j = 0; j < block_siz; ++j) {

                // ��ȡ T ����Ԫ��
                T_item *x = t0->pool->get_node(block_id, j);

                // sv Ϊ���, ev Ϊ�������ĵ�, dis Ϊ���룬hop Ϊ����
                int sv = node_id, ev = x->vertex, h = hop_now;
                weight_type dis = x->distance;

                // �����ڵ� ev ����չ
                for (int k = out_pointer[ev]; k < out_pointer[ev + 1]; ++k) {
                    int v = out_edge[k];
                    
                    // rank pruning������ͬһ����Ҳ�����㡣
                    if (sv >= v) continue;

                    // h Ϊ������Щ��ǩ�������� h + 1Ϊ����Ҫ��ӵı�ǩ����
                    int dv = dis + out_edge_weight[k];
                    weight_type q_dis = query_dis_by_hash_table(sv, v, Has + tid, L_gpu + v, h + 1, hop_cst);
                    
                    if (dv < q_dis) {
                        // ��ӱ�ǩ��ѹ�� T ����
                        L_gpu[v].push_back({sv, h + 1, dv});
                        t1->push_back({v, dv});
                    }

                }
            }
        }

        // �Ļ� hashtable
        for (int i = 0; i < L->blocks_num; ++i) {
            int block_id = L->block_idx_array[i];
            int block_siz = L->pool->get_block_size(block_id);
            for (int j = 0; j < block_siz; ++j) {
                hub_type* x = L->pool->get_node(block_id, j);
                has->modify(x->hub_vertex, x->hop, hop_cst, 1e9);
            }
        }
    }
}

// �������ɹ���_v2
__global__ void gen_label_hsdl_v2 (int V, int thread_num, int hop_cst, int hop_now, int* out_pointer, int* out_edge, int* out_edge_weight,
            cuda_vector_v2<hub_type> *L_gpu, cuda_hashTable_v2<weight_type> *Has, cuda_hashTable_v2<weight_type> *Das,
            cuda_vector_v2<T_item> *T0, cuda_vector_v2<T_item> *T1, cuda_vector_v2<T_item> *D) {
    
    // �߳�id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // hash table
    cuda_hashTable_v2<weight_type> *has = (Has + tid);
    cuda_hashTable_v2<weight_type> *das = (Das + tid);

    for (int node_id = tid; node_id < V; node_id += thread_num) {

        // node_id �� T ����
        cuda_vector_v2<T_item> *t0 = (T0 + node_id);
        cuda_vector_v2<T_item> *t1 = (T1 + node_id);
        cuda_vector_v2<T_item> *d = (D + node_id);

        // node_id �� label
        cuda_vector_v2<hub_type> *L = (L_gpu + node_id);

        // ��ʼ�� hashtable�����Ǳ��� label ���ϲ�һһ�޸��� hashtable �е�ֵ
        for (int i = 0; i < L->blocks_num; ++i) {
            int block_id = L->block_idx_array[i];
            int block_siz = L->pool->get_block_size(block_id);
            for (int j = 0; j < block_siz; ++j) {
                hub_type* x = L->pool->get_node(block_id, j);
                has->modify(x->hub_vertex, x->hop, hop_cst, x->distance);
            }
        }

        // ���� T ���У������� D ����
        for (int i = 0; i < t0->blocks_num; ++i) {
            int block_id = t0->block_idx_array[i];
            int block_siz = t0->pool->get_block_size(block_id);
            for (int j = 0; j < block_siz; ++j) {

                // ��ȡ T ����Ԫ��
                T_item *x = t0->pool->get_node(block_id, j);

                // sv Ϊ���, ev Ϊ�������ĵ�, dis Ϊ���룬hop Ϊ����
                int sv = node_id, ev = x->vertex, h = hop_now;
                weight_type dis = x->distance;

                // �����ڵ� ev ����չ
                for (int k = out_pointer[ev]; k < out_pointer[ev + 1]; ++k) {
                    int v = out_edge[k];
                    
                    // rank pruning������ͬһ����Ҳ�����㡣
                    if (sv >= v) continue;

                    // h Ϊ������Щ��ǩ�������� h + 1Ϊ����Ҫ��ӵı�ǩ����
                    int dv = dis + out_edge_weight[k];

                    // �ж����� D ����
                    weight_type d_hash = das->get(v);
                    if (d_hash == 1e9) {
                        d->push_back({v, d_hash});
                        das->modify(v, dv);
                    }else{
                        if (d_hash > dv) {
                            das->modify(v, dv);
                        }
                    }

                }
            }
        }

        // ���� D ����
        for (int i = 0; i < d->blocks_num; ++i) {
            int block_id = d->block_idx_array[i];
            int block_siz = d->pool->get_block_size(block_id);
            for (int j = 0; j < block_siz; ++j) {

                // ��ȡ D ����Ԫ��
                T_item *x = d->pool->get_node(block_id, j);

                int sv = node_id, v = x->vertex, h = hop_now;
                weight_type dv = das->get(v);
                weight_type q_dis = query_dis_by_hash_table(sv, v, Has + tid, L_gpu + v, h + 1, hop_cst);
                
                if (dv < q_dis) {
                    // ��ӱ�ǩ��ѹ�� T ����
                    L_gpu[v].push_back({sv, h + 1, dv});
                    t1->push_back({v, dv});
                }

                das->modify(v, 1e9);
            }
        }

        // �Ļ� hashtable
        for (int i = 0; i < L->blocks_num; ++i) {
            int block_id = L->block_idx_array[i];
            int block_siz = L->pool->get_block_size(block_id);
            for (int j = 0; j < block_siz; ++j) {
                hub_type* x = L->pool->get_node(block_id, j);
                has->modify(x->hub_vertex, x->hop, hop_cst, 1e9);
            }
        }
    }
}

// ���� label �Ĺ���
void label_gen (CSR_graph<weight_type>& input_graph, hop_constrained_case_info_v2 *info, int hop_cst, vector<vector<hub_type> >&L) {
    
    int V = input_graph.OUTs_Neighbor_start_pointers.size() - 1;
    int E = input_graph.OUTs_Edges.size();
    int* out_edge = input_graph.out_edge;
    int* out_edge_weight = input_graph.out_edge_weight;
    int* out_pointer = input_graph.out_pointer;

    int dimGrid = 1, dimBlock = 1024;
    dimGrid = (V + dimBlock - 1) / dimBlock;

    int thread_num = 1000;

    printf("V, E: %d, %d\n", V, E);

    // ׼�� info
    info->init(V, V * V * hop_cst, hop_cst);
    printf("init case_info success\n");

    // ׼�� L_hashTable
    cuda_hashTable_v2<weight_type> *L_hash;
    cudaMallocManaged(&L_hash, thread_num * sizeof(cuda_hashTable_v2<weight_type>));
    for (int i = 0; i < thread_num; i++) {
        new (L_hash + i) cuda_hashTable_v2 <weight_type> (V * (hop_cst + 1));
    }

    // ׼�� D_hashTable
    cuda_hashTable_v2<weight_type> *D_hash;
    cudaMallocManaged(&D_hash, thread_num * sizeof(cuda_hashTable_v2<weight_type>));
    for (int i = 0; i < thread_num; i++) {
        new (D_hash + i) cuda_hashTable_v2 <weight_type> (V);
    }
    
    printf("init hash_table success\n");
    
    // ���ԽС�ĵ㣬rank Խ��
    // for (int i = 0; i < V; i ++){
    //     printf("degree %d, %d\n", i, out_pointer[i + 1] - out_pointer[i]);
    // }

    // ͬ������֤���ݳ�ʼ�����
    cudaDeviceSynchronize();

    // ���� cuda_vector �� cuda_hash �Ĳ���
    // test_mmpool(V, thread_num, 3, info, L_hash)
    
    init_T <<<dimGrid, dimBlock>>> (V, info->T0, info->L_cuda);
    cudaDeviceSynchronize();

    // ��ʱ
    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    // ����������������ֱ��̽�� T �Ƿ�Ϊ��
    int iter = 0;

    while (1) {

        if (iter++ >= hop_cst) break;
        // iter = 1 -> ���� ���� 1
        // iter = 2 -> ���� ���� 2
        // iter = 3 -> ���� ���� 3
        // iter = 4 -> ���� ���� 4
        // iter = 5 -> ���� ���� 5
        
        printf("iteration_hop: %d\n", iter);

        // ������ż�ԣ�����ʹ�� T0��T1������Ҫ����ָ��
        if (iter % 2 == 1) {
            // gen_label_hsdl <<<1, thread_num>>> (V, thread_num, hop_cst, iter - 1, out_pointer, out_edge, out_edge_weight,
            // info->L_cuda, L_hash, info->T0, info->T1);
            gen_label_hsdl_v2 <<<1, thread_num>>> (V, thread_num, hop_cst, iter - 1, out_pointer, out_edge, out_edge_weight,
            info->L_cuda, L_hash, D_hash, info->T0, info->T1, info->D);
            cudaDeviceSynchronize();

            // ��ϴ T ����
            clear_T <<<dimGrid, dimBlock>>> (V, info->T0, info->D);
        }else{
            // gen_label_hsdl <<<1, thread_num>>> (V, thread_num, hop_cst, iter - 1, out_pointer, out_edge, out_edge_weight,
            // info->L_cuda, L_hash, info->T1, info->T0);
            gen_label_hsdl_v2 <<<1, thread_num>>> (V, thread_num, hop_cst, iter - 1, out_pointer, out_edge, out_edge_weight,
            info->L_cuda, L_hash, D_hash, info->T1, info->T0, info->D);
            cudaDeviceSynchronize();

            // ��ϴ T ����
            clear_T <<<dimGrid, dimBlock>>> (V, info->T1, info->D);
        }
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("Time generation in hop %d : %.5lf s\n", iter, elapsedTime / 1000.0);

    }
    cudaError_t err;
    err = cudaGetLastError(); // ����ں��ڴ��������
    if (err != cudaSuccess) {
        printf("!INIT CUDA ERROR: %s\n", cudaGetErrorString(err));
    }
    
    // timer record
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time generation: %.5f s\n", elapsedTime / 1000.0);
    info->time_generate_labels = elapsedTime / 1000.0;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("hub, parent, hop, dis:\n");
    int mx_hop = 0;
    for (int v = 0; v < V; ++v) {
        // printf("vertex %d\n", v);
        for (int i = 0; i < info->L_cuda[v].blocks_num; ++i) {
            int block_id = info->L_cuda[v].block_idx_array[i];
            int block_siz = info->L_cuda[v].pool->get_block_size(block_id);
            for (int j = 0; j < block_siz; ++j) {
                hub_type* x = info->L_cuda[v].pool->get_node(block_id, j);
                // printf("{%d, %d, %d, %d}, ", x->hub_vertex, x->parent_vertex, x->hop, x->distance);
                L[v].push_back({x->hub_vertex, x->hop, x->distance});
                info->label_size ++;
                mx_hop = max(mx_hop, x->hop);
            }
        }
        // printf("\n");
    }
    printf("max hop: %d\n", mx_hop);
    info->label_size = info->label_size / (double)V;
    printf("average label size: %.5lf\n", info->label_size);
    printf("Generation end!\n");

    info->destroy_L_cuda();
    
    for(int i = 0; i < thread_num; ++i){
        L_hash[i].~cuda_hashTable_v2();
    }
    cudaFree(L_hash);
    
    return;
}