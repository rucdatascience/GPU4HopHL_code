#include "label/gen_label.cuh"

// ͨ�� hashtable �Ŀ��ٲ�ѯ
__device__ int query_dis_by_hash_table
(int u, int v, cuda_hashTable_v2<weight_type> *H, cuda_vector_v2<hub_type> *L, int hop_now, int hop_cst) {
    int min_dis = 1e9;

    int block_num = L->blocks_num;
    int cnt = 0;
    for (int i = 0; i < block_num; ++i) {
        int block_id = L->block_idx_array[i];
        int block_siz = L->pool->get_block_size(block_id);
        for (int j = 0; j < block_siz; ++j) {
            hub_type *x = &(L->pool->blocks_pool[block_id].data[j]);
            for (int k = hop_now - x->hop; k >= 0; --k) {
                min_dis = min(min_dis, x->distance + H->get(x->hub_vertex, k, hop_cst));
            }
            if (++cnt >= L->last_size) break;
        }
    }
    return min_dis;

}

// ��̬���м��ٲ�ѯ
// u, v, d_size, d_has, d, label, hop, hop_cst;
// (node_id, d->current_size, das, d, has, L_gpu, t1, hop_now, hop_cst)
__global__ void query_parallel (int sv, int st, int ed, int sz, cuda_hashTable_v2<weight_type> *das, int *d,
cuda_hashTable_v2<weight_type> *has, cuda_vector_v2<hub_type> *L_gpu, cuda_vector_v2<T_item> *t1,
int V, int thread_num, int tidd, int* L_push_back, int* T_push_back, int hop_now, int hop_cst) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < 0 || tid >= sz) {
        return;
    }

    // ��ȡ D ����Ԫ��
    int v = d[st + tid];
    weight_type dv = das->get(v);
    weight_type q_dis = query_dis_by_hash_table(sv, v, has, L_gpu + v, hop_now + 1, hop_cst);
    
    if (dv < q_dis) {
        // ���ӱ�ǩ��ѹ�� T ����
        // ��ʾ�� v �� L ��Ҫ push_back һ�� sv �Ĳ��Ҿ���Ϊ dv ��Ԫ�ء�
        L_push_back[v * thread_num + tidd] = dv;

        // ��ʾ�� tidd �� T ��Ҫ push_back һ�� v �Ĳ��Ҿ���Ϊ dv ��Ԫ�ء�
        T_push_back[tidd * V + v] = dv;
        // L_gpu[v].push_back({sv, hop_now + 1, dv});
        // t1->push_back({v, dv});
    }
    // __syncthreads();

    das->modify(v, 1e9);

}

// ͨ�� L_push_back ���뵽 Lable ��
__global__ void Push_Back_L
(int V, int thread_num, int start_id, int end_id, int hop, int* L_push_back, cuda_vector_v2<hub_type> *L_gpu) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 0 || tid >= V) {
        return;
    }
    int st = tid * thread_num;
    for (int i = st; i < st + thread_num; ++i) {
        if (L_push_back[i] != 0) {
            L_gpu[tid].push_back({start_id + i - st, hop, L_push_back[i]});
            L_push_back[i] = 0;
        }
    }
}

// ͨ�� L_push_back ���뵽 Lable ��
__global__ void Push_Back_T
(int V, int thread_num, int start_id, int end_id, int hop, int* T_push_back, cuda_vector_v2<T_item> *T) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 0 || tid >= thread_num) {
        return;
    }
    int st = tid * V;
    for (int i = st; i < st + V; ++i) {
        if (T_push_back[i] != 0) {
            T[start_id + tid].push_back({i - st, T_push_back[i]});
            T_push_back[i] = 0;
        }
    }
}

// ��ʼ�� T
__global__ void init_T (int V, cuda_vector_v2<T_item> *T, cuda_vector_v2<hub_type> *L_gpu) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < V) {
        // target_vertex, distance
        T[tid].push_back({tid, 0});
        L_gpu[tid].push_back({tid, 0, 0});
        L_gpu[tid].last_size = 1;
    }
}

// ��� T
__global__ void clear_T (int V, cuda_vector_v2<T_item> *T, cuda_vector_v2<hub_type> *L_gpu) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < V) {
        T[tid].init(V, tid);
        L_gpu[tid].last_size = L_gpu[tid].current_size;
    }
}

// �������ɹ��̣����صĲ���
__global__ void gen_label_hsdl (int V, int thread_num, int hop_cst, int hop_now, int* out_pointer, int* out_edge, int* out_edge_weight,
            cuda_vector_v2<hub_type> *L_gpu, cuda_hashTable_v2<weight_type> *Has, cuda_vector_v2<T_item> *T0, cuda_vector_v2<T_item> *T1) {
    
    // �߳�id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < 0 || tid >= thread_num) {
        return;
    }

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

                    // h Ϊ������Щ��ǩ�������� h + 1Ϊ����Ҫ���ӵı�ǩ����
                    int dv = dis + out_edge_weight[k];
                    weight_type q_dis = query_dis_by_hash_table(sv, v, Has + tid, L_gpu + v, h + 1, hop_cst);
                    
                    if (dv < q_dis) {
                        // ���ӱ�ǩ��ѹ�� T ����
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

// �������ɹ���_v2�������� D �����Ż���������
__global__ void gen_label_hsdl_v2 (int V, int thread_num, int hop_cst, int hop_now, int* out_pointer, int* out_edge, int* out_edge_weight,
            cuda_vector_v2<hub_type> *L_gpu, cuda_hashTable_v2<weight_type> *Has, cuda_hashTable_v2<weight_type> *Das,
            cuda_vector_v2<T_item> *T0, cuda_vector_v2<T_item> *T1, int *d) {
    
    // �߳�id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < 0 || tid >= thread_num) {
        return;
    }

    // hash table
    cuda_hashTable_v2<weight_type> *has = (Has + tid);
    cuda_hashTable_v2<weight_type> *das = (Das + tid);
    int d_start = tid * V, d_end = d_start;
    int block_id, block_siz;

    for (int node_id = tid; node_id < V; node_id += thread_num) {
        
        // node_id �� T ����
        cuda_vector_v2<T_item> *t0 = (T0 + node_id);
        cuda_vector_v2<T_item> *t1 = (T1 + node_id);

        // node_id �� label
        cuda_vector_v2<hub_type> *L = (L_gpu + node_id);

        // ��ʼ�� hashtable�����Ǳ��� label ���ϲ�һһ�޸��� hashtable �е�ֵ
        int cnt = 0;
        for (int i = 0; i < L->blocks_num; ++i) {
            block_id = L->block_idx_array[i];
            block_siz = L->pool->get_block_size(block_id);
            for (int j = 0; j < block_siz; ++j) {
                hub_type* x = L->pool->get_node(block_id, j);
                has->modify(x->hub_vertex, x->hop, hop_cst, x->distance);
                cnt ++;
                // if (cnt >= L->last_size) break;
            }
        }

        // ���� T ���У������� D ����
        for (int i = 0; i < t0->blocks_num; ++i) {
            block_id = t0->block_idx_array[i];
            block_siz = t0->pool->get_block_size(block_id);
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

                    // h Ϊ������Щ��ǩ�������� h + 1Ϊ����Ҫ���ӵı�ǩ����
                    int dv = dis + out_edge_weight[k];

                    // �ж����� D ����
                    weight_type d_hash = das->get(v);
                    if (d_hash == 1e9) {
                        d[d_end ++] = v;
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
        for (int i = d_start; i < d_end; ++i) {
            int sv = node_id, v = d[i], h = hop_now;
            weight_type dv = das->get(v);
            weight_type q_dis = query_dis_by_hash_table(sv, v, has, L_gpu + v, h + 1, hop_cst);
            
            if (dv < q_dis) {
                // ���ӱ�ǩ��ѹ�� T ����
                L_gpu[v].push_back({sv, h + 1, dv});
                t1->push_back({v, dv});
            }

            das->modify(v, 1e9);
        }

        // �Ļ� hashtable
        cnt = 0;
        for (int i = 0; i < L->blocks_num; ++i) {
            block_id = L->block_idx_array[i];
            block_siz = L->pool->get_block_size(block_id);
            for (int j = 0; j < block_siz; ++j) {
                hub_type* x = L->pool->get_node(block_id, j);
                has->modify(x->hub_vertex, x->hop, hop_cst, 1e9);
                cnt ++;
                // if (cnt >= L->last_size) break;
            }
        }
    }
}

// �������ɹ���_v3�������� D �����Ż���ʵ���� D ���б����Ĳ��У�������
__global__ void gen_label_hsdl_v3
(int V, int thread_num, int hop_cst, int hop_now, int* out_pointer, int* out_edge, int* out_edge_weight,
cuda_vector_v2<hub_type> *L_gpu, cuda_hashTable_v2<weight_type> *Has, cuda_hashTable_v2<weight_type> *Das,
cuda_vector_v2<T_item> *T0, cuda_vector_v2<T_item> *T1, int start_id, int end_id, int* L_push_back, int* T_push_back,
int *d, clock_t *timer_hash1, clock_t *timer_gett, clock_t *timer_query, clock_t *timer_hash2) {
    
    // �߳�id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < 0 || start_id + tid > end_id) {
        return;
    }

    // hash table
    cuda_hashTable_v2<weight_type> *has = (Has + tid);
    cuda_hashTable_v2<weight_type> *das = (Das + tid);
    int d_start, d_end;
    int block_id, block_siz;
    
    d_start = tid * V;
    d_end = d_start;

    int node_id = start_id + tid;
    // printf("%d, ", node_id);
    // node_id �� T ����
    cuda_vector_v2<T_item> *t0 = (T0 + node_id);
    cuda_vector_v2<T_item> *t1 = (T1 + node_id);

    // node_id �� label
    cuda_vector_v2<hub_type> *L = (L_gpu + node_id);
    
    timer_hash1[tid] += clock() / 1000;
    // ��ʼ�� hashtable�����Ǳ��� label ���ϲ�һһ�޸��� hashtable �е�ֵ
    int cnt = 0;
    for (int i = 0; i < L->blocks_num; ++i) {
        block_id = L->block_idx_array[i];
        block_siz = L->pool->get_block_size(block_id);
        for (int j = 0; j < block_siz; ++j) {
            hub_type *x = &(L->pool->blocks_pool[block_id].data[j]);
            has->modify(x->hub_vertex, x->hop, hop_cst, x->distance);
            if (++cnt >= L->last_size) break;
        }
    }
    timer_hash1[tid + thread_num] += clock() / 1000;

    timer_gett[tid] += clock() / 1000;
    // ���� T ���У������� D ����
    for (int i = 0; i < t0->blocks_num; ++i) {
        block_id = t0->block_idx_array[i];
        block_siz = t0->pool->get_block_size(block_id);
        for (int j = 0; j < block_siz; ++j) {

            // ��ȡ T ����Ԫ��
            // T_item *x = t0->pool->get_node(block_id, j);
            T_item *x = &(t0->pool->blocks_pool[block_id].data[j]);

            // sv Ϊ���, ev Ϊ�������ĵ�, dis Ϊ���룬hop Ϊ����
            int sv = node_id, ev = x->vertex, h = hop_now;
            weight_type dis = x->distance;

            // �����ڵ� ev ����չ
            for (int k = out_pointer[ev]; k < out_pointer[ev + 1]; ++k) {
                int v = out_edge[k];
                
                // rank pruning������ͬһ����Ҳ�����㡣
                if (sv >= v) continue;

                // h Ϊ������Щ��ǩ�������� h + 1Ϊ����Ҫ���ӵı�ǩ����
                int dv = dis + out_edge_weight[k];

                // �ж����� D ����
                weight_type d_hash = das->get(v);
                if (d_hash == 1e9) {
                    d[d_end ++] = v;
                    das->modify(v, dv);
                }else{
                    if (d_hash > dv) {
                        das->modify(v, dv);
                    }
                }

            }
        }
    }
    timer_gett[tid + thread_num] += clock() / 1000;

    // u, v, d_size, hash, label, t, hop, hop_cst;
    timer_query[tid] += clock() / 1000;
    query_parallel <<< (d_end - d_start + 1023) / 1024, 1024 >>>
    (node_id, d_start, d_end, d_end - d_start, das, &d[0], has, L_gpu, t1,
    V, thread_num, tid, L_push_back, T_push_back, hop_now, hop_cst);
    cudaDeviceSynchronize();
    timer_query[tid + thread_num] += clock() / 1000;

    // �Ļ� hashtable
    timer_hash2[tid] += clock() / 1000;
    cnt = 0;
    for (int i = 0; i < L->blocks_num; ++i) {
        block_id = L->block_idx_array[i];
        block_siz = L->pool->get_block_size(block_id);
        for (int j = 0; j < block_siz; ++j) {
            hub_type *x = &(L->pool->blocks_pool[block_id].data[j]);
            has->modify(x->hub_vertex, x->hop, hop_cst, 1e9);
            if (++cnt >= L->last_size) break;
        }
    }
    timer_hash2[tid + thread_num] += clock() / 1000;
}

__global__ void add_timer (clock_t* tot, clock_t *t, int thread_num) {
    for (int i = 0; i < thread_num; ++i) {
        (*tot) += (t[i + thread_num] - t[i]) / 1000;
    }
    printf("t: %lld\n", (long long)(*tot));
}

// ���� label �Ĺ���
void label_gen (CSR_graph<weight_type>& input_graph, hop_constrained_case_info_v2 *info, int hop_cst, vector<vector<hub_type> >&L) {

    int V = input_graph.OUTs_Neighbor_start_pointers.size() - 1;
    int E = input_graph.OUTs_Edges.size();
    int* out_edge = input_graph.out_edge;
    int* out_edge_weight = input_graph.out_edge_weight;
    int* out_pointer = input_graph.out_pointer;

    int thread_num = 1000;

    int dimGrid = 1, dimGrid_V, dimBlock = 1;
    dimGrid_V = (V + dimBlock - 1) / dimBlock;
    dimGrid = (thread_num + dimBlock - 1) / dimBlock;

    printf("V, E: %d, %d\n", V, E);

    // ׼�� info
    info->init(V, V * V * (hop_cst + 1), hop_cst);
    printf("init case_info success\n");

    // ׼�� L_hashTable
    cuda_hashTable_v2<weight_type> *L_hash;
    cudaMallocManaged(&L_hash, thread_num * sizeof(cuda_hashTable_v2<weight_type>));
    cudaDeviceSynchronize();
    for (int i = 0; i < thread_num; i++) {
        new (L_hash + i) cuda_hashTable_v2 <weight_type> (V * (hop_cst + 1));
    }

    // ׼�� D_hashTable
    cuda_hashTable_v2<weight_type> *D_hash;
    cudaMallocManaged(&D_hash, thread_num * sizeof(cuda_hashTable_v2<weight_type>));
    cudaDeviceSynchronize();
    for (int i = 0; i < thread_num; i++) {
        new (D_hash + i) cuda_hashTable_v2 <weight_type> (V);
    }
    printf("init hash_table success\n");

    // ׼�� D_vector
    int *D_vector;
    cudaMallocManaged(&D_vector, thread_num * V * sizeof(int));
    
    // ׼�� L_push_back_table
    int *L_push_back;
    cudaMallocManaged(&L_push_back, thread_num * V * sizeof(int));

    // ׼�� T_push_back_table
    int *T_push_back;
    cudaMallocManaged(&T_push_back, thread_num * V * sizeof(int));

    // ���ԽС�ĵ㣬rank Խ��
    // for (int i = 0; i < V; i ++){
    //     printf("degree %d, %d\n", i, out_pointer[i + 1] - out_pointer[i]);
    // }

    // ͬ������֤���ݳ�ʼ�����
    cudaDeviceSynchronize();

    // ��ʱ label gen �еĺ�ʱ
    clock_t *timer_hash1, *timer_gett, *timer_query, *timer_hash2;
    clock_t *timer_hash1_tot, *timer_gett_tot, *timer_query_tot, *timer_hash2_tot;
    cudaMallocManaged(&timer_hash1, thread_num * 2 * sizeof(clock_t));
    cudaMallocManaged(&timer_hash1_tot, sizeof(clock_t));
    cudaMallocManaged(&timer_gett, thread_num * 2 * sizeof(clock_t));
    cudaMallocManaged(&timer_gett_tot, sizeof(clock_t));
    cudaMallocManaged(&timer_query, thread_num * 2 * sizeof(clock_t));
    cudaMallocManaged(&timer_query_tot, sizeof(clock_t));
    cudaMallocManaged(&timer_hash2, thread_num * 2 * sizeof(clock_t));
    cudaMallocManaged(&timer_hash2_tot, sizeof(clock_t));
    
    // ���� cuda_vector �� cuda_hash �Ĳ���
    // test_mmpool(V, thread_num, 2, info, L_hash);
    
    init_T <<<dimGrid_V, dimBlock>>> (V, info->T0, info->L_cuda);
    cudaDeviceSynchronize();

    // ����������������ֱ��̽�� T �Ƿ�Ϊ��
    int iter = 0;
    int start_id, end_id;

    // ��ʱ
    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    while (1) {
        if (iter++ >= hop_cst) break;
        // iter = 1 -> ���� ���� 1
        // iter = 2 -> ���� ���� 2
        // iter = 3 -> ���� ���� 3
        // iter = 4 -> ���� ���� 4
        // iter = 5 -> ���� ���� 5
        
        printf("iteration_hop: %d\n", iter);

        start_id = V;
        while (start_id > 0) {
            end_id = start_id - 1;
            start_id = max(0, start_id - thread_num);
            // printf("st_id, ed_id: %d %d\n", start_id, end_id);
            
            // ������ż�ԣ�����ʹ�� T0��T1������Ҫ����ָ��
            if (iter % 2 == 1) {
                gen_label_hsdl_v3 <<< dimGrid, dimBlock >>> 
                (V, thread_num, hop_cst, iter - 1, out_pointer, out_edge, out_edge_weight,
                info->L_cuda, L_hash, D_hash, info->T0, info->T1, start_id, end_id, L_push_back, T_push_back,
                D_vector, timer_hash1, timer_gett, timer_query, timer_hash2);
                cudaDeviceSynchronize();

                // push_back L
                Push_Back_L <<< dimGrid_V, dimBlock >>>
                (V, thread_num, start_id, end_id, iter, L_push_back, info->L_cuda);

                // push_back T
                Push_Back_T <<< dimGrid_V, dimBlock >>>
                (V, thread_num, start_id, end_id, iter, T_push_back, info->T1);
                cudaDeviceSynchronize();
            
            }else{
                gen_label_hsdl_v3 <<< dimGrid, dimBlock >>> 
                (V, thread_num, hop_cst, iter - 1, out_pointer, out_edge, out_edge_weight,
                info->L_cuda, L_hash, D_hash, info->T1, info->T0, start_id, end_id, L_push_back, T_push_back,
                D_vector, timer_hash1, timer_gett, timer_query, timer_hash2);
                cudaDeviceSynchronize();

                // push_back L
                Push_Back_L <<< dimGrid_V, dimBlock >>>
                (V, thread_num, start_id, end_id, iter, L_push_back, info->L_cuda);
                
                // push_back T
                Push_Back_T <<< dimGrid_V, dimBlock >>>
                (V, thread_num, start_id, end_id, iter, T_push_back, info->T0);
                cudaDeviceSynchronize();
            }
        }
        if (iter % 2 == 1) {
            // ��ϴ T ����
            clear_T <<< dimGrid_V, dimBlock >>> (V, info->T0, info->L_cuda);
        }else{
            // ��ϴ T ����
            clear_T <<< dimGrid_V, dimBlock >>> (V, info->T1, info->L_cuda);
        }
        cudaDeviceSynchronize();
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        printf("Time generation in hop %d : %.8lf s\n", iter, elapsedTime / 1000.0);
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
    printf("Time generation: %.8f s\n", elapsedTime / 1000.0);
    info->time_generate_labels = elapsedTime / 1000.0;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    add_timer <<< 1, 1 >>> (timer_hash1_tot, timer_hash1, thread_num);
    add_timer <<< 1, 1 >>> (timer_gett_tot, timer_gett, thread_num);
    add_timer <<< 1, 1 >>> (timer_query_tot, timer_query, thread_num);
    add_timer <<< 1, 1 >>> (timer_hash2_tot, timer_hash2, thread_num);

    // printf("hub, parent, hop, dis:\n");
    auto beforeTime = std::chrono::steady_clock::now();
    info->label_size = 0;
    for (int v = 0; v < V; ++v) {
        L[v].clear();
        // printf("vertex %d\n", v);
        for (int i = 0; i < info->L_cuda[v].blocks_num; ++i) {
            int block_id = info->L_cuda[v].block_idx_array[i];
            int block_siz = info->L_cuda[v].pool->get_block_size_host(block_id);
            for (int j = 0; j < block_siz; ++j) {
                hub_type* x = info->L_cuda[v].pool->get_node_host(block_id, j);
                // printf("{%d, %d, %d, %d}, ", x->hub_vertex, x->parent_vertex, x->hop, x->distance);
                L[v].push_back({x->hub_vertex, x->hop, x->distance});
                // info->L_cpu[v].push_back({x->hub_vertex, x->hop, x->distance});
                info->label_size ++;
            }
        }
        // printf("\n");
    }
    auto afterTime = std::chrono::steady_clock::now();
    printf("time traverse labels: %.8lf\n", std::chrono::duration<double>(afterTime - beforeTime).count());

    info->label_size = info->label_size / (double)V;
    printf("average label size: %.8lf\n", info->label_size);
    printf("Generation end!\n");

    info->destroy_L_cuda();
    
    for(int i = 0; i < thread_num; ++i){
        L_hash[i].~cuda_hashTable_v2();
    }
    cudaFree(L_hash);
    
    return;
}