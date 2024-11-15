#include "label/gen_label.cuh"

void test_cuda_error (std::string s) {
    cudaError_t err = cudaGetLastError(); // Check for kernel memory request errors
    if (err != cudaSuccess) {
        printf("!INIT CUDA ERROR: %s %s\n", s.c_str(), cudaGetErrorString(err));
    }
}

__device__ void test_cuda_error_device (char s[]) {
    cudaError_t err = cudaGetLastError(); // Check for kernel memory request errors
    if (err != cudaSuccess) {
        printf("!INIT CUDA ERROR: %s %s\n", s, cudaGetErrorString(err));
    }
}

// Quick queries via hashtable
__device__ int query_dis_by_hash_table
(int u, int v, cuda_hashTable_v2<weight_type> *H, cuda_vector_v2<hub_type> *L, int hop_now, int hop_cst) {
    int min_dis = 1e9;
    int block_num = L->blocks_num;
    int cnt = 0;
    hub_type *x;
    for (int i = 0; i < block_num; ++i) {
        int block_id = L->block_idx_array[i];
        int block_siz = L->pool->get_block_size(block_id);
        for (int j = 0; j < block_siz; ++j) {
            x = &(L->pool->blocks_pool[block_id].data[j]);
            for (int k = hop_now - (x->hop & ((1 << 5) - 1)); k >= 0; --k) {
                min_dis = min(min_dis, x->distance + H->get(x->hub_vertex, k, hop_cst));
            }
            if (++cnt >= L->last_size) break;
        }
    }
    return min_dis;
}

// Dynamic parallel query acceleration
// u, v, d_size, d_has, d, label, hop, hop_cst;
__global__ void query_parallel (int sv, int st, int sz, cuda_hashTable_v2<weight_type> *das, int *d, cuda_hashTable_v2<weight_type> *has,
cuda_vector_v2<hub_type> *L_gpu, int thread_num, int tidd, int hop_now, int hop_cst, int *Num_L, std::pair<int, int> *L_push_back) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < 0 || tid >= sz) {
        return;
    }

    // Gets the D queue element
    int v = d[st + tid];
    weight_type dv = das->get(v);
    weight_type q_dis = query_dis_by_hash_table(sv, v, has, L_gpu + v, hop_now + 1, hop_cst);
    
    int x;
    int y = v * thread_num;
    if (dv < q_dis) {
        // 添加标签并压入 T 队列
        // 表示第 v 个 L 需要 push_back 一个 sv 的并且距离为 dv 的元素。
        x = atomicAdd(&Num_L[v], 1);
        L_push_back[y + x].first = tidd, L_push_back[y + x].second = dv;
    }

    das->modify(v, 1e9);
}

// Inserted into Label by L_push_back
__global__ void Push_Back_L (int V, int thread_num, int start_id, int end_id, int hop, cuda_vector_v2<hub_type> *L_gpu,
int *d_par, int *nid, int *Num_L, std::pair<int, int> *L_push_back, int *Num_T, std::pair<int, int> *T_push_back) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 0 || tid >= V) {
        return;
    }
    // tid = nid[start_id + tid];
    int st = tid * thread_num;
    int x;
    int bas;
    for (int i = st; i < st + Num_L[tid]; ++i) {
        bas = L_push_back[i].first * V;
        L_gpu[tid].push_back({start_id + L_push_back[i].first, 
                                (d_par[bas + tid] << 6) + hop, L_push_back[i].second});

        x = atomicAdd(&Num_T[L_push_back[i].first], 1);
        T_push_back[bas + x].first = tid;
        T_push_back[bas + x].second = L_push_back[i].second;
    }
    Num_L[tid] = 0;
    L_gpu[tid].last_size = L_gpu[tid].current_size;
}

// Inserted into T by T_push_back
__global__ void Push_Back_T (int V, int thread_num, int start_id, int end_id, cuda_vector_v2<T_item> *T, 
int *nid, int *Num_T, std::pair<int, int> *T_push_back) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < 0 || tid >= thread_num) {
        return;
    }
    
    int st = tid * V;
    for (int i = st; i < st + Num_T[tid]; i++) {
        T[start_id + tid].push_back({T_push_back[i].first, T_push_back[i].second});
    }

    Num_T[tid] = 0;
}

// Clear T
__global__ void clear_T (int G_max, cuda_vector_v2<T_item> *T) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < G_max) {
        // target_vertex, distance
        T[tid].init(G_max, tid);
    }
}

// Clear T
__global__ void clear_T (int G_max, cuda_vector_v2<T_item> *T, cuda_vector_v2<hub_type> *L_gpu) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < G_max) {
        T[tid].init(G_max, tid);
    }
}

// Clear L
__global__ void clear_L (int V, cuda_vector_v2<hub_type> *L_gpu) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < V) {
        // target_vertex, distance
        L_gpu[tid].init(V, tid);
    }
}

// Initialize T
__global__ void init_T (int G_max, cuda_vector_v2<T_item> *T, cuda_vector_v2<hub_type> *L_gpu, int *nid) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < G_max) {
        L_gpu[nid[tid]].push_back({tid, (nid[tid] << 6), 0});
        L_gpu[nid[tid]].last_size = 1;
    
        // tid = nid[tid];
        // target_vertex, distance
        T[tid].push_back({nid[tid], 0});
    }
}

// Index generation process, naive parallelism
__global__ void gen_label_hsdl (int V, int thread_num, int hop_cst, int hop_now, int* out_pointer, int* out_edge, int* out_edge_weight,
            cuda_vector_v2<hub_type> *L_gpu, cuda_hashTable_v2<weight_type> *Has, cuda_vector_v2<T_item> *T0, cuda_vector_v2<T_item> *T1) {
    
    // 线程id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < 0 || tid >= thread_num) {
        return;
    }

    // hash table
    cuda_hashTable_v2<weight_type> *has = (Has + tid);

    for (int node_id = tid; node_id < V; node_id += thread_num) {

        // node_id 的 T 队列
        cuda_vector_v2<T_item> *t0 = (T0 + node_id);
        cuda_vector_v2<T_item> *t1 = (T1 + node_id);

        // node_id 的 label
        cuda_vector_v2<hub_type> *L = (L_gpu + node_id);

        // 初始化 hashtable，就是遍历 label 集合并一一修改在 hashtable 中的值
        for (int i = 0; i < L->blocks_num; ++i) {
            int block_id = L->block_idx_array[i];
            int block_siz = L->pool->get_block_size(block_id);
            for (int j = 0; j < block_siz; ++j) {
                hub_type* x = L->pool->get_node(block_id, j);
                has->modify(x->hub_vertex, x->hop, hop_cst, x->distance);
            }
        }

        // 遍历 T 队列
        for (int i = 0; i < t0->blocks_num; ++i) {
            int block_id = t0->block_idx_array[i];
            int block_siz = t0->pool->get_block_size(block_id);
            for (int j = 0; j < block_siz; ++j) {

                // 获取 T 队列元素
                T_item *x = t0->pool->get_node(block_id, j);

                // sv 为起点, ev 为遍历到的点, dis 为距离，hop 为跳数
                int sv = node_id, ev = x->vertex, h = hop_now;
                weight_type dis = x->distance;

                // 遍历节点 ev 并扩展
                for (int k = out_pointer[ev]; k < out_pointer[ev + 1]; ++k) {
                    int v = out_edge[k];
                    
                    // rank pruning，并且同一个点也不能算。
                    if (sv >= v) continue;

                    // h 为现在这些标签的跳数， h + 1为现在要添加的标签跳数
                    int dv = dis + out_edge_weight[k];
                    weight_type q_dis = query_dis_by_hash_table(sv, v, Has + tid, L_gpu + v, h + 1, hop_cst);
                    
                    if (dv < q_dis) {
                        // 添加标签并压入 T 队列
                        L_gpu[v].push_back({sv, h + 1, dv});
                        t1->push_back({v, dv});
                    }

                }
            }
        }

        // 改回 hashtable
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

// The index generation process _v2 is added to the D queue optimization without redundancy
__global__ void gen_label_hsdl_v2 (int V, int thread_num, int hop_cst, int hop_now, int* out_pointer, int* out_edge, int* out_edge_weight,
            cuda_vector_v2<hub_type> *L_gpu, cuda_hashTable_v2<weight_type> *Has, cuda_hashTable_v2<weight_type> *Das,
            cuda_vector_v2<T_item> *T0, cuda_vector_v2<T_item> *T1, int *d) {
    
    // 线程id
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
        
        // node_id 的 T 队列
        cuda_vector_v2<T_item> *t0 = (T0 + node_id);
        cuda_vector_v2<T_item> *t1 = (T1 + node_id);

        // node_id 的 label
        cuda_vector_v2<hub_type> *L = (L_gpu + node_id);

        // 初始化 hashtable，就是遍历 label 集合并一一修改在 hashtable 中的值
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

        // 遍历 T 队列，并生成 D 队列
        for (int i = 0; i < t0->blocks_num; ++i) {
            block_id = t0->block_idx_array[i];
            block_siz = t0->pool->get_block_size(block_id);
            for (int j = 0; j < block_siz; ++j) {

                // 获取 T 队列元素
                T_item *x = t0->pool->get_node(block_id, j);

                // sv 为起点, ev 为遍历到的点, dis 为距离，hop 为跳数
                int sv = node_id, ev = x->vertex, h = hop_now;
                weight_type dis = x->distance;

                // 遍历节点 ev 并扩展
                for (int k = out_pointer[ev]; k < out_pointer[ev + 1]; ++k) {
                    int v = out_edge[k];
                    
                    // rank pruning，并且同一个点也不能算。
                    if (sv >= v) continue;

                    // h 为现在这些标签的跳数， h + 1为现在要添加的标签跳数
                    int dv = dis + out_edge_weight[k];

                    // 判断生成 D 队列
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

        // 遍历 D 队列
        for (int i = d_start; i < d_end; ++i) {
            int sv = node_id, v = d[i], h = hop_now;
            weight_type dv = das->get(v);
            weight_type q_dis = query_dis_by_hash_table(sv, v, has, L_gpu + v, h + 1, hop_cst);
            
            if (dv < q_dis) {
                // 添加标签并压入 T 队列
                L_gpu[v].push_back({sv, h + 1, dv});
                t1->push_back({v, dv});
            }

            das->modify(v, 1e9);
        }

        // 改回 hashtable
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

// The index generation process _v3 is added to D queue optimization to realize parallel D queue traversal without redundancy
__global__ void gen_label_hsdl_v3
(int V, int thread_num, int hop_cst, int hop_now, int* out_pointer, int* out_edge, int* out_edge_weight,
cuda_vector_v2<hub_type> *L_gpu, cuda_hashTable_v2<weight_type> *Has, cuda_hashTable_v2<weight_type> *Das,
cuda_vector_v2<T_item> *T0, int start_id, int end_id, int *d, int *d_par, int *nid, int *Num_L, std::pair<int, int> *L_push_back) {
    
    // thread id
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

    int node_id = nid[start_id + tid];
    
    // T queue of node_id
    cuda_vector_v2<T_item> *t0 = (T0 + start_id + tid);

    // Label of node_id
    cuda_vector_v2<hub_type> *L = (L_gpu + node_id);

    // 遍历 T 队列，并生成 D 队列
    for (int i = 0; i < t0->blocks_num; ++i) {
        block_id = t0->block_idx_array[i];
        block_siz = t0->pool->get_block_size(block_id);
        for (int j = 0; j < block_siz; ++j) {

            // 获取 T 队列元素
            // T_item *x = t0->pool->get_node(block_id, j);
            T_item *x = &(t0->pool->blocks_pool[block_id].data[j]);

            // sv 为起点, ev 为遍历到的点, dis 为距离，hop 为跳数
            int sv = node_id, ev = x->vertex, h = hop_now;
            weight_type dis = x->distance;

            // 遍历节点 ev 并扩展
            for (int k = out_pointer[ev]; k < out_pointer[ev + 1]; ++k) {
                int v = out_edge[k];
                
                // rank pruning，并且同一个点也不能算。
                if (sv >= v) continue;

                // h 为现在这些标签的跳数， h + 1为现在要添加的标签跳数
                weight_type dv = dis + out_edge_weight[k];

                // 判断生成 D 队列
                weight_type d_hash = das->get(v);
                if (d_hash == 1e9) {
                    d[d_end ++] = v;
                    das->modify(v, dv);
                    d_par[d_start + v] = ev;
                } else {
                    if (d_hash > dv) {
                        das->modify(v, dv);
                        d_par[d_start + v] = ev;
                    }
                }

            }
        }
    }
    cudaDeviceSynchronize();
    
    if (d_end - d_start > 0) {

        // 初始化 hashtable，就是遍历 label 集合并一一修改在 hashtable 中的值
        int cnt = 0;
        hub_type *x;
        for (int i = 0; i < L->blocks_num; ++i) {
            block_id = L->block_idx_array[i];
            block_siz = L->pool->get_block_size(block_id);
            for (int j = 0; j < block_siz; ++j) {
                x = &(L->pool->blocks_pool[block_id].data[j]);
                has->modify(x->hub_vertex, (x->hop & ((1<<5) - 1)), hop_cst, x->distance);
                if (++cnt >= L->last_size) break;
            }
        }

        // u, v, d_size, hash, label, t, hop, hop_cst
        query_parallel <<< (d_end - d_start + 127) / 128, 128 >>>
        (node_id, d_start, d_end - d_start, das, d, has, L_gpu, thread_num, tid, hop_now, hop_cst, Num_L, L_push_back);
        cudaDeviceSynchronize();
    
        // change back to hashtable
        cnt = 0;
        for (int i = 0; i < L->blocks_num; ++i) {
            block_id = L->block_idx_array[i];
            block_siz = L->pool->get_block_size(block_id);
            for (int j = 0; j < block_siz; ++j) {
                x = &(L->pool->blocks_pool[block_id].data[j]);
                has->modify(x->hub_vertex, (x->hop & ((1 << 5) - 1)), hop_cst, 1e9);
                if (++cnt >= L->last_size) break;
            }
        }

    }

}

__global__ void add_timer (clock_t* tot, clock_t *t, int thread_num) {
    for (int i = 0; i < thread_num; ++i) {
        (*tot) += (t[i + thread_num] - t[i]) / 1000;
    }
    printf("t: %lld\n", (long long)(*tot));
}

// 生成 label 的过程
void label_gen (CSR_graph<weight_type>& input_graph, hop_constrained_case_info_v2 *info, std::vector<std::vector<hub_type_v2> >&L, std::vector<int>& nid_vec, int nid_vec_id) {

    cudaError_t err;

    int V = input_graph.OUTs_Neighbor_start_pointers.size() - 1;
    int E = input_graph.OUTs_Edges.size();
    int* out_edge = input_graph.out_edge;
    int* out_edge_weight = input_graph.out_edge_weight;
    int* out_pointer = input_graph.out_pointer;
    int hop_cst = info->hop_cst;
    int thread_num = info->thread_num;
    int vertex_num = info->nid_size[nid_vec_id];
    
    // thread_num = min(thread_num, vertex_num);
    if (info->use_new_algo) {
        int dimGrid_thread, dimGrid_V, dimGrid_G_max, dimBlock = 64;
        dimGrid_V = (V + dimBlock - 1) / dimBlock;
        dimGrid_thread = (thread_num + dimBlock - 1) / dimBlock;
        dimGrid_G_max = (vertex_num + dimBlock - 1) / dimBlock;

        // printf("V, E, vertex_num: %d, %d, %d\n", V, E, vertex_num);

        // 准备 info
        cuda_hashTable_v2<weight_type> *L_hash = info->L_hash;
        cuda_hashTable_v2<weight_type> *D_hash = info->D_hash;
        int *D_vector = info->D_vector;
        int *D_par = info->D_pare;

        int *Num_T = info->Num_T; // 测试用
        int *Num_L = info->Num_L; // 测试用
        std::pair<int, int> *T_push_back = info->T_push_back;
        std::pair<int, int> *L_push_back = info->L_push_back;
        
        // 编号越小的点，rank 越高
        // for (int i = 0; i < V; i ++){
        //     printf("degree %d, %d\n", i, out_pointer[i + 1] - out_pointer[i]);
        // }
        
        // 测试 cuda_vector 和 cuda_hash 的部分
        // test_mmpool(V, thread_num, 2, info, L_hash);
        int *nid = info->nid[nid_vec_id];

        clear_L <<< dimGrid_V, dimBlock >>> (V, info->L_cuda);
        clear_T <<< dimGrid_G_max, dimBlock >>> (vertex_num, info->T0);
        clear_T <<< dimGrid_G_max, dimBlock >>> (vertex_num, info->T1);
        init_T <<< dimGrid_G_max, dimBlock >>> (vertex_num, info->T0, info->L_cuda, nid);

        // Auxiliary variable, it is not convenient to directly detect whether T is empty
        int iter = 0;
        int start_id, end_id;

        // 计时
        cudaEvent_t start, stop;
        clock_t start_time, end_time;
        double time1 = 0.0, time2 = 0.0, time3 = 0.0, time4 = 0.0;
        float elapsedTime = 0.0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start, 0);
        while (1) {
            if (iter ++ >= hop_cst) break;
            // iter = 1 -> 生成 跳数 1
            // iter = 2 -> 生成 跳数 2
            // iter = 3 -> 生成 跳数 3
            // iter = 4 -> 生成 跳数 4
            // iter = 5 -> 生成 跳数 5
            
            // printf("iteration_hop: %d\n", iter);

            start_id = vertex_num;
            // end_id = -1;
            // start_id = -1;
            while (start_id > 0) {
                // start_id = end_id + 1;
                // end_id = min(V - 1, start_id + thread_num - 1);
                end_id = start_id - 1;
                start_id = max(0, start_id - thread_num);
                // printf("start, end: %d %d !\n", start_id, end_id);

                // 根据奇偶性，轮流使用 T0、T1，不需要交换指针
                if (iter % 2 == 1) {

                    start_time = clock();
                    gen_label_hsdl_v3 <<< dimGrid_thread, dimBlock >>> (V, thread_num, hop_cst, iter - 1, out_pointer, out_edge, out_edge_weight,
                    info->L_cuda, L_hash, D_hash, info->T0, start_id, end_id, D_vector, D_par, nid, Num_L, L_push_back);
                    cudaDeviceSynchronize();
                    end_time = clock();
                    time1 += (double)(end_time - start_time) / CLOCKS_PER_SEC;
                    
                    // push_back L
                    start_time = clock();
                    Push_Back_L <<< dimGrid_V, dimBlock >>> (V, thread_num, start_id, end_id, iter, info->L_cuda, D_par, nid, Num_L, L_push_back, Num_T, T_push_back);
                    cudaDeviceSynchronize();
                    end_time = clock();
                    time2 += (double)(end_time - start_time) / CLOCKS_PER_SEC;
                    // clear_T <<< dimGrid_V, dimBlock >>> (V, info->T0, info->L_cuda);
                    // cudaDeviceSynchronize();

                    // push_back T
                    start_time = clock();
                    Push_Back_T <<< dimGrid_thread, dimBlock >>> (V, thread_num, start_id, end_id, info->T1, nid, Num_T, T_push_back);
                    cudaDeviceSynchronize();
                    end_time = clock();
                    time3 += (double)(end_time - start_time) / CLOCKS_PER_SEC;

                }else{
                    start_time = clock();
                    gen_label_hsdl_v3 <<< dimGrid_thread, dimBlock >>> (V, thread_num, hop_cst, iter - 1, out_pointer, out_edge, out_edge_weight,
                    info->L_cuda, L_hash, D_hash, info->T1, start_id, end_id, D_vector, D_par, nid, Num_L, L_push_back);
                    cudaDeviceSynchronize();
                    end_time = clock();
                    time1 += (double)(end_time - start_time) / CLOCKS_PER_SEC;

                    // push_back L
                    start_time = clock();
                    Push_Back_L <<< dimGrid_V, dimBlock >>> (V, thread_num, start_id, end_id, iter, info->L_cuda, D_par, nid, Num_L, L_push_back, Num_T, T_push_back);
                    cudaDeviceSynchronize();
                    end_time = clock();
                    time2 += (double)(end_time - start_time) / CLOCKS_PER_SEC;
                    // clear_T <<< dimGrid_V, dimBlock >>> (V, info->T0, info->L_cuda);
                    // cudaDeviceSynchronize();

                    // push_back T
                    start_time = clock();
                    Push_Back_T <<< dimGrid_thread, dimBlock >>> (V, thread_num, start_id, end_id, info->T0, nid, Num_T, T_push_back);
                    cudaDeviceSynchronize();
                    end_time = clock();
                    time3 += (double)(end_time - start_time) / CLOCKS_PER_SEC;
                }
            }

            start_time = clock();
            if (iter % 2 == 1) {
                // 清洗 T 数组
                clear_T <<< dimGrid_V, dimBlock >>> (vertex_num, info->T0);
            }else{
                // 清洗 T 数组
                clear_T <<< dimGrid_V, dimBlock >>> (vertex_num, info->T1);
            }
            cudaDeviceSynchronize();
            end_time = clock();
            time4 += (double)(end_time - start_time) / CLOCKS_PER_SEC;
            
            // cudaEventRecord(stop, 0);
            // cudaEventSynchronize(stop);
            // cudaEventElapsedTime(&elapsedTime, start, stop);
            // printf("Time generation in hop %d : %.8lf s\n", iter, elapsedTime / 1000.0);
        }
        // printf("time 1, 2, 3, 4: %.5lf, %.5lf, %.5lf, %.5lf \n", time1, time2, time3, time4);

        err = cudaGetLastError(); // 检查内核内存申请错误
        if (err != cudaSuccess) {
            printf("!INIT CUDA ERROR3: %s\n", cudaGetErrorString(err));
        }
        
        // timer record
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop);
        // printf("Time generation: %.6fs\n", elapsedTime / 1000.0);

        info->time_generate_labels += elapsedTime / 1000.0;
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    } else if (info->use_2023WWW_GPU_version) {
        
    }

    // printf("hub, parent, hop, dis:\n");
    auto begin = std::chrono::high_resolution_clock::now();
    long long label_size = 0;
    
    // int cpu_thread_num = 100;
    // ThreadPool pool(cpu_thread_num);
    // std::vector<std::future<int>> results;
    // for (int q = 0; q < cpu_thread_num; ++q) {
    //     results.emplace_back(pool.enqueue(
    // 		[q, cpu_thread_num, V, &L, &nid_vec, info] {
    //         for (int v = q; v < V; v += cpu_thread_num) {
    //             for (int i = 0; i < info->L_cuda[v].blocks_num; ++i) {
    //                 int block_id = info->L_cuda[v].block_idx_array[i];
    //                 int block_siz = info->L_cuda[v].pool->get_block_size_host(block_id);
    //                 for (int j = 0; j < block_siz; ++j) {
    //                     hub_type* x = info->L_cuda[v].pool->get_node_host(block_id, j);
    //                     L[v].push_back({nid_vec[x->hub_vertex], 0, x->hop, x->distance});
    //                     // L[v].insert(L[v].end(), {nid_vec[x->hub_vertex], 0, x->hop, x->distance});
    //                 }
    //             }
    //         }
    //         return 1;
    //     }));
    // }
    // for (auto &&result : results){
    // 	result.get();
    // }
    // results.clear();

    for (int v = 0; v < V; ++v) {
        for (int i = 0; i < info->L_cuda[v].blocks_num; ++i) {
            int block_id = info->L_cuda[v].block_idx_array[i];
            int block_siz = info->L_cuda[v].pool->get_block_size_host(block_id);
            for (int j = 0; j < block_siz; ++j) {
                hub_type* x = info->L_cuda[v].pool->get_node_host(block_id, j);
                // L[v].push_back({nid_vec[x->hub_vertex], 0, x->hop, x->distance});
                L[v].insert(L[v].end(), {nid_vec[x->hub_vertex], (x->hop >> 6), x->hop & ((1 << 5) - 1), x->distance});
                label_size ++;
            }
        }
    }

    //printf Num_L
    // for (int v = V / 100 * 99; v < V; ++v) {
    //     printf("%d ", Num_L[v]);
    //     Num_L[v] = 0;
    // }
    // puts("");
    
    auto end = std::chrono::high_resolution_clock::now();
    // printf("Time traverse: %6lf\n", std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9);
    info->time_traverse_labels += std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count() / 1e9;
    
    info->label_size += label_size / (double)V;
    printf("average label size: %.6lf\n", label_size / (double)V);
    // printf("Generation GPU end!\n");
    
    return;
}