// #include "label/gen_label.cuh"

// // 通过 hashtable 的快速查询
// __device__ int query_dis_by_hash_table (int u, int v, cuda_hashTable_v2<weight_type> *H, cuda_vector_v2<hub_type> *L, int hop_now, int hop_cst) {
//     int min_dis = 1e9;

//     int block_num = L->blocks_num;
//     int cnt = 0;
//     for (int i = 0; i < block_num; ++i) {
//         int block_id = L->block_idx_array[i];
//         // __threadfence_system();
//         int block_siz = L->pool->get_block_size(block_id);
//         // __threadfence_system();
//         for (int j = 0; j < block_siz; ++j) {
//             // hub_type* x = L->pool->get_node(block_id, j);
//             hub_type *x = &(L->pool->blocks_pool[block_id].data[j]);
//             // __threadfence_system();
//             for (int k = hop_now - x->hop; k >= 0; --k) {
//                 // int ddis = x->distance + H->get(x->hub_vertex, k, hop_cst);
//                 // if (ddis < min_dis){
//                 //     min_dis = ddis;
//                 //     break;
//                 // }
//                 min_dis = min(min_dis, x->distance + H->get(x->hub_vertex, k, hop_cst));
//                 // __threadfence_system();
//             }
//             cnt ++;
//             if (cnt >= L->last_size) break;
//         }
//     }
//     return min_dis;

// }

// // 动态并行加速查询
// // u, v, d_size, d_has, d, label, hop, hop_cst;
// // (node_id, d->current_size, das, d, has, L_gpu, t1, hop_now, hop_cst)
// __global__ void query_parallel (int sv, int st, int ed, int sz, cuda_hashTable_v2<weight_type> *das, int *d,
// cuda_hashTable_v2<weight_type> *has, cuda_vector_v2<hub_type> *L_gpu, cuda_vector_v2<T_item> *t1, int hop_now, int hop_cst) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
//     if (tid < 0 || tid >= sz) {
//         return;
//     }

//     // 获取 D 队列元素
//     int v = d[st + tid];
//     weight_type dv = das->get(v);
//     weight_type q_dis = query_dis_by_hash_table(sv, v, has, L_gpu + v, hop_now + 1, hop_cst);
    
//     if (dv < q_dis) {
//         // 添加标签并压入 T 队列
//         L_gpu[v].push_back({sv, hop_now + 1, dv});
//         t1->push_back({v, dv});
//     }
//     // __syncthreads();

//     das->modify(v, 1e9);

// }

// // 初始化 T
// __global__ void init_T (int V, cuda_vector_v2<T_item> *T, cuda_vector_v2<hub_type> *L_gpu) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < V) {
//         // target_vertex, distance
//         T[tid].push_back({tid, 0});
//         L_gpu[tid].push_back({tid, 0, 0});
//         L_gpu[tid].last_size = 1;
//     }
// }

// // 清空 T
// __global__ void clear_T (int V, cuda_vector_v2<T_item> *T, cuda_vector_v2<hub_type> *L_gpu) {
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
//     if (tid < V) {
//         T[tid].init(V, tid);
//         L_gpu[tid].last_size = L_gpu[tid].current_size;
//     }
// }

// // 索引生成过程，朴素的并行
// __global__ void gen_label_hsdl (int V, int thread_num, int hop_cst, int hop_now, int* out_pointer, int* out_edge, int* out_edge_weight,
//             cuda_vector_v2<hub_type> *L_gpu, cuda_hashTable_v2<weight_type> *Has, cuda_vector_v2<T_item> *T0, cuda_vector_v2<T_item> *T1) {
    
//     // 线程id
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
//     if (tid < 0 || tid >= thread_num) {
//         return;
//     }

//     // hash table
//     cuda_hashTable_v2<weight_type> *has = (Has + tid);

//     for (int node_id = tid; node_id < V; node_id += thread_num) {

//         // node_id 的 T 队列
//         cuda_vector_v2<T_item> *t0 = (T0 + node_id);
//         cuda_vector_v2<T_item> *t1 = (T1 + node_id);

//         // node_id 的 label
//         cuda_vector_v2<hub_type> *L = (L_gpu + node_id);

//         // 初始化 hashtable，就是遍历 label 集合并一一修改在 hashtable 中的值
//         for (int i = 0; i < L->blocks_num; ++i) {
//             int block_id = L->block_idx_array[i];
//             int block_siz = L->pool->get_block_size(block_id);
//             for (int j = 0; j < block_siz; ++j) {
//                 hub_type* x = L->pool->get_node(block_id, j);
//                 has->modify(x->hub_vertex, x->hop, hop_cst, x->distance);
//             }
//         }

//         // 遍历 T 队列
//         for (int i = 0; i < t0->blocks_num; ++i) {
//             int block_id = t0->block_idx_array[i];
//             int block_siz = t0->pool->get_block_size(block_id);
//             for (int j = 0; j < block_siz; ++j) {

//                 // 获取 T 队列元素
//                 T_item *x = t0->pool->get_node(block_id, j);

//                 // sv 为起点, ev 为遍历到的点, dis 为距离，hop 为跳数
//                 int sv = node_id, ev = x->vertex, h = hop_now;
//                 weight_type dis = x->distance;

//                 // 遍历节点 ev 并扩展
//                 for (int k = out_pointer[ev]; k < out_pointer[ev + 1]; ++k) {
//                     int v = out_edge[k];
                    
//                     // rank pruning，并且同一个点也不能算。
//                     if (sv >= v) continue;

//                     // h 为现在这些标签的跳数， h + 1为现在要添加的标签跳数
//                     int dv = dis + out_edge_weight[k];
//                     weight_type q_dis = query_dis_by_hash_table(sv, v, Has + tid, L_gpu + v, h + 1, hop_cst);
                    
//                     if (dv < q_dis) {
//                         // 添加标签并压入 T 队列
//                         L_gpu[v].push_back({sv, h + 1, dv});
//                         t1->push_back({v, dv});
//                     }

//                 }
//             }
//         }

//         // 改回 hashtable
//         for (int i = 0; i < L->blocks_num; ++i) {
//             int block_id = L->block_idx_array[i];
//             int block_siz = L->pool->get_block_size(block_id);
//             for (int j = 0; j < block_siz; ++j) {
//                 hub_type* x = L->pool->get_node(block_id, j);
//                 has->modify(x->hub_vertex, x->hop, hop_cst, 1e9);
//             }
//         }
//     }
// }

// // 索引生成过程_v2，加入了 D 队列优化，无冗余
// __global__ void gen_label_hsdl_v2 (int V, int thread_num, int hop_cst, int hop_now, int* out_pointer, int* out_edge, int* out_edge_weight,
//             cuda_vector_v2<hub_type> *L_gpu, cuda_hashTable_v2<weight_type> *Has, cuda_hashTable_v2<weight_type> *Das,
//             cuda_vector_v2<T_item> *T0, cuda_vector_v2<T_item> *T1, int *d) {
    
//     // 线程id
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
//     if (tid < 0 || tid >= thread_num) {
//         return;
//     }

//     // hash table
//     cuda_hashTable_v2<weight_type> *has = (Has + tid);
//     cuda_hashTable_v2<weight_type> *das = (Das + tid);
//     int d_start = tid * V, d_end = d_start;
//     int block_id, block_siz;

//     for (int node_id = tid; node_id < V; node_id += thread_num) {
        
//         // node_id 的 T 队列
//         cuda_vector_v2<T_item> *t0 = (T0 + node_id);
//         cuda_vector_v2<T_item> *t1 = (T1 + node_id);

//         // node_id 的 label
//         cuda_vector_v2<hub_type> *L = (L_gpu + node_id);

//         // 初始化 hashtable，就是遍历 label 集合并一一修改在 hashtable 中的值
//         int cnt = 0;
//         for (int i = 0; i < L->blocks_num; ++i) {
//             block_id = L->block_idx_array[i];
//             block_siz = L->pool->get_block_size(block_id);
//             for (int j = 0; j < block_siz; ++j) {
//                 hub_type* x = L->pool->get_node(block_id, j);
//                 has->modify(x->hub_vertex, x->hop, hop_cst, x->distance);
//                 cnt ++;
//                 // if (cnt >= L->last_size) break;
//             }
//         }

//         // 遍历 T 队列，并生成 D 队列
//         for (int i = 0; i < t0->blocks_num; ++i) {
//             block_id = t0->block_idx_array[i];
//             block_siz = t0->pool->get_block_size(block_id);
//             for (int j = 0; j < block_siz; ++j) {

//                 // 获取 T 队列元素
//                 T_item *x = t0->pool->get_node(block_id, j);

//                 // sv 为起点, ev 为遍历到的点, dis 为距离，hop 为跳数
//                 int sv = node_id, ev = x->vertex, h = hop_now;
//                 weight_type dis = x->distance;

//                 // 遍历节点 ev 并扩展
//                 for (int k = out_pointer[ev]; k < out_pointer[ev + 1]; ++k) {
//                     int v = out_edge[k];
                    
//                     // rank pruning，并且同一个点也不能算。
//                     if (sv >= v) continue;

//                     // h 为现在这些标签的跳数， h + 1为现在要添加的标签跳数
//                     int dv = dis + out_edge_weight[k];

//                     // 判断生成 D 队列
//                     weight_type d_hash = das->get(v);
//                     if (d_hash == 1e9) {
//                         d[d_end ++] = v;
//                         das->modify(v, dv);
//                     }else{
//                         if (d_hash > dv) {
//                             das->modify(v, dv);
//                         }
//                     }

//                 }
//             }
//         }

//         // 遍历 D 队列
//         for (int i = d_start; i < d_end; ++i) {
//             int sv = node_id, v = d[i], h = hop_now;
//             weight_type dv = das->get(v);
//             weight_type q_dis = query_dis_by_hash_table(sv, v, has, L_gpu + v, h + 1, hop_cst);
            
//             if (dv < q_dis) {
//                 // 添加标签并压入 T 队列
//                 L_gpu[v].push_back({sv, h + 1, dv});
//                 t1->push_back({v, dv});
//             }

//             das->modify(v, 1e9);
//         }

//         // 改回 hashtable
//         cnt = 0;
//         for (int i = 0; i < L->blocks_num; ++i) {
//             block_id = L->block_idx_array[i];
//             block_siz = L->pool->get_block_size(block_id);
//             for (int j = 0; j < block_siz; ++j) {
//                 hub_type* x = L->pool->get_node(block_id, j);
//                 has->modify(x->hub_vertex, x->hop, hop_cst, 1e9);
//                 cnt ++;
//                 // if (cnt >= L->last_size) break;
//             }
//         }
//     }
// }

// // 索引生成过程_v3，加入了 D 队列优化，实现了 D 队列遍历的并行，无冗余
// __global__ void gen_label_hsdl_v3 (int V, int thread_num, int hop_cst, int hop_now, int* out_pointer, int* out_edge, int* out_edge_weight,
//             cuda_vector_v2<hub_type> *L_gpu, cuda_hashTable_v2<weight_type> *Has, cuda_hashTable_v2<weight_type> *Das,
//             cuda_vector_v2<T_item> *T0, cuda_vector_v2<T_item> *T1, int *d,
//             clock_t *timer_hash1, clock_t *timer_gett, clock_t *timer_query, clock_t *timer_hash2) {
    
//     // 线程id
//     int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
//     if (tid < 0 || tid >= thread_num) {
//         return;
//     }

//     // hash table
//     cuda_hashTable_v2<weight_type> *has = (Has + tid);
//     cuda_hashTable_v2<weight_type> *das = (Das + tid);
//     int d_start, d_end;
//     int block_id, block_siz;
    
//     for (int node_id = tid; node_id < V; node_id += thread_num) {
        
//         d_start = tid * V;
//         d_end = d_start;

//         // node_id 的 T 队列
//         cuda_vector_v2<T_item> *t0 = (T0 + node_id);
//         cuda_vector_v2<T_item> *t1 = (T1 + node_id);

//         // node_id 的 label
//         cuda_vector_v2<hub_type> *L = (L_gpu + node_id);
        
//         timer_hash1[tid] += clock() / 1000;
//         // 初始化 hashtable，就是遍历 label 集合并一一修改在 hashtable 中的值
//         int cnt = 0;
//         for (int i = 0; i < L->blocks_num; ++i) {
//             block_id = L->block_idx_array[i];
//             block_siz = L->pool->get_block_size(block_id);
//             for (int j = 0; j < block_siz; ++j) {
//                 hub_type *x = &(L->pool->blocks_pool[block_id].data[j]);
//                 has->modify(x->hub_vertex, x->hop, hop_cst, x->distance);
//                 cnt ++;
//                 if (cnt >= L->last_size) break;
//             }
//         }
//         timer_hash1[tid + thread_num] += clock() / 1000;

//         timer_gett[tid] += clock() / 1000;
//         // 遍历 T 队列，并生成 D 队列
//         for (int i = 0; i < t0->blocks_num; ++i) {
//             block_id = t0->block_idx_array[i];
//             block_siz = t0->pool->get_block_size(block_id);
//             for (int j = 0; j < block_siz; ++j) {

//                 // 获取 T 队列元素
//                 // T_item *x = t0->pool->get_node(block_id, j);
//                 T_item *x = &(t0->pool->blocks_pool[block_id].data[j]);

//                 // sv 为起点, ev 为遍历到的点, dis 为距离，hop 为跳数
//                 int sv = node_id, ev = x->vertex, h = hop_now;
//                 weight_type dis = x->distance;

//                 // 遍历节点 ev 并扩展
//                 for (int k = out_pointer[ev]; k < out_pointer[ev + 1]; ++k) {
//                     int v = out_edge[k];
                    
//                     // rank pruning，并且同一个点也不能算。
//                     if (sv >= v) continue;

//                     // h 为现在这些标签的跳数， h + 1为现在要添加的标签跳数
//                     int dv = dis + out_edge_weight[k];

//                     // 判断生成 D 队列
//                     weight_type d_hash = das->get(v);
//                     if (d_hash == 1e9) {
//                         d[d_end ++] = v;
//                         das->modify(v, dv);
//                     }else{
//                         if (d_hash > dv) {
//                             das->modify(v, dv);
//                         }
//                     }

//                 }
//             }
//         }
//         timer_gett[tid + thread_num] += clock() / 1000;

//         // u, v, d_size, hash, label, t, hop, hop_cst;
//         timer_query[tid] += clock() / 1000;
//         query_parallel <<< (d_end - d_start + 1023) / 1024, 1024 >>>
//         (node_id, d_start, d_end, d_end - d_start, das, &d[0], has, L_gpu, t1, hop_now, hop_cst);
//         cudaDeviceSynchronize();
//         timer_query[tid + thread_num] += clock() / 1000;

//         // 改回 hashtable
//         timer_hash2[tid] += clock() / 1000;
//         cnt = 0;
//         for (int i = 0; i < L->blocks_num; ++i) {
//             block_id = L->block_idx_array[i];
//             block_siz = L->pool->get_block_size(block_id);
//             for (int j = 0; j < block_siz; ++j) {
//                 hub_type *x = &(L->pool->blocks_pool[block_id].data[j]);
//                 has->modify(x->hub_vertex, x->hop, hop_cst, 1e9);
//                 cnt ++;
//                 if (cnt >= L->last_size) break;
//             }
//         }
//         timer_hash2[tid + thread_num] += clock() / 1000;
//     }
    
// }

// __global__ void add_timer (clock_t* tot, clock_t *t, int thread_num) {
//     for (int i = 0; i < thread_num; ++i) {
//         (*tot) += (t[i + thread_num] - t[i]) / 1000;
//     }
//     printf("t: %lld\n", (long long)(*tot));
// }

// // 生成 label 的过程
// void label_gen (CSR_graph<weight_type>& input_graph, hop_constrained_case_info_v2 *info, int hop_cst, vector<vector<hub_type> >&L) {

//     int V = input_graph.OUTs_Neighbor_start_pointers.size() - 1;
//     int E = input_graph.OUTs_Edges.size();
//     int* out_edge = input_graph.out_edge;
//     int* out_edge_weight = input_graph.out_edge_weight;
//     int* out_pointer = input_graph.out_pointer;

//     int thread_num = 1000;

//     int dimGrid = 1, dimGrid_V, dimBlock = 1;
//     dimGrid_V = (V + dimBlock - 1) / dimBlock;
//     dimGrid = (thread_num + dimBlock - 1) / dimBlock;

//     printf("V, E: %d, %d\n", V, E);

//     // 准备 info
//     info->init(V, V * V * (hop_cst + 1), hop_cst);
//     printf("init case_info success\n");

//     // 准备 L_hashTable
//     cuda_hashTable_v2<weight_type> *L_hash;
//     cudaMallocManaged(&L_hash, thread_num * sizeof(cuda_hashTable_v2<weight_type>));
//     cudaDeviceSynchronize();
//     for (int i = 0; i < thread_num; i++) {
//         new (L_hash + i) cuda_hashTable_v2 <weight_type> (V * (hop_cst + 1));
//     }

//     // 准备 D_hashTable
//     cuda_hashTable_v2<weight_type> *D_hash;
//     cudaMallocManaged(&D_hash, thread_num * sizeof(cuda_hashTable_v2<weight_type>));
//     cudaDeviceSynchronize();
//     for (int i = 0; i < thread_num; i++) {
//         new (D_hash + i) cuda_hashTable_v2 <weight_type> (V);
//     }
//     printf("init hash_table success\n");

//     // 准备 D_vector
//     int *D_vector;
//     cudaMallocManaged(&D_vector, thread_num * V * sizeof(int));

//     // 编号越小的点，rank 越高
//     // for (int i = 0; i < V; i ++){
//     //     printf("degree %d, %d\n", i, out_pointer[i + 1] - out_pointer[i]);
//     // }

//     // 同步，保证数据初始化完成
//     cudaDeviceSynchronize();

//     // 计时 label gen 中的耗时
//     clock_t *timer_hash1, *timer_gett, *timer_query, *timer_hash2;
//     clock_t *timer_hash1_tot, *timer_gett_tot, *timer_query_tot, *timer_hash2_tot;
//     cudaMallocManaged(&timer_hash1, thread_num * 2 * sizeof(clock_t));
//     cudaMallocManaged(&timer_hash1_tot, sizeof(clock_t));
//     cudaMallocManaged(&timer_gett, thread_num * 2 * sizeof(clock_t));
//     cudaMallocManaged(&timer_gett_tot, sizeof(clock_t));
//     cudaMallocManaged(&timer_query, thread_num * 2 * sizeof(clock_t));
//     cudaMallocManaged(&timer_query_tot, sizeof(clock_t));
//     cudaMallocManaged(&timer_hash2, thread_num * 2 * sizeof(clock_t));
//     cudaMallocManaged(&timer_hash2_tot, sizeof(clock_t));
    
//     // 测试 cuda_vector 和 cuda_hash 的部分
//     // test_mmpool(V, thread_num, 2, info, L_hash);
    
//     init_T <<<dimGrid_V, dimBlock>>> (V, info->T0, info->L_cuda);
//     cudaDeviceSynchronize();

//     // 计时
//     cudaEvent_t start, stop;
//     float elapsedTime = 0.0;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     cudaEventRecord(start, 0);
    
//     // 辅助变量，不方便直接探测 T 是否为空
//     int iter = 0;

//     while (1) {

//         if (iter++ >= hop_cst) break;
//         // iter = 1 -> 生成 跳数 1
//         // iter = 2 -> 生成 跳数 2
//         // iter = 3 -> 生成 跳数 3
//         // iter = 4 -> 生成 跳数 4
//         // iter = 5 -> 生成 跳数 5
        
//         printf("iteration_hop: %d\n", iter);

//         // 根据奇偶性，轮流使用 T0、T1，不需要交换指针
//         if (iter % 2 == 1) {
//             // gen_label_hsdl <<<dimGrid, dimBlock>>> (V, thread_num, hop_cst, iter - 1, out_pointer, out_edge, out_edge_weight,
//             // info->L_cuda, L_hash, info->T0, info->T1);
//             // gen_label_hsdl_v2 <<< dimGrid, dimBlock >>> (V, thread_num, hop_cst, iter - 1, out_pointer, out_edge, out_edge_weight,
//             // info->L_cuda, L_hash, D_hash, info->T0, info->T1, D_vector);
//             gen_label_hsdl_v3 <<< dimGrid, dimBlock >>> (V, thread_num, hop_cst, iter - 1, out_pointer, out_edge, out_edge_weight,
//             info->L_cuda, L_hash, D_hash, info->T0, info->T1, D_vector, timer_hash1, timer_gett, timer_query, timer_hash2);
//             cudaDeviceSynchronize();

//             // 清洗 T 数组
//             clear_T <<< dimGrid_V, dimBlock >>> (V, info->T0, info->L_cuda);
//         }else{
//             // gen_label_hsdl <<<dimGrid, dimBlock>>> (V, thread_num, hop_cst, iter - 1, out_pointer, out_edge, out_edge_weight,
//             // info->L_cuda, L_hash, info->T1, info->T0);
//             // gen_label_hsdl_v2 <<< dimGrid, dimBlock >>> (V, thread_num, hop_cst, iter - 1, out_pointer, out_edge, out_edge_weight,
//             // info->L_cuda, L_hash, D_hash, info->T1, info->T0, D_vector);
//             gen_label_hsdl_v3 <<< dimGrid, dimBlock >>> (V, thread_num, hop_cst, iter - 1, out_pointer, out_edge, out_edge_weight,
//             info->L_cuda, L_hash, D_hash, info->T1, info->T0, D_vector, timer_hash1, timer_gett, timer_query, timer_hash2);
//             cudaDeviceSynchronize();

//             // 清洗 T 数组
//             clear_T <<< dimGrid_V, dimBlock >>> (V, info->T1, info->L_cuda);
//         }
//         cudaDeviceSynchronize();
//         cudaEventRecord(stop, 0);
//         cudaEventSynchronize(stop);
//         cudaEventElapsedTime(&elapsedTime, start, stop);
//         printf("Time generation in hop %d : %.8lf s\n", iter, elapsedTime / 1000.0);

//     }
//     cudaError_t err;
//     err = cudaGetLastError(); // 检查内核内存申请错误
//     if (err != cudaSuccess) {
//         printf("!INIT CUDA ERROR: %s\n", cudaGetErrorString(err));
//     }
    
//     // timer record
//     cudaEventRecord(stop, 0);
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&elapsedTime, start, stop);
//     printf("Time generation: %.8f s\n", elapsedTime / 1000.0);
//     info->time_generate_labels = elapsedTime / 1000.0;
//     cudaEventDestroy(start);
//     cudaEventDestroy(stop);

//     add_timer <<< 1, 1 >>> (timer_hash1_tot, timer_hash1, thread_num);
//     add_timer <<< 1, 1 >>> (timer_gett_tot, timer_gett, thread_num);
//     add_timer <<< 1, 1 >>> (timer_query_tot, timer_query, thread_num);
//     add_timer <<< 1, 1 >>> (timer_hash2_tot, timer_hash2, thread_num);

//     // printf("hub, parent, hop, dis:\n");
//     auto beforeTime = std::chrono::steady_clock::now();
//     info->label_size = 0;
//     for (int v = 0; v < V; ++v) {
//         L[v].clear();
//         // printf("vertex %d\n", v);
//         for (int i = 0; i < info->L_cuda[v].blocks_num; ++i) {
//             int block_id = info->L_cuda[v].block_idx_array[i];
//             int block_siz = info->L_cuda[v].pool->get_block_size_host(block_id);
//             for (int j = 0; j < block_siz; ++j) {
//                 hub_type* x = info->L_cuda[v].pool->get_node_host(block_id, j);
//                 // printf("{%d, %d, %d, %d}, ", x->hub_vertex, x->parent_vertex, x->hop, x->distance);
//                 L[v].push_back({x->hub_vertex, x->hop, x->distance});
//                 // info->L_cpu[v].push_back({x->hub_vertex, x->hop, x->distance});
//                 info->label_size ++;
//             }
//         }
//         // printf("\n");
//     }
//     auto afterTime = std::chrono::steady_clock::now();
//     printf("time traverse labels: %.8lf\n", std::chrono::duration<double>(afterTime - beforeTime).count());

//     info->label_size = info->label_size / (double)V;
//     printf("average label size: %.8lf\n", info->label_size);
//     printf("Generation end!\n");

//     info->destroy_L_cuda();
    
//     for(int i = 0; i < thread_num; ++i){
//         L_hash[i].~cuda_hashTable_v2();
//     }
//     cudaFree(L_hash);
    
//     return;
// }