#include "label/gen_label.cuh"

// 通过 hashtable 的快速查询
__device__ int query_dis_by_hash_table (int u, int v, cuda_hashTable_v2<int> *H, cuda_vector_v2<hub_type> *L, int hop_now, int hop_cst) {
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

// 初始化 T
__global__ void init_T (int V, cuda_vector_v2<hub_type> *T, cuda_vector_v2<hub_type> *L_gpu) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < V) {
        // start_vertex, target_vertex, hop, distance
        T[tid].push_back(tid, {tid, tid, 0, 0});
        L_gpu[tid].push_back(tid, {tid, tid, 0, 0});
    }
}

// 清空 T
__global__ void clear_T (int V, cuda_vector_v2<hub_type> *T) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < V) {
        T[tid].init(V, tid);
    }
}

// 索引生成过程
__global__ void gen_label_hsdl (int V, int thread_num, int hop_cst, int hop_now, int* out_pointer, int* out_edge, int* out_edge_weight,
            cuda_hashTable_v2<int> *Has, cuda_vector_v2<hub_type> *L_gpu, cuda_vector_v2<hub_type> *T0, cuda_vector_v2<hub_type> *T1) {
    
    // 线程id
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // hash table
    cuda_hashTable_v2<int> *has = (Has + tid);

    for (int node_id = tid; node_id < V; node_id += thread_num) {

        // node_id 的 T 队列
        cuda_vector_v2<hub_type> *t0 = (T0 + node_id);
        cuda_vector_v2<hub_type> *t1 = (T1 + node_id);

        // node_id 的 label
        cuda_vector_v2<hub_type> *L = (L_gpu + node_id);

        // 初始化 hashtable，就是遍历 label集合并一一修改在 hashtable 中的值
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
                hub_type *x = t0->pool->get_node(block_id, j);

                // sv 为起点, ev 为遍历到的点, dis 为距离，hop 为跳数
                int sv = x->hub_vertex, ev = x->parent_vertex;
                int dis = x->distance, h = x->hop;

                // 遍历节点 ev 并扩展
                for (int k = out_pointer[ev]; k < out_pointer[ev + 1]; ++k) {
                    int v = out_edge[k];
                    
                    // rank pruning，并且同一个点也不能算。
                    if (sv >= v) continue;

                    // h 为现在这些标签的跳数， h + 1为现在要添加的标签跳数
                    int dv = dis + out_edge_weight[k];
                    int q_dis = query_dis_by_hash_table(sv, v, Has + tid, L_gpu + v, h + 1, hop_cst);
                    
                    if (dv < q_dis) {
                        // printf("dv q_dis: %d %d\n", dv, q_dis);

                        // 现在的 label 还保留了原本格式后面label和T的元素类型还要再改。
                        L_gpu[v].push_back(0, {sv, ev, h + 1, dv});
                        t1->push_back(0, {sv, v, h + 1, dv});
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

// 生成 label 的过程
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

    // 准备 info
    info = new hop_constrained_case_info_v2();
    info->init(V, V * V * hop_cst, hop_cst);
    printf("init case_info success\n");

    // 准备 hashTable
    cuda_hashTable_v2<int> *L_hash;
    cudaMallocManaged(&L_hash, thread_num * sizeof(cuda_hashTable_v2<int>));
    for (int i = 0; i < thread_num; i++) {
        new (L_hash + i) cuda_hashTable_v2<int> (V * (hop_cst + 1));
    }
    printf("init hash_table success\n");
    
    // 编号越小的点，rank 越高
    // for (int i = 0; i < V; i ++){
    //     printf("degree %d, %d\n", i, out_pointer[i + 1] - out_pointer[i]);
    // }

    // 同步，保证数据初始化完成
    cudaDeviceSynchronize();

    // 测试 cuda_vector 和 cuda_hash 的部分
    // test_mmpool(V, thread_num, 3, info, L_hash)
    
    init_T <<<dimGrid, dimBlock>>> (V, info->T0, info->L_cuda);
    cudaDeviceSynchronize();

    // 计时
    cudaEvent_t start, stop;
    float elapsedTime = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    
    // 辅助变量，不方便直接探测 T 是否为空
    int iter = 0;

    while (1) {

        if (iter++ > hop_cst + 1) break;
        printf("iteration_hop: %d\n", iter);

        // 根据奇偶性，轮流使用 T0、T1，不需要交换指针
        if (iter % 2 == 1) {
            gen_label_hsdl <<<1, thread_num>>> (V, thread_num, hop_cst, iter - 1, out_pointer, out_edge, out_edge_weight, L_hash,
            info->L_cuda, info->T0, info->T1);
            cudaDeviceSynchronize();

            // 清洗 T 数组
            clear_T <<<dimGrid, dimBlock>>> (V, info->T0);
        }else{
            gen_label_hsdl <<<1, thread_num>>> (V, thread_num, hop_cst, iter - 1, out_pointer, out_edge, out_edge_weight, L_hash,
            info->L_cuda, info->T1, info->T0);
            cudaDeviceSynchronize();

            // 清洗 T 数组
            clear_T <<<dimGrid, dimBlock>>> (V, info->T1);
        }
        cudaDeviceSynchronize();

    }
    // cudaError_t err;
    // err = cudaGetLastError(); // 检查内核内存申请错误
    // if (err != cudaSuccess) {
    //     printf("!INIT CUDA ERROR: %s\n", cudaGetErrorString(err));
    // }
    
    // timer record
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("Time generation: %.8f s\n", elapsedTime / 1000.0);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("hub, parent, hop, dis:\n");
    int cnt_label = 0;
    for (int v = 0; v < V; ++v) {
        // printf("vertex %d\n", v);
        for (int i = 0; i < info->L_cuda[v].blocks_num; ++i) {
            int block_id = info->L_cuda[v].block_idx_array[i];
            int block_siz = info->L_cuda[v].pool->get_block_size(block_id);
            for (int j = 0; j < block_siz; ++j) {
                hub_type* x = info->L_cuda[v].pool->get_node(block_id, j);
                // printf("{%d, %d, %d, %d}, ", x->hub_vertex, x->parent_vertex, x->hop, x->distance);
                L[v].push_back({x->hub_vertex, x->parent_vertex, x->hop, x->distance});
                cnt_label ++;
            }
        }
        // printf("\n");
    }
    printf("average label size: %.5lf\n", (double)cnt_label / (double)V);
    printf("Generation end!\n");

    info->destroy_L_cuda();
    
    return;
}