import redis
import networkx as nx
import community as community_louvain

def read_graph_from_redis(r, key):
    edges = []
    # 假设边以Redis List存储，每个元素为"起点 终点 权重"
    edges_data = r.lrange(key, 0, -1)
    for edge_data in edges_data:
        u, v, w = map(int, edge_data.decode('utf-8').split())
        edges.append((u, v, w))
    return edges

def perform_louvain_clustering(edges):
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    partition = community_louvain.best_partition(G, resolution=1, random_state=105, weight='weight')
    return partition

def write_partition_to_redis(r, partition, key):
    # 以Hash形式存储社区划分结果，键为节点，值为社区ID
    for node, community in partition.items():
        r.hset(key, node, community)
    print('Partition result has been written to Redis.')

def main():
    r = redis.Redis(host='localhost', port=6379, db=0)
    input_key = 'graph_edges'  # 存储边信息的Key
    output_key = 'partition_result'  # 存储社区划分结果的Key
    edges = read_graph_from_redis(r, input_key)
    if(len(edges) == 0):
        print('No edges found in Redis.Please import graph data first.\n')
        return
    partition = perform_louvain_clustering(edges)
    write_partition_to_redis(r, partition, output_key)

if __name__ == "__main__":
    main()
