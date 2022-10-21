import networkx as nx
import matplotlib.pyplot as plt
import os
import heapq
import util, dijkstra, heap

# dataset_id = "001"
dataset_id = "002"

ALPHA = 0.7
NODE_SIZE = 500
FIGURE_SHOW = False

if __name__ == "__main__":
    print(f"use dataset{dataset_id}")
    # 読み込むデータの設定 #
    data_dir_name = os.path.join("data", "dataset"+dataset_id)
    graph_file_name = "graph" + dataset_id + ".txt"
    pos_file_name = "pos" + dataset_id + ".txt"
    srcdst_file_name = "srcdst" + dataset_id + ".txt"

    # Graphの読み込み #
    G = nx.read_weighted_edgelist(os.path.join(data_dir_name, graph_file_name), nodetype=str)
    pos = util.read_pos(os.path.join(data_dir_name, pos_file_name))
    src, dst = util.read_src_dst(os.path.join(data_dir_name, srcdst_file_name))
    
    # Original Graph 描画
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f"Original Graph({dataset_id})")
    edge_labels = {(i, j): w['weight'] for i, j, w in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G,pos, edge_labels=edge_labels)
    nx.draw_networkx(G, pos, with_labels=True, alpha=ALPHA, node_size=NODE_SIZE)
    plt.savefig(os.path.join(data_dir_name, "out", "OriginalGraph"+dataset_id))
    if FIGURE_SHOW:
        plt.show()

    # shortest path tree
    pred, distance = nx.dijkstra_predecessor_and_distance(G, source=dst)
    T = G.edge_subgraph((i, pred[i][0]) for i in pred if pred[i])

    # shortest path tree 描画 #
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f"Shortest Path Graph({dataset_id})")
    edge_labels = {(i, j): w['weight'] for i, j, w in T.edges(data=True)}
    nx.draw_networkx_edge_labels(T, pos, edge_labels=edge_labels)
    nx.draw_networkx(T, pos, with_labels=True, alpha=ALPHA, node_size=NODE_SIZE)
    plt.savefig(os.path.join(data_dir_name, "out", "ShortestPathTree"+dataset_id))
    if FIGURE_SHOW:
        plt.show()

    # Potential計算 #
    for edge in G.edges():
        tail, head = edge
        G[tail][head]['delta'] = G[tail][head]['weight'] + distance[head] - distance[tail]

    # Potential 描画 #
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f"Potentil Graph({dataset_id})")
    nx.draw_networkx_nodes(G, pos, alpha=ALPHA, node_size=NODE_SIZE) # node描画
    for e in G.edges():
        if e not in T.edges():
            # Gの辺の中で，Tに含まれる辺について
            nx.draw_networkx_edges(G, pos, edgelist=[e]) # edge描画
            nx.draw_networkx_labels(G, pos, labels=None)
            nx.draw_networkx_edge_labels(G, pos, edge_labels={edge: G.edges[edge]["delta"] for edge in G.edges() if edge not in T.edges()})
    fig.savefig(os.path.join(data_dir_name, "out", "PotentialGraph"+dataset_id))
    if FIGURE_SHOW:
        plt.show()
    
    # sidetrack関数の使い方 #
    # print(util.sidetrack("A", G, T))
    # print(util.sidetrack("B", G, T))
    # print(util.sidetrack("C", G, T))
    # print(util.sidetrack("D", G, T))

    # # dijkstra.uni_methodの使い方 #
    # print("uni_method")
    # length, path, S = dijkstra.uni_method(G, src, dst)
    # print(f"length: {length}")
    # print(f"path: {path}")
    # print(f"S: {S}")
    # print("")

    # # dijkstra.bib_methodの使い方
    # print("bib_method")
    # length, path, (S, T) = dijkstra.bib_method(G, src, dst)
    # print(f"length: {length}")
    # print(f"path: {path}")
    # print(f"S: {S}")
    # print(f"T: {T}")
    # print("")

    