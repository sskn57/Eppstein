import networkx as nx
import matplotlib.pyplot as plt
import os
import util, dijkstra, heap
from copy import deepcopy

dataset_id = "001"
# dataset_id = "002"

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
    G = nx.read_weighted_edgelist(os.path.join(data_dir_name, graph_file_name), nodetype=str, create_using=nx.DiGraph)
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
    pred, distance = nx.dijkstra_predecessor_and_distance(G.reverse(), source=dst)
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

    # H_OUTの生成
    H_OUT = {}
    for v in G:
        # print(f"node {v}")
        h_out = heap.Heap()
        sidetracks = util.sidetrack(v, G, T)
        for (h, t) in sidetracks:
            # print(h, t, G[h][t]["delta"])
            h_out.insert(h, t, G[h][t]["delta"])
        fpath = os.path.join(data_dir_name, "out", "h_out", f"h_out({v}){dataset_id}")
        h_out.show(fpath)
        H_OUT[v] = h_out
    
    print("---H_OUT---")
    for v, h_out in H_OUT.items():
        print(v, h_out.data)

    # H_Tの生成
    H_T = {}
    for v in G.nodes():
        # print(f"---------{v}----")
        H_T[v] = heap.Heap()
    # 経路記憶用
    paths = {}
    paths[dst] = [dst]
    for (t, h) in nx.bfs_edges(T.reverse(), source=dst):
        # print(h, t)
        paths[h] = paths[t] + [h]
    # # 経路出力
    # for k, p in paths.items():
    #     print(k, p)
    for v in G:
        for u in paths[v]:
            if len(H_OUT[u].data) > 0:
                h = H_OUT[u].data[0][1][0]
                t = H_OUT[u].data[0][1][1]
                val = H_OUT[u].data[0][0]
                H_T[v].insert(h, t, val)
    print("---H_T---")
    for v, h_t in H_T.items():
        print(v, h_t.data)
    
    for v, h_t in H_T.items():
        fpath = os.path.join(data_dir_name, "out", "h_t", f"h_t({v}){dataset_id}")
        h_t.show(fpath)
    
    # H_Gの生成（H_OUTとH_Gのマージ）
    H_G = deepcopy(H_T)
    for v, p in paths.items():
        for u in p:
            for _, (val, (h, t)) in enumerate(H_OUT[u].data):
                if (val, (h, t)) not in H_G[v].data:
                    H_G[v].insert(h, t, val)
    
    print("---H_G---")
    for v, h_g in H_G.items():
        print(v, h_g.data)

    for v, h_g in H_G.items():
        fpath = os.path.join(data_dir_name, "out", "h_g", f"h_g({v}){dataset_id}")
        h_g.show(fpath)

    # Path Graphの作成
    P = nx.DiGraph()

    # P(step1): H_GからパスグラフであるヒープPを作成する．
    for v, h_g in H_G.items():
        for i, (parent_val, (parent_h, parent_t)) in enumerate(h_g.data):
            parent_node = (v, (parent_h, parent_t))
            P.add_node(parent_node)
            if 2*(i+1)-1 < len(h_g.data):
                (child_val, (child_h, child_t)) = h_g.data[2*(i+1)-1]
                child_node = (v, (child_h, child_t))
                w = G[child_h][child_t]["delta"] - G[parent_h][parent_t]["delta"]
                P.add_edge(parent_node, child_node, weight=w)
                P[parent_node][child_node]["color"] = "black"
            if 2*(i+1) < len(h_g.data):
                (child_val, (child_h, child_t)) = h_g.data[2*(i+1)]
                child_node = (v, (child_h, child_t))
                w = G[child_h][child_t]["delta"] - G[parent_h][parent_t]["delta"]
                P.add_edge(parent_node, child_node, weight=w)
                P[parent_node][child_node]["color"] = "black"
    
    # P(step1)描画
    p_pos = nx.shell_layout(P)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f"P(step1)({dataset_id})")
    edge_color = [v["color"] for v in P.edges.values()]
    nx.draw_networkx_edges(P, p_pos, edgelist=P.edges(), edge_color=edge_color)
    nx.draw_networkx(P, p_pos, with_labels=True, alpha=ALPHA, node_size=NODE_SIZE*0.5)
    plt.savefig(os.path.join(data_dir_name, "out", "P(step1)"+dataset_id))
    if FIGURE_SHOW:
        plt.show()

    # P(step2): P(step1)の各ノード(v, (h, t))について，そのノードからH_G(t)への辺を追加
    for p_node in list(P):
        _, strack = p_node
        if strack not in T.edges():
            h, t = strack
            if len(H_G[t].data) > 0:
                child = (t, H_G[t].data[0][1])
                w = G[H_G[t].data[0][1][0]][H_G[t].data[0][1][1]]["delta"]
                P.add_edge(p_node, child, weight=w)
                P[p_node][child]["color"] = "red"

    # P(step2)描画
    p_pos = nx.shell_layout(P)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f"P(step2)({dataset_id})")
    edge_color = [v["color"] for v in P.edges.values()]
    nx.draw_networkx_edges(P, p_pos, edgelist=P.edges(), edge_color=edge_color)
    nx.draw_networkx(P, p_pos, with_labels=True, alpha=ALPHA, node_size=NODE_SIZE*0.5)
    plt.savefig(os.path.join(data_dir_name, "out", "P(step2)"+dataset_id))
    if FIGURE_SHOW:
        plt.show()
    
    # P(step3): Rootの追加
    P.add_node("Root")
    child = (src, H_G[src].data[0][1])
    w = G[H_G[src].data[0][1][0]][H_G[src].data[0][1][1]]["delta"]
    P.add_edge("Root", child, weight=w)
    P["Root"][child]["color"] = "green"

    # P(step3)描画
    p_pos = nx.shell_layout(P)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f"P(step3)({dataset_id})")
    edge_color = [v["color"] for v in P.edges.values()]
    nx.draw_networkx_edges(P, p_pos, edgelist=P.edges(), edge_color=edge_color)
    nx.draw_networkx(P, p_pos, with_labels=True, alpha=ALPHA, node_size=NODE_SIZE*0.5)
    plt.savefig(os.path.join(data_dir_name, "out", "P(step3)"+dataset_id))
    if FIGURE_SHOW:
        plt.show()

    # P(step3)の冗長部分を削除し，出力
    tmp = set()
    for e in nx.bfs_edges(P, source="Root"):
        tmp.add(e)
    Q = P.edge_subgraph(tmp)
    p_pos = nx.shell_layout(P)
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_title(f"PathGraph({dataset_id})")
    edge_color = [v["color"] for v in Q.edges.values()]
    nx.draw_networkx_edges(Q, p_pos, edgelist=Q.edges(), edge_color=edge_color)
    nx.draw_networkx(Q, p_pos, with_labels=True, alpha=ALPHA, node_size=NODE_SIZE*0.5)
    plt.savefig(os.path.join(data_dir_name, "out", "PathGraph"+dataset_id))
    if FIGURE_SHOW:
        plt.show()