import networkx as nx
import matplotlib.pyplot as plt
import os
import util, dijkstra, heap #自作
from heapq import heappush, heappop
from copy import deepcopy
from collections import deque

# dataset_id = "001"
# dataset_id = "002"
dataset_id = "003"

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
    edge_labels = {(i, j): w['weight'] for i, j, w in Q.edges(data=True)}
    nx.draw_networkx_edge_labels(Q, p_pos, edge_labels=edge_labels)
    nx.draw_networkx(Q, p_pos, with_labels=True, alpha=ALPHA, node_size=NODE_SIZE*0.5)
    plt.savefig(os.path.join(data_dir_name, "out", "PathGraph"+dataset_id))
    if FIGURE_SHOW:
        plt.show()

    # ovの計算
    print("--- OW cal---")
    popt = nx.shortest_path(T, source=src, target=dst, weight="weight")
    popt_edges = util.nodetype2edgetype(popt) # pathの表現形式をedgetypeに
    edges = []
    OW = {dst: 0.0}
    for t, h in nx.dfs_edges(T.reverse(), source=dst):
        print(f"node: {h}")
        print(f"    edges: {edges[::-1]}")
        print(f"    edge: {(h, t)}")
        changed = False
        for i, (_h, _t) in enumerate(edges):
            if t == _h:
                edges = edges[:(i+1)] + [(h, t)]
                changed = True
        if not changed:
            edges = [(h, t)]
            # edges.append((h, t))
        print(f"    edges: {edges[::-1]}")
        OW[h] = 0.0
        for e in edges[::-1]:
            if e in popt_edges:
                print(f"    {e}")
                OW[h] += G[e[0]][e[1]]["weight"]

    for v, val in sorted(OW.items()):
        print(v, val)

    minus_weight = -sum(abs(T.edges[edge]["weight"]) for edge in T.edges())
    A = [] #  パス格納
    B = [(0, ["Root"])] # 幅優先用キュー
    OVERLAP = {} #overlap(sidetrack)の計算
    k_path = []
    k = 0
    NUM = 10

    print("---find k-shortest path---")
    while B and k < NUM:
        print(f"k:{k}")
        k_path.append({}) # k番目のsidetrackについて情報を管理する辞書の初期化
        potentials, stracks = heappop(B)
        # print("stracks", stracks)
        print(f"sidetrack: {stracks[-1]} について")
        k_path[k]["sidetrack"] = stracks[-1]
        if len(OVERLAP) == 0:
            OVERLAP["Root"] = OW[src]
            k_path[k]["overlap"] = OVERLAP["Root"]
        else:
            d = OW[stracks[-1][1][1]] - OW[stracks[-1][1][0]]
            print(f"OVERLAP[{util.get_parent(stracks, popt)}]: {OVERLAP[util.get_parent(stracks, popt)]}")
            print(f"d = ow({stracks[-1][1][1]}) - ow({stracks[-1][1][0]}) = {OW[stracks[-1][1][1]]} - {OW[stracks[-1][1][0]]} = {d}")
            # print(f"d: {d}")
            # C.append(C[-1] + d)
            # print(f"key {util.get_parent(stracks, popt)} のoverlapを利用したい")
            OVERLAP[stracks[-1]] = OVERLAP[util.get_parent(stracks, popt)] + d
            k_path[k]["overlap"] = OVERLAP[stracks[-1]]
            # print(f"key {stracks[-1]} に 追加")
            # C.append()
            # print(util.get_parent(stracks, popt))
        # print("C", C)

        tmp_T = T.copy()
        for parent, child in zip(stracks[1:], stracks[2:]):
            if P[parent][child]["color"] == "red":
                # *parent[1]は tuple("G", "J")をアンパックして， G Jにしている．下と同じ意味
                tmp_T.add_edge(*parent[1], weight=minus_weight)
                # h = parent[1][0]
                # t = parent[1][1]
                # tmp_T.add_edge(h, t, weight=minus_weight)
        if len(stracks) > 1:
            last_stracks = stracks[-1]
            tmp_T.add_edge(*last_stracks[1], weight=minus_weight)
        # nx.shortest_path(tmp_T, source=src, target=dst, weight="weight")
        A.append(nx.shortest_path(tmp_T, source=src, target=dst, weight="weight"))    
        # print("Path", A)
        print("Path", A[-1])
        k_path[k]["path"] = A[-1]
        last_stracks = stracks[-1]
        for v in P[last_stracks]:
            new_stracks = stracks + [v]
            new_potentials = potentials + P[last_stracks][v]["weight"]
            heappush(B, (new_potentials, new_stracks))
        k += 1
        print("")

    # 最小のoverlapを求める
    min_overlap = k_path[0]["overlap"]
    for info in k_path:
        if info["overlap"] < min_overlap:
            min_overlap = info["overlap"]

    # overlapからdetourを計算 #
    for info in k_path:
        if info["overlap"] == min_overlap:
            info["detour"] = True
        else:
            info["detour"] = False
    k_path[0]["detour"] = True # Rootはdetour

    for k, info in enumerate(k_path):
        print(f"{k+1}-shortest")
        print(f"    sidetrack: {info['sidetrack']}")
        print(f"    path: {info['path']}")
        print(f"    ovarlap: {info['overlap']}")
    print("")

    # print("---deotur---")
    # for info in k_path:
    #     if info["detour"]:
    #         print(f"    path: {info['path']}")

    print("---deotur---")
    detour_count = 1
    for info in k_path:
        if info["detour"]:
            print(f"    path: {info['path']}")
            edge_list = util.nodetype2edgetype(info["path"]) # pathをedgetypeに変換
            # D = G.edge_subgraph(p) # detourのpathの辺からGのサブグラフを作成
            # edge_list

            fig = plt.figure()
            ax = fig.add_subplot()
            ax.set_title(f"detour[{detour_count}]({dataset_id})")
            # nx.draw_networkx_nodes(D, pos, node_size=NODE_SIZE, alpha=ALPHA)
            # edge_labels = {(i, j): w['weight'] for i, j, w in D.edges(data=True)}
            # nx.draw_networkx_edge_labels(D, pos, edge_labels=edge_labels)
            # nx.draw_networkx(D, pos, with_labels=True, alpha=ALPHA, node_size=NODE_SIZE)
            # Gの描画
            edge_labels = {(i, j): w['weight'] for i, j, w in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            nx.draw_networkx_edges(G, pos, alpha=ALPHA, width=1.0)
            nx.draw_networkx_edges(G, pos, edgelist=edge_list, edge_color="r", alpha=ALPHA*0.5, width=6.0)
            nx.draw_networkx_nodes(G, pos, alpha=ALPHA, node_size=NODE_SIZE)
            nx.draw_networkx(G, pos, with_labels=True, alpha=ALPHA, node_size=NODE_SIZE)
            
            # edge_labels = {(i, j): w['weight'] for i, j, w in D.edges(data=True)}
            # nx.draw_networkx_edge_labels(D, pos, edge_labels=edge_labels)

            plt.savefig(os.path.join(data_dir_name, "out", "detour", f"detour[{detour_count}]"+dataset_id))
            detour_count += 1
            if FIGURE_SHOW:
                plt.show()





    #     pred, distance = nx.dijkstra_predecessor_and_distance(G.reverse(), source=dst)
    # T = G.edge_subgraph((i, pred[i][0]) for i in pred if pred[i])


    # # shortest path tree 描画 #
    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.set_title(f"Shortest Path Graph({dataset_id})")
    # edge_labels = {(i, j): w['weight'] for i, j, w in T.edges(data=True)}
    # nx.draw_networkx_edge_labels(T, pos, edge_labels=edge_labels)
    # nx.draw_networkx(T, pos, with_labels=True, alpha=ALPHA, node_size=NODE_SIZE)
    # plt.savefig(os.path.join(data_dir_name, "out", "ShortestPathTree"+dataset_id))
    # if FIGURE_SHOW:
    #     plt.show()
