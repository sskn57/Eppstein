import networkx as nx
import matplotlib.pyplot as plt
import os
import heapq
import util, dijkstra, heap
from copy import deepcopy
from collections import deque

dataset_id = "001"
# dataset_id = "002"
# dataset_id = "003"
threshold = 3.0
ALPHA = 0.7
FIGSIZE = (10, 10)
FIGSIZE_PathGraph = (30, 30)
NODE_SIZE = 1000
FIGURE_SHOW = False

if __name__ == "__main__":
    print(f"use dataset{dataset_id}")
    # 読み込むデータの設定 #
    data_dir_name = os.path.join("bib", "data", "dataset"+dataset_id)
    graph_file_name = "graph" + dataset_id + ".txt"
    pos_file_name = "pos" + dataset_id + ".txt"
    srcdst_file_name = "srcdst" + dataset_id + ".txt"

    # Graphの読み込み #
    G = nx.read_weighted_edgelist(os.path.join(data_dir_name, graph_file_name), nodetype=str)
    pos = util.read_pos(os.path.join(data_dir_name, pos_file_name))
    src, dst = util.read_src_dst(os.path.join(data_dir_name, srcdst_file_name))

    # 出力用ディレクトリ生成
    if not os.path.exists(os.path.join(data_dir_name, "out")):
        os.mkdir(os.path.join(data_dir_name, "out"))
    if not os.path.exists(os.path.join(data_dir_name, "out", "fromsrc")):
        os.mkdir(os.path.join(data_dir_name, "out", "fromsrc"))
    if not os.path.exists(os.path.join(data_dir_name, "out", "fromdst")):
        os.mkdir(os.path.join(data_dir_name, "out", "fromdst"))

    # Original Graph 描画
    fig = plt.figure(figsize=FIGSIZE)
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
    SPT = G.edge_subgraph((i, pred[i][0]) for i in pred if pred[i])
    # Potential計算 #
    for edge in G.edges():
        tail, head = edge
        # 辺の重みを変更
        hs = G[tail][head]['weight'] + distance[head] - distance[tail]
        ht = G[tail][head]['weight'] + distance[tail] - distance[head]
        G[tail][head]['weight_dash'] = (hs + ht) / 2
        print(f"({tail}, {head}): {G[tail][head]['weight_dash']}")
    
    # 双方向ダイクストラ法
    print("bib_method")
    length, path, (S, T), F, U, (ps, pt) = dijkstra.bib_method(G, src, dst, threshold=threshold, weight_label="weight_dash")

    print(f"    length: {length}") # 最短経路距離
    print(f"    path: {path}") # 最短経路
    print(f"    S: {S}") # srcから探索した点集合
    print(f"    T: {T}") # dstから探索した点集合
    print(f"    F: {F}") # 追加する辺集合
    print(f"    U: {U}") # 追加する点集合

    # Hmidの作成 #
    Hmid = heap.Heap_Node()
    for u in U:
        qu = ps[u] + pt[u]
        Hmid.insert(u, qu)
    # Hmid出力 #
    Hmid.show(os.path.join(data_dir_name, "out", "Hmid"))

    # Gsの作成 #
    Gs = nx.DiGraph()
    for (u, v) in G.edges():
        if u in S:
            if v in S or v in U:
                Gs.add_edge(u, v, weight=G[u][v]["weight_dash"])
        if v in S:
            if u in S or u in U:
                Gs.add_edge(v, u, weight=G[v][u]["weight_dash"])
    # Gsの出力 #
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot()
    ax.set_title(f"Gs({dataset_id})")
    nx.draw_networkx_nodes(G, pos, alpha=ALPHA, node_size=NODE_SIZE) # for G
    nx.draw_networkx_labels(G, pos, alpha=ALPHA) # for G
    edge_labels = {(i, j): w['weight'] for i, j, w in Gs.edges(data=True)}
    nx.draw_networkx_edge_labels(Gs, pos, edge_labels=edge_labels)
    nx.draw_networkx(Gs, pos, with_labels=True, alpha=ALPHA, node_size=NODE_SIZE)
    plt.savefig(os.path.join(data_dir_name, "out", "fromsrc", "Gs"+dataset_id))
    if FIGURE_SHOW:
        plt.show()

    Gs_reverse = Gs.reverse()
    # Gs_reverseの出力 #
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot()
    ax.set_title(f"Gs_reverse({dataset_id})")
    nx.draw_networkx_nodes(G, pos, alpha=ALPHA, node_size=NODE_SIZE) # for G
    nx.draw_networkx_labels(G, pos, alpha=ALPHA) # for G
    edge_labels = {(i, j): w['weight'] for i, j, w in Gs_reverse.edges(data=True)}
    nx.draw_networkx_edge_labels(Gs_reverse, pos, edge_labels=edge_labels)
    nx.draw_networkx(Gs_reverse, pos, with_labels=True, alpha=ALPHA, node_size=NODE_SIZE)
    plt.savefig(os.path.join(data_dir_name, "out", "fromsrc", "Gs_reverse"+dataset_id))
    if FIGURE_SHOW:
        plt.show()


    # Gsにおけるsrcからの最短経路木
    pred, distance = nx.dijkstra_predecessor_and_distance(Gs, source=src)
    # SPT_s = Gs.edge_subgraph((pred[i][0], i) for i in pred if pred[i])    
    SPT_s = Gs_reverse.edge_subgraph((i, pred[i][0]) for i in pred if pred[i])    
    print("SPT_s", SPT_s.edges())

    # Gsにおけるsrcからの最短経路木 出力
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot()
    ax.set_title(f"Shortest Path Tree from {src}({dataset_id})")
    nx.draw_networkx_nodes(G, pos, alpha=ALPHA, node_size=NODE_SIZE) # for G
    nx.draw_networkx_labels(G, pos, alpha=ALPHA) # for G
    edge_labels = {(i, j): w['weight'] for i, j, w in SPT_s.edges(data=True)}
    nx.draw_networkx_edge_labels(SPT_s, pos, edge_labels=edge_labels)
    nx.draw_networkx(SPT_s, pos, with_labels=True, alpha=ALPHA, node_size=NODE_SIZE)
    plt.savefig(os.path.join(data_dir_name, "out", "fromsrc", "ShortestPathTree_from_src"+dataset_id))
    if FIGURE_SHOW:
        plt.show()

    # Gs内におけるsidetrack 描画 #
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot()
    ax.set_title(f"SideTrack in Gs_reverse({dataset_id})")
    nx.draw_networkx_nodes(G, pos, alpha=ALPHA, node_size=NODE_SIZE) # for G
    nx.draw_networkx_labels(G, pos, alpha=ALPHA) # for G
    edge_labels = {(i, j): w['weight'] for i, j, w in Gs.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    sidetrack_edge_list = []
    for (u, v) in Gs_reverse.edges():
        if (u, v) not in SPT_s.edges():
            if u in path and v in path:
                continue
            sidetrack_edge_list.append((u, v))
    nx.draw_networkx_edges(Gs_reverse, pos, edgelist=sidetrack_edge_list, edge_color="green", alpha=ALPHA*0.5, width=6.0)
    nx.draw_networkx(Gs_reverse, pos, with_labels=True, alpha=ALPHA, node_size=NODE_SIZE)
    fig.savefig(os.path.join(data_dir_name, "out", "fromsrc", "Sidetrack_in_Gs_reverse"+dataset_id))
    if FIGURE_SHOW:
        plt.show()

    # Hs_OUTの生成
    if not os.path.exists(os.path.join(data_dir_name, "out", "fromsrc", "Hs_OUT")):
        os.mkdir(os.path.join(data_dir_name, "out", "fromsrc", "Hs_OUT"))
    Hs_OUT = {}
    # print(SPT_s)
    for v in Gs_reverse:
        hs_out = heap.Heap_Edge()
        sidetracks = util.sidetrack(v, Gs_reverse, SPT_s)
        if len(sidetracks) > 0:
            for (h, t) in sidetracks:
                if h in path and t in path:
                    continue
                hs_out.insert(h, t, Gs_reverse[h][t]["weight"])
                hs_out.show(os.path.join(data_dir_name, "out", "fromsrc", "Hs_OUT", f"hs_out({v}){dataset_id}"))
                Hs_OUT[v] = hs_out
        else:
            Hs_OUT[v] = heap.Heap_Edge()

    # Hs_Tの生成
    if not os.path.exists(os.path.join(data_dir_name, "out", "fromsrc", "Hs_T")):
        os.mkdir(os.path.join(data_dir_name, "out", "fromsrc", "Hs_T"))
    Hs_T = {}
    paths = {src: [src]}
    for v in Gs_reverse.nodes():
        Hs_T[v] = heap.Heap_Edge()
        paths[v] = []
    for (t, h) in nx.bfs_edges(SPT_s.reverse(), source=src):
        paths[h] = paths[t] + [h]
    print(paths)
    for v in Gs_reverse:
        for u in paths[v]:
            if len(Hs_OUT[u].data) > 0:
                val, (h, t) = Hs_OUT[u].data[0]
                Hs_T[v].insert(h, t, val)
    # Hs_Tの出力
    for v, hs_t in Hs_T.items():
        hs_t.show(os.path.join(data_dir_name, "out", "fromsrc", "Hs_T", f"hs_t({v}){dataset_id}"))

    # Hs_Gの生成
    if not os.path.exists(os.path.join(data_dir_name, "out", "fromsrc", "Hs_G")):
        os.mkdir(os.path.join(data_dir_name, "out", "fromsrc", "Hs_G"))
    Hs_G = deepcopy(Hs_T)
    for v, p in paths.items():
        for u in p:
            for _, (val, (h, t)) in enumerate(Hs_OUT[u].data):
                if (val, (h, t)) not in Hs_G[v].data:
                    Hs_G[v].insert(h, t, val)

    # Hs_Gの出力
    for v, hs_g in Hs_G.items():
        hs_g.show(os.path.join(data_dir_name, "out", "fromsrc", "Hs_G", f"hs_g({v}){dataset_id}"))


    # Gtの作成 #
    Gt = nx.DiGraph()
    for (u, v) in G.edges():
        if u in T and v in T:
            Gt.add_edge(u, v, weight=G[u][v]["weight_dash"])
        if v in T and u in T:
            Gt.add_edge(v, u, weight=G[v][u]["weight_dash"])
    # Gtの出力 #
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot()
    ax.set_title(f"Gt({dataset_id})")
    nx.draw_networkx_nodes(G, pos, alpha=ALPHA, node_size=NODE_SIZE) # for G
    nx.draw_networkx_labels(G, pos, alpha=ALPHA) # for G
    edge_labels = {(i, j): w['weight'] for i, j, w in Gt.edges(data=True)}
    nx.draw_networkx_edge_labels(Gt, pos, edge_labels=edge_labels)
    nx.draw_networkx(Gt, pos, with_labels=True, alpha=ALPHA, node_size=NODE_SIZE)
    plt.savefig(os.path.join(data_dir_name, "out", "fromdst", "Gt"+dataset_id))
    if FIGURE_SHOW:
        plt.show()

    # Gtにおけるdstからの最短経路木
    pred, distance = nx.dijkstra_predecessor_and_distance(Gt.reverse(), source=dst)
    SPT_t = Gt.edge_subgraph((i, pred[i][0]) for i in pred if pred[i])
    print("SPT_t", SPT_t.edges())
    # Gtにおけるsrcからの最短経路木 出力
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot()
    ax.set_title(f"Shortest Path Tree from {dst}({dataset_id})")
    nx.draw_networkx_nodes(G, pos, alpha=ALPHA, node_size=NODE_SIZE) # for G
    nx.draw_networkx_labels(G, pos, alpha=ALPHA) # for G
    edge_labels = {(i, j): w['weight'] for i, j, w in SPT_t.edges(data=True)}
    nx.draw_networkx_edge_labels(SPT_t, pos, edge_labels=edge_labels)
    nx.draw_networkx(SPT_t, pos, with_labels=True, alpha=ALPHA, node_size=NODE_SIZE)
    plt.savefig(os.path.join(data_dir_name, "out", "fromdst", "ShortestPathTree_from_dst_in_Gt"+dataset_id))
    if FIGURE_SHOW:
        plt.show()

    # Gt内におけるsidetrack 描画 #
    fig = plt.figure(figsize=FIGSIZE)
    ax = fig.add_subplot()
    ax.set_title(f"SideTrack in Gt({dataset_id})")
    nx.draw_networkx_nodes(G, pos, alpha=ALPHA, node_size=NODE_SIZE) # for G
    nx.draw_networkx_labels(G, pos, alpha=ALPHA) # for G
    edge_labels = {(i, j): w['weight_dash'] for i, j, w in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    sidetrack_edge_list = []
    for (u, v) in Gt.edges():
        if (u, v) not in SPT_t.edges():
            if u in path and v in path: # poptと一致する辺はsidetrackではない
                continue
            if u == dst: # dstから出るsidetrackは不適
                continue
            sidetrack_edge_list.append((u, v))
    nx.draw_networkx_edges(Gt, pos, edgelist=sidetrack_edge_list, edge_color="green", alpha=ALPHA*0.5, width=6.0)
    nx.draw_networkx(Gt, pos, with_labels=True, alpha=ALPHA, node_size=NODE_SIZE)
    fig.savefig(os.path.join(data_dir_name, "out", "fromdst", "Sidetrack_in_Gt"+dataset_id))
    if FIGURE_SHOW:
        plt.show() 
    
    # Ht_OUTの生成
    if not os.path.exists(os.path.join(data_dir_name, "out", "fromdst", "Ht_OUT")):
        os.mkdir(os.path.join(data_dir_name, "out", "fromdst", "Ht_OUT"))
    Ht_OUT = {dst: heap.Heap_Edge()}
    for v in Gt:
        ht_out = heap.Heap_Edge()
        sidetracks = util.sidetrack(v, Gt, SPT_t)
        if len(sidetracks) > 0:
            for (h, t) in sidetracks:
                if h in path and t in path: # poptと一致する辺はsidetrackではない
                    continue
                if h == dst: # dstから出るsidetrackは不適
                    continue
                ht_out.insert(h, t, Gt[h][t]["weight"])
                ht_out.show(os.path.join(data_dir_name, "out", "fromdst", "Ht_OUT", f"ht_out({v}){dataset_id}"))
                Ht_OUT[v] = ht_out
        else:
            Ht_OUT[v] = heap.Heap_Edge()

    # Ht_Tの生成
    if not os.path.exists(os.path.join(data_dir_name, "out", "fromdst", "Ht_T")):
        os.mkdir(os.path.join(data_dir_name, "out", "fromdst", "Ht_T"))
    Ht_T = {}
    for v in Gt.nodes():
        Ht_T[v] = heap.Heap_Edge()
        paths[v] = []
    paths = {dst: [dst]}
    for (t, h) in nx.bfs_edges(SPT_t.reverse(), source=dst):
        paths[h] = paths[t] + [h]
    for v in Gt:
        for u in paths[v]:
            if len(Ht_OUT[u].data) > 0:
                val, (h, t) = Ht_OUT[u].data[0]
                Ht_T[v].insert(h, t, val)
    # Ht_Tの出力
    for v, ht_t in Ht_T.items():
        ht_t.show(os.path.join(data_dir_name, "out", "fromdst", "Ht_T", f"ht_t({v}){dataset_id}"))
    
    # Ht_Gの生成
    if not os.path.exists(os.path.join(data_dir_name, "out", "fromdst", "Ht_G")):
        os.mkdir(os.path.join(data_dir_name, "out", "fromdst", "Ht_G"))
    Ht_G = deepcopy(Ht_T)
    for v, p in paths.items():
        for u in p:
            for _, (val, (h, t)) in enumerate(Ht_OUT[u].data):
                if (val, (h, t)) not in Ht_G[v].data:
                    Ht_G[v].insert(h, t, val)
    # Ht_Gの出力
    for v, ht_g in Ht_G.items():
        ht_g.show(os.path.join(data_dir_name, "out", "fromdst", "Ht_G", f"ht_g({v}){dataset_id}"))

    P = nx.DiGraph()
    Q = {}

    P.add_node("Root")
    child_node = Hmid.data[0][1]
    w = Hmid.data[0][0]
    P.add_edge("Root", child_node, weight=w, type="Root_Hmid")

    for i, (parent_val, parent_name) in enumerate(Hmid.data):
        P.add_node(parent_name, type="Hmid")
        # Q[parent_name].add_node(parent_name, type="Hmid")
        if 2*(i+1)-1 < len(Hmid.data):
            child_val, child_name = Hmid.data[2*(i+1)-1]
            # w = child_val - parent_val
            w = 0.0
            P.add_edge(parent_name, child_name, weight=w, type="Hmid_Hmid")
        if 2*(i+1) < len(Hmid.data):
            child_val, child_name = Hmid.data[2*(i+1)]
            # w = child_val - parent_val
            w = 0.0
            P.add_edge(parent_name, child_name, weight=w, type="Hmid_Hmid")

    for v_val, v_name in Hmid.data:
        Q[v_name] = nx.DiGraph()
        # print(v_name, v_val)
        # # HmidのノードをPに追加
        # P.add_node(v_name, type="Hmid")
        Q[v_name].add_node(v_name, type="Hmid")

        # Hmid - HsG's rootの辺追加
        if len(Hs_G[v_name].data) > 0:
            node = (v_name, Hs_G[v_name].data[0][1])
            w = Hs_G[v_name].data[0][0]
            P.add_edge(v_name, node, weight=w, type="Hmid_HsG")
            Q[v_name].add_edge(v_name, node, weight=w, type="Hmid_HsG")

        # Hmid - HtG's rootの辺追加
        if len(Ht_G[v_name].data) > 0:
            node = (v_name, Ht_G[v_name].data[0][1])
            w = Ht_G[v_name].data[0][0]
            P.add_edge(v_name, node, weight=w, type="Hmid_HtG")
            Q[v_name].add_edge(v_name, node, weight=w, type="Hmid_HtG")

        # HsG内の辺を追加
        for i, (parent_val, (parent_u, parent_v)) in enumerate(Hs_G[v_name].data):
            parent_node = (v_name, (parent_u, parent_v))
            P.add_node(parent_node)
            Q[v_name].add_node(parent_node)
            if 2*(i+1)-1 < len(Hs_G[v_name].data):
                (child_val, (child_h, child_t)) = Hs_G[v_name].data[2*(i+1)-1]
                child_node = (v_name, (child_h, child_t))
                w = Hs_G[v_name].data[2*(i+1)-1][0] - parent_val # 子 - 親
                P.add_edge(parent_node, child_node, weight=w, type="HsG_HsG")
                Q[v_name].add_edge(parent_node, child_node, weight=w, type="HsG_HsG")
            if 2*(i+1) < len(Hs_G[v_name].data):
                (child_val, (child_h, child_t)) = Hs_G[v_name].data[2*(i+1)]
                child_node = (v_name, (child_h, child_t))
                w = Hs_G[v_name].data[2*(i+1)][0] - parent_val # 子 - 親
                P.add_edge(parent_node, child_node, weight=w, type="HsG_HsG")
                Q[v_name].add_edge(parent_node, child_node, weight=w, type="HsG_HsG")

            # HsGのノードparent_node:(parent_u,parent_v)のとき，parent_nodeとHt_G(parent_v)のrootを接続
            if len(Ht_G[parent_v].data) > 0:
                child_node = (parent_v, Ht_G[parent_v].data[0][1])
                w = Ht_G[parent_v].data[0][0]
                # print(f":AAA: {v_name}: ", parent_node, child_node)
                P.add_edge(parent_node, child_node, weight=w, type="HsG_HtG")
                Q[v_name].add_edge(parent_node, child_node, weight=w, type="HsG_HtG")
                # その後，Ht_G(v)内の辺を接続
                for i, (child_val, (child_u, child_v)) in enumerate(Ht_G[parent_v].data):
                    child_node = (parent_v, (child_u, child_v))
                    P.add_node(child_node)
                    Q[v_name].add_node(child_node)
                    if 2*(i+1)-1 < len(Ht_G[parent_v].data):
                        (grandchild_val, (grandchild_u, grandchild_v)) = Ht_G[parent_v].data[2*(i+1)-1]
                        grandchild_node = (parent_v, (grandchild_u, grandchild_v))
                        w = Ht_G[parent_v].data[2*(i+1)-1][0] - child_val # 孫 - 子
                        P.add_edge(child_node, grandchild_node, weight=w, type="HtG_HtG")
                        Q[v_name].add_edge(child_node, grandchild_node, weight=w, type="HtG_HtG")
                    if 2*(i+1) < len(Ht_G[parent_v].data):
                        (grandchild_val, (grandchild_u, grandchild_v)) = Ht_G[parent_v].data[2*(i+1)]
                        grandchild_node = (parent_v, (grandchild_u, grandchild_v))
                        w = Ht_G[parent_v].data[2*(i+1)][0] - child_val # 孫 - 子
                        P.add_edge(child_node, grandchild_node, weight=w, type="HtG_HtG")
                        Q[v_name].add_edge(child_node, grandchild_node, weight=w, type="HtG_HtG")

        # HtG内の辺を追加
        for i, (parent_val, (parent_u, parent_v)) in enumerate(Ht_G[v_name].data):
            parent_node = (v_name, (parent_u, parent_v))
            P.add_node(parent_node)
            Q[v_name].add_node(parent_node)
            if 2*(i+1)-1 < len(Ht_G[v_name].data):
                (child_val, (child_u, child_v)) = Ht_G[v_name].data[2*(i+1)-1]
                child_node = (v_name, (child_u, child_v))
                w = Ht_G[v_name].data[2*(i+1)-1][0] - parent_val # 子 - 親
                P.add_edge(parent_node, child_node, weight=w, type="HtG_HtG")
                Q[v_name].add_edge(parent_node, child_node, weight=w, type="HtG_HtG")
            if 2*(i+1) < len(Ht_G[v_name].data):
                (child_val, (child_u, child_v)) = Ht_G[v_name].data[2*(i+1)]
                child_node = (v_name, (child_u, child_v))
                w = Ht_G[v_name].data[2*(i+1)][0] - parent_val # 子 - 親
                P.add_edge(parent_node, child_node, weight=w, type="HtG_HtG")
                Q[v_name].add_edge(parent_node, child_node, weight=w, type="HtG_HtG")

    for e, val in P.edges().items():
        # print(f"    e: {e}, val: {val}")
        if val["type"] == "Hmid_HsG":
            val["color"] = "blue"
        elif val["type"] == "Hmid_HtG":
            val["color"] = "red"
        elif val["type"] == "HsG_HsG":
            val["color"] = "blue"
        elif val["type"] == "HtG_HtG":
            val["color"] = "red"
        elif val["type"] == "HsG_HtG":
            val["color"] = "pink"
        elif val["type"] == "Hmid_Hmid":
            val["color"] = "yellow"
        if val["type"] == "Root_Hmid":
            val["color"] = "yellow"

    for v_name, q in Q.items():
        for e, val in q.edges().items():
            if val["type"] == "Hmid_HsG":
                val["color"] = "blue"
            elif val["type"] == "Hmid_HtG":
                val["color"] = "red"
            elif val["type"] == "HsG_HsG":
                val["color"] = "blue"
            elif val["type"] == "HtG_HtG":
                val["color"] = "red"
            elif val["type"] == "HsG_HtG":
                val["color"] = "pink"


    # for v in P.edges().values():
    #     print(v)

    # p_pos = nx.spring_layout(P)
    # p_pos = nx.spiral_layout(P)
    # p_pos = nx.random_layout(P)
    # p_pos = nx.spectral_layout(P) # 円形
    # p_pos = nx.planar_layout(P) # 辺が交わらない
    p_pos = nx.shell_layout(P)
    fig = plt.figure(figsize=FIGSIZE_PathGraph)
    ax = fig.add_subplot()
    ax.set_title(f"Path Graph({dataset_id})")
    edge_color = [v["color"] for v in P.edges.values()]
    nx.draw_networkx_edges(P, p_pos, edgelist=P.edges(), edge_color=edge_color, alpha=ALPHA)
    edge_labels = {(i, j): w['weight'] for i, j, w in P.edges(data=True)}
    nx.draw_networkx_edge_labels(P, p_pos, edge_labels=edge_labels) #エッジのラベルを描画
    nx.draw_networkx(P, p_pos, with_labels=True, alpha=ALPHA, node_size=NODE_SIZE)
    plt.savefig(os.path.join(data_dir_name, "out", "Path Graph"+dataset_id))
    if FIGURE_SHOW:
        plt.show()

    if not os.path.exists(os.path.join(data_dir_name, "out", "PathGraph_(Hmid)")):
        os.mkdir(os.path.join(data_dir_name, "out", "PathGraph_(Hmid)"))

    # Path Graphの始点(Hmidのノード)ごとのグラフ
    for v_name, q in Q.items():
        pos = nx.planar_layout(q)
        fig = plt.figure(figsize=FIGSIZE)
        ax = fig.add_subplot()
        ax.set_title(f"Path Graph_{v_name}({dataset_id})")
        edge_color = [v["color"] for v in q.edges.values()]
        nx.draw_networkx_edges(q, pos, edgelist=q.edges(), edge_color=edge_color, alpha=ALPHA)
        edge_labels = {(i, j): w['weight'] for i, j, w in q.edges(data=True)}
        nx.draw_networkx_edge_labels(q, pos, edge_labels=edge_labels) #エッジのラベルを描画
        nx.draw_networkx(q, pos, with_labels=True, alpha=ALPHA, node_size=NODE_SIZE)
        plt.savefig(os.path.join(data_dir_name, "out", "PathGraph_(Hmid)", f"Path Graph_{v_name}({dataset_id})"))
        if FIGURE_SHOW:
            plt.show()

    print("---find k-shortest path---")
    NUM = 15
    k = 0

    q = deque()
    # q.append("Root")
    # q.append(("Root", P["Root"][Hmid.data[0][1]]["weight"]))
    q.append(("Root", 0))

    while len(q) > 0:
        v, potential = q.popleft()
        # print(f"    {v} -> {P[v]}")
        for key, val in P[v].items():
            # print(f"    {u_key} - {u_val}")
            if val["type"] == "Root_Hmid":
                srcName = key
                potential += val["weight"]
                print(f"    Hmid: {srcName}: {potential}")
                q.append((key, potential))
            elif val["type"] == "Hmid_Hmid":
                srcName = key
                potential += val["weight"]
                print(f"    Hmid: {srcName}: {potential}")
                q.append((key, potential))
            else:
                # 辺(u, v)のvがサイドトラック(h, t): (srcName, (h, t))を表している
                (srcName, (h, t)) = key
                potential += val["weight"]
                print(f"    sidetrack: ({h}, {t}): {potential}")
                q.append((key, potential))
            


    # while not que.empty():
    #     v = que.get()
    #     print(f"{v} -> {P[v].nodes()}")
    #     # print(v, P[v])
    #     k += 1
 