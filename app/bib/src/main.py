import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import os, shutil
import heapq
import util, dijkstra, heap, centrality
from copy import deepcopy
from collections import deque
import settings

# DATASET_NAME = "DATASET001"
CONFIG_PATH = os.path.join("bib", "data", "config.ini")

FIGSIZE = (8, 8)
FIGSIZE_PathGraph = (30, 30)
NODE_SIZE = 1000
FIGURE_SHOW = False
CYCLE_FLAG = True

if __name__ == "__main__":
    print(f"use {settings.dataset_name}")
    G = settings.info["G"]["data"]
    pos = settings.info["G"]["pos"]
    src = settings.info["src"]
    dst = settings.info["dst"]
    threshold = settings.info["threshold"]
    
    # 出力用ディレクトリ生成
    if os.path.exists(settings.info["output_dir_path"]):
        shutil.rmtree(settings.info["output_dir_path"])
    os.mkdir(settings.info["output_dir_path"])

    # 通信成功率の計算
    centrality.cal_rate(G, settings)

    # Original Graph 描画
    d = util.DrawOriginalGraph(graph=G, settings=settings)
    d.draw(weight_label="weight")
    d.draw(weight_label="rate")

    pred, distance = nx.dijkstra_predecessor_and_distance(G, source=dst)
    # Potential計算 #
    for edge in G.edges():
        tail, head = edge
        # 辺の重みを変更
        hs = G[tail][head]['weight'] + distance[head] - distance[tail]
        ht = G[tail][head]['weight'] + distance[tail] - distance[head]
        G[tail][head]['weight_dash'] = (hs + ht) / 2
    
    # 双方向ダイクストラ法
    length_shortest, path_shortest, (S, T), F, U, (ps, pt) = dijkstra.bib_method(G, src, dst, threshold=threshold, weight_label="weight_dash", verbose=settings.info["bib_method_verbose"])

    print(f"length: {length_shortest}") # 最短経路距離
    print(f"path: {path_shortest}") # 最短経路
    print(f"S: {S}") # srcから探索した点集合
    print(f"T: {T}") # dstから探索した点集合
    print(f"F: {F}") # 追加する辺集合
    print(f"U: {U}") # 追加する点集合

    if len(U) == 0:
        # Uが空の場合は終了
        exit()
        # pass

    # Hmidの作成 #
    Hmid = heap.Heap_Node()
    for u in U:
        qu = ps[u] + pt[u]
        Hmid.insert(u, qu)
    # Hmid出力 #
    if settings.info["Hmid"]["verbose"]:
        Hmid.show(settings.info["Hmid"]["data_path"], format=settings.info["Hmid"]["format"])
    
    if settings.info["fromsrc"]["verbose"]:
        if not os.path.exists(settings.info["fromsrc"]["dir_path"]):
           os.mkdir(settings.info["fromsrc"]["dir_path"])

    if settings.info["fromdst"]["verbose"]:
        if not os.path.exists(settings.info["fromdst"]["dir_path"]):
            os.mkdir(settings.info["fromdst"]["dir_path"])

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
    d = util.DrawGs(graph=Gs, settings=settings, pos=pos)
    d.draw(G)

    # Gsにおけるsrcからの最短経路木
    pred, distance = nx.dijkstra_predecessor_and_distance(Gs, source=src)  
    SPT_s = Gs.reverse().edge_subgraph((i, pred[i][0]) for i in pred if pred[i])    
    print("SPT_s", SPT_s.edges())
    # Gsにおけるsrcからの最短経路木 出力
    d = util.DrawSPT_s(graph=SPT_s, settings=settings, pos=pos)
    d.draw(G, Gs, path_shortest)
    
    # srcからGs内の各ノードvへの最短経路path_s
    path_s = {}
    for v in Gs.nodes():
        path_s[v] = []
    path_s[src].append(src)
    for (u, v) in nx.bfs_edges(SPT_s.reverse(), source=src):
        path_s[v] = path_s[u] + [v]
    print(f"path_s: {path_s}")

    # Gtの作成 #
    Gt = nx.DiGraph()
    for (u, v) in G.edges():
        if u in T and v in T:
            Gt.add_edge(u, v, weight=G[u][v]["weight_dash"])
        if v in T and u in T:
            Gt.add_edge(v, u, weight=G[v][u]["weight_dash"])
    # Gtの出力 #
    d = util.DrawGt(graph=Gt, settings=settings, pos=pos)
    d.draw(G)

    # Gtにおけるdstからの最短経路木
    pred, distance = nx.dijkstra_predecessor_and_distance(Gt.reverse(), source=dst)
    SPT_t = Gt.edge_subgraph((i, pred[i][0]) for i in pred if pred[i])
    print("SPT_t", SPT_t.edges())
    # # Gtにおけるdstからの最短経路木 出力
    d = util.DrawSPT_t(graph=SPT_t, settings=settings, pos=pos)
    d.draw(G, Gt, path_shortest)

    # Gt内の各ノードvからdstへの最短経路path_t
    path_t = {}
    for v in Gt.nodes():
        path_t[v] = []
    path_t[dst].append(dst)
    for (u, v) in nx.bfs_edges(SPT_t.reverse(), source=dst):
        path_t[v] = [v] + path_t[u]
        # path_t[v] = path_t[u] + [v]
    print(f"path_t: {path_t}")

    # ヒープHs，Htの作成
    H = {
        "s": {"step1": {}, "step2": {}, "step3": {}},
        "t": {"step1": {}, "step2": {}, "step3": {}}
        }
    # 出力用ディレクトリの初期化 Hs
    if not os.path.exists(settings.info["H"]["s"]["step1"]["dir_path"]):
        os.mkdir(settings.info["H"]["t"]["step1"]["dir_path"])
    if not os.path.exists(settings.info["H"]["s"]["step2"]["dir_path"]):
        os.mkdir(settings.info["H"]["t"]["step2"]["dir_path"])
    if not os.path.exists(settings.info["H"]["s"]["step3"]["dir_path"]):
        os.mkdir(settings.info["H"]["t"]["step3"]["dir_path"])
    # 出力用ディレクトリの初期化 Ht
    if not os.path.exists(settings.info["H"]["t"]["step1"]["dir_path"]):
        os.mkdir(settings.info["H"]["t"]["step1"]["dir_path"])
    if not os.path.exists(settings.info["H"]["t"]["step2"]["dir_path"]):
        os.mkdir(settings.info["H"]["t"]["step2"]["dir_path"])
    if not os.path.exists(settings.info["H"]["t"]["step3"]["dir_path"]):
        os.mkdir(settings.info["H"]["t"]["step3"]["dir_path"])

    # Hs (step1)の実行
    for v in S + U:
        H["s"]["step1"][v] = heap.Heap_Edge()
    for v in Gs.reverse():
        sidetracks = util.sidetrack(v, Gs.reverse(), SPT_s)
        for (_, w) in sidetracks:
            if v in path_shortest and w in path_shortest:
                continue
            H["s"]["step1"][v].insert(v, w, Gs.reverse()[v][w]["weight"])
            if settings.info["H"]["s"]["step1"]["flag"]:
                fpath = os.path.join(
                    settings.info["H"]["s"]["step1"]["dir_path"], 
                    f"Hs({v}) (step1)[{settings.info['dataset_name']}])"
                    )
                H["s"]["step1"][v].show(fpath, format=settings.info["H"]["s"]["step1"]["format"])


    # Hs(step2)の実行
    for v in S + U:
        H["s"]["step2"][v] = heap.Heap_Edge()
    for v in Gs.reverse():
        for u in path_s[v]:
            # print(v, path_s[v])
            if u in H["s"]["step1"].keys():
                if len(H["s"]["step1"][u].data) > 0:
                    val, (h, t) = H["s"]["step1"][u].data[0]
                    H["s"]["step2"][v].insert(h, t, val)
    # Hs(step2)の出力
    for v, hv in H["s"]["step2"].items():
        if settings.info["H"]["s"]["step2"]["flag"]:
                fpath = os.path.join(
                    settings.info["H"]["s"]["step2"]["dir_path"], 
                    f"Hs({v}) (step2)[{settings.info['dataset_name']}])"
                    )
                H["s"]["step2"][v].show(fpath, format=settings.info["H"]["s"]["step2"]["format"])

    # Hs(step3)の実行
    H["s"]["step3"] = deepcopy(H["s"]["step2"])
    for v, p in path_s.items():
        for u in p:
            # for _, (val, (h, t)) in enumerate(Hs_OUT[u].data):
            for (val, (h, t)) in H["s"]["step1"][u].data:
                if (val, (h, t)) not in H["s"]["step3"][v].data:
                    H["s"]["step3"][v].insert(h, t, val)
    # Hs(step3)の出力
    for v, hv in H["s"]["step3"].items():
        if settings.info["H"]["s"]["step3"]["flag"]:
                fpath = os.path.join(
                    settings.info["H"]["s"]["step3"]["dir_path"], 
                    f"Hs({v}) (step3)[{settings.info['dataset_name']}])"
                    )
                H["s"]["step3"][v].show(fpath, format=settings.info["H"]["s"]["step3"]["format"])
    
    # Ht(step1)の実行
    for v in T:
        H["t"]["step1"][v] = heap.Heap_Edge()
    for v in Gt:
        sidetracks = util.sidetrack(v, Gt, SPT_t)
        for (_, w) in sidetracks:
            if v in path_shortest and w in path_shortest:
                continue
            if v == dst: # dstから出るsidetrackは不適
                continue
            H["t"]["step1"][v].insert(v, w, Gt[v][w]["weight"])
            if settings.info["H"]["t"]["step1"]["flag"]:
                fpath = os.path.join(
                    settings.info["H"]["t"]["step1"]["dir_path"], 
                    f"Ht({v}) (step1)[{settings.info['dataset_name']}])"
                    )
                H["t"]["step1"][v].show(fpath, format=settings.info["H"]["t"]["step1"]["format"])

    # Ht(step2)の実行
    for v in T:
        H["t"]["step2"][v] = heap.Heap_Edge()
    for v in Gt:
        for u in path_t[v]:
            if u in H["t"]["step1"].keys():
                if len(H["t"]["step1"][u].data) > 0:
                    val, (h, t) = H["t"]["step1"][u].data[0]
                    # H["t"]["step2"][u].insert(t, h, val)
                    H["t"]["step2"][v].insert(h, t, val)
    # Ht(step2)の出力
    for v, hv in H["t"]["step2"].items():
        if settings.info["H"]["t"]["step2"]["flag"]:
                fpath = os.path.join(
                    settings.info["H"]["t"]["step2"]["dir_path"], 
                    f"Ht({v}) (step2)[{settings.info['dataset_name']}])"
                    )
                H["t"]["step2"][v].show(fpath, format=settings.info["H"]["t"]["step2"]["format"])
    
    # Ht(step3)の実行
    H["t"]["step3"] = deepcopy(H["t"]["step2"])
    for v, p in path_t.items():
        for u in p:
            for (val, (h, t)) in H["t"]["step1"][u].data:
                if (val, (h, t)) not in H["t"]["step3"][v].data:
                    H["t"]["step3"][v].insert(h, t, val)
    # Ht(step3)の出力
    for v, hv in H["t"]["step3"].items():
        if settings.info["H"]["t"]["step3"]["flag"]:
                fpath = os.path.join(
                    settings.info["H"]["t"]["step3"]["dir_path"], 
                    f"Ht({v}) (step3)[{settings.info['dataset_name']}])"
                    )
                H["t"]["step3"][v].show(fpath, format=settings.info["H"]["t"]["step3"]["format"])


    # # PathGraphの生成
    P, Q, sidetrack_weight = util.make_PathGraph(H["s"]["step3"], H["t"]["step3"], Hmid, settings)


    # PathGraph Pの出力
    d = util.DrawP(P, settings)
    d.draw()

    # 出力用ディレクトリの初期化
    if not os.path.exists(settings.info["Q"]["dir_path"]):
        os.mkdir(settings.info["Q"]["dir_path"])
    # PathGraph Q(Hmidごと)の出力
    d = util.DrawQ(Q, settings)
    d.draw()

    print("--- start find k-shortest path ---")
    k_path = {}
    q = deque()
    q.append((settings.info["P"]["RootNodeName"], 0))
    # k_pathの構築
    while len(q) > 0:
        v, potential = q.popleft()
        for key, val in P[v].items():
            # print(key, val)
            if val["type"] == settings.info["P"]["Root-Hmid"]["type"]:
                potential += val["weight"]
                q.append((key, potential))
            elif val["type"] == settings.info["P"]["Hmid-Hmid"]["type"]:
                potential += val["weight"]
                q.append((key, potential))
            else:
                (_, (u, v)) = key
                potential += val["weight"]
                q.append((key, potential))
                # 最短経路の計算
                if u in path_s.keys() and v in path_t.keys():
                    path = path_s[u] + path_t[v]
                    length = ps[u] + sidetrack_weight[(u, v)] + pt[v]
                    # print(util.nodetype2strtype(path))
                    k_path[util.nodetype2strtype(path)] = {
                        "path": path,
                        "length": length
                        }
    print("--- fin. find k-shortest path ---")
    print("--- k-path ---")
    print(f"k-path: {len(k_path.keys())}")
    for p, info in k_path.items():
        print(f"    {p}")

    # Gd = Gs + Gtの作成
    Gd = nx.compose(Gs, Gt)
    d = util.DrawGs(graph=Gd, settings=settings, pos=pos)
    d.draw(G)

    # overlapの計算
    print("--- start calculate detour ---")
    for info in k_path.values():
        info["overlap"] = None
        # overlapの計算
        overlap = 0.0
        for e1 in util.nodetype2edgetype(info["path"]):
            if e1 in util.nodetype2edgetype(path_shortest):
                u1, v1 = e1
                overlap += Gd[u1][v1]["weight"]
        if settings.info["overlap_verbose"]:
            path_str = util.nodetype2strtype(info["path"])
            print(f"    len: {info['length']}, path: {path_str}")
            print(f"        overlap: {overlap}")
        info["overlap"] = overlap
    
    # 最小のoverlapを算出し，detourを定義
    detour = []
    min_overlap = float("inf")
    for info in k_path.values():
        if info["overlap"] < min_overlap:
            min_overlap = info["overlap"]
    for info in k_path.values():
        if not util.cycle(info["path"]):
            if info["overlap"] == min_overlap:
                info["detour"] = True
                detour.append({
                    "path": info["path"],
                    "length": info["length"]}
                    )
            else:
                info["detour"] = False
    # 最短経路を追加
    k_path[util.nodetype2strtype(path_shortest)] = {
        "path": path_shortest,
        "length": length_shortest,
        "overlap": None, 
        "detour": True
    }
    detour.append({
        "path": path_shortest,
        "length": length_shortest
        })

    if settings.info["k-path_verbose"]:
        for info in sorted(k_path.values(), key=lambda v:v["length"]):
            print(f"   {util.nodetype2strtype(info['path'])}")
            print(f"        len   : {info['length']}")
            print(f"        cycle : {util.cycle(info['path'])}")
            print(f"        detour: {info['detour']}")
    print("--- fin. calculate detour ---")

    print("--- detour ---")
    for info in detour:
        print(f"    detour: {info}")
        # RPscoreの計算
        rp = centrality.RPscore(info["path"], G)
        print(f"        RPscore: {rp}")
        info["RP"] = rp

    # detourの描画
    if not os.path.exists(settings.info["detour"]["dir_path"]):
        os.mkdir(settings.info["detour"]["dir_path"])
    for k, info in enumerate(detour[::-1]):
        fig = plt.figure(figsize=settings.info["detour"]["figsize"])
        ax = fig.add_subplot()
        p_str = util.nodetype2strtype(info["path"])
        ax.set_title(f"detour: {p_str}(len: {info['length']}, RP: {round(info['RP'], 3)}")
        # G 描画
        edge_labels = {(i, j): w['weight'] for i, j, w in G.edges(data=True)}
        nx.draw_networkx_edge_labels(
            G, 
            pos, 
            edge_labels=edge_labels
            )
        nx.draw_networkx(
            G, 
            pos, 
            with_labels=True, 
            alpha=settings.info["G"]["alpha"], 
            node_size=settings.info["G"]["nodesize"]
            )
        # detour
        nx.draw_networkx_edges(
            G, 
            pos, 
            edgelist=util.nodetype2edgetype(info["path"]), 
            edge_color=settings.info["detour"]["edge_color"], 
            alpha=settings.info["detour"]["alpha"], 
            width=settings.info["detour"]["edge_width"],
            arrowsize=settings.info["detour"]["arrowsize"]
            )
        # save
        # fname = f"detour[{k}]" + "." + settings.info["detour"]["format"]
        fname = f"detour[{k}]" + ".png" 
        output_path = os.path.join(settings.info["detour"]["dir_path"], fname)
        plt.savefig(output_path)
        for f in settings.info["detour"]["format"]:
            fname = f"detour[{k}]" + "." + f
            output_path = os.path.join(settings.info["detour"]["dir_path"], fname)
            plt.savefig(output_path)

    # 中心性の計算
    nodes = {}
    for v in G.nodes():
        if v == src:
            nodes[v] = {"src": True, "dst": False}
        elif v ==dst:
            nodes[v] = {"src": False, "dst": True}
        if v != src and v != dst:
            nodes[v] = {"src": False, "dst": False}

    # 媒介度Cbet(v)計算
    centrality.cal_Cbet(nodes, detour, settings)
    for v, info in nodes.items():
        if v != src and v != dst:
            print(f" Cbet({v}): {info['Cbet']}")
    # 中心性C(v)計算
    exist_critical_node = centrality.centrality(nodes, detour, settings)
    # exist_critical_node = centrality.centrality_v2(nodes, detour, settings, path_shortest)

    print("--- Centrality (正規化前) ---")
    for v, info in nodes.items():
        if nodes[v]["src"]:
            print(f"    C({v}): src")
        elif nodes[v]["dst"]:
            print(f"    C({v}): dst")
        else:
            print(f"    C({v}): {round(info['C']['type0'], 3)}")
    for v, info in nodes.items():
        if nodes[v]["src"]:
            print(f"    C({v}): src")
        elif nodes[v]["dst"]:
            print(f"    C({v}): dst")
        else:
            print(f"    C({v}): {round(info['C']['type1'], 3)}")
    print("--- Centrality (without critical_node) ---")
    for v, info in nodes.items():
        if nodes[v]["src"]:
            print(f"    C({v}): src")
        elif nodes[v]["dst"]:
            print(f"    C({v}): dst")
        else:
            # if "critical_node" in nodes[v].keys():
            if exist_critical_node:
                if nodes[v]["critical_node"]:
                    print(f"    C({v}): critical node")
                else:
                    print(f"    C({v}): {round(info['C']['type2'], 3)}")

    x_for_graph = []
    y1_for_graph = []
    y2_for_graph = []
    for v, info in nodes.items():
        if info["src"] == True:
            pass
        elif info["dst"] == True:
            pass
        else:
            x_for_graph.append(v)
            y1_for_graph.append(info["C"]["type1"])
            if exist_critical_node:
                if info["critical_node"]:
                    y2_for_graph.append(0.0)
                else:
                    y2_for_graph.append(info["C"]["type2"])

    fig = plt.figure(figsize=(8, 5))
    plt.title("Centrality")
    plt.xlabel("node")
    plt.ylabel("centrality")
    left = np.arange(len(y1_for_graph))
    width = 0.4
    plt.bar(left, y1_for_graph, linewidth=0, align="center", color="red", width=width)
    if exist_critical_node:
        plt.bar(left+width, y2_for_graph, linewidth=0, align="center", color="blue", width=width)
    plt.xticks(left + width/2, x_for_graph)
    for f in settings.info["cent"]["format"]:
        fname = settings.info["cent"]["dir_path"] + "cent" + "." + f
        plt.savefig(fname)