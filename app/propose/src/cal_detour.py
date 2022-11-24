import dijkstra
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from heap import Heap_Node, Heap_Edge
from copy import deepcopy
from collections import deque
import sys, os

# 迂回経路の取得
def cal_detour(data):
    
    print("------ cal_detour ------")
    G = data["G"]
    src = data["src"]
    dst = data["dst"]
    threshold = data["threshold"]

    # (1))双方向ダイクストラ法
    print(f"    --- (1) bib dijkstra method ---")
    length_shortest, path_shortest, (S, T), F, U, (ps, pt) = dijkstra.bib_method(
        G, 
        src,
        dst,
        threshold,
        weight_label="length",
        verbose=False
    )
    print(f"        length: {length_shortest}") # 最短経路距離
    print(f"        path: {path_shortest}") # 最短経路
    print(f"        S: {S}") # srcから探索した点集合
    print(f"        T: {T}") # dstから探索した点集合
    print(f"        F: {F}") # 追加する辺集合
    print(f"        U: {U}") # 追加する点集合

    # dataに追加
    data["shortest"] = {"length": length_shortest, "path": {}}
    data["shortest"]["path"]["nodetype"] = path_shortest
    data["shortest"]["path"]["edgetype"] = nodetype2edgetype(path_shortest)
    data["shortest"]["path"]["strtype"] = nodetype2strtype(path_shortest)
    data["S"] = S
    data["T"] = T
    data["F"] = F
    data["U"] = U

    if len(U) == 0:
        # Uが空の場合は終了
        exit()
    
    # (2) Hmid作成
    print(f"    --- (2) make Hmid ---")
    Hmid = Heap_Node()
    for u in U:
        qu = ps[u] + pt[u]
        Hmid.insert(u, qu)
    # 出力(省略)
    # dataに追加
    data["Hmid"] = {}
    data["Hmid"]["graph"] = Hmid

    # (3) Gs, Gtに関する処理
    print(f"    --- (3) process about Gs ---")
    # (3-1) Gsの作成
    Gs = nx.DiGraph()
    for (u, v) in G.edges():
        if u in S:
            if v in S or v in U:
                Gs.add_edge(u, v, length=G[u][v]["length"])
        if v in S:
            if u in S or u in U:
                Gs.add_edge(v, u, length=G[v][u]["length"])
    # 出力(省略)
    # dataに追加
    data["Gs"] = {}
    data["Gs"]["graph"] = Gs

    # (3-2) Gsにおけるsrcからの最短経路木
    pred, distance = nx.dijkstra_predecessor_and_distance(Gs, source=src)  
    SPT_s = Gs.reverse().edge_subgraph((i, pred[i][0]) for i in pred if pred[i])    
    # 出力(省略)
    # dataに追加
    data["SPT_s"] = {}
    data["SPT_s"]["graph"] = SPT_s

    # (3-3) srcからGs内の各ノードvへの最短経路path_s
    path_s = {}
    for v in Gs.nodes():
        path_s[v] = []
    path_s[src].append(src)
    for (u, v) in nx.bfs_edges(SPT_s.reverse(), source=src):
        path_s[v] = path_s[u] + [v]
    # 出力
    # print(f"path_s: {path_s}")
    # dataに追加
    data["path_s"] = path_s

    # (4-1) Gtの作成
    print(f"    --- (4) process about Gt---")
    Gt = nx.DiGraph()
    for (u, v) in G.edges():
        if u in T and v in T:
            Gt.add_edge(u, v, length=G[u][v]["length"])
        if v in T and u in T:
            Gt.add_edge(v, u, length=G[v][u]["length"])
    # 出力(省略)
    # dataに追加
    data["Gt"] = {}
    data["Gt"]["graph"] = Gt

    # (4-2) Gtにおけるdstからの最短経路木
    pred, distance = nx.dijkstra_predecessor_and_distance(Gt.reverse(), source=dst)
    SPT_t = Gt.edge_subgraph((i, pred[i][0]) for i in pred if pred[i])
    # 出力(省略)
    # dataに追加
    data["SPT_t"] = {}
    data["SPT_t"]["graph"] = SPT_t

    # (4-3) srcからGs内の各ノードvへの最短経路path_s
    path_t = {}
    for v in Gt.nodes():
        path_t[v] = []
    path_t[dst].append(dst)
    for (u, v) in nx.bfs_edges(SPT_t.reverse(), source=dst):
        path_t[v] = [v] + path_t[u]
    # 出力
    # print(f"path_t: {path_t}")
    # dataに追加
    data["path_t"] = path_t

    print(f"    --- (5) make Hs & Ht ---")
    # (5) Hs, Htの作成
    H = {
        "s": {"step1": {}, "step2": {}, "step3": {}},
        "t": {"step1": {}, "step2": {}, "step3": {}}
        }
    
    # (5-1) Hs(step1)の作成
    for v in S + U:
        H["s"]["step1"][v] = Heap_Edge()
    for v in Gs.reverse():
        sidetracks = sidetrack(v, Gs.reverse(), SPT_s)
        for (_, w) in sidetracks:
            if v in path_shortest and w in path_shortest:
                continue
            H["s"]["step1"][v].insert(v, w, Gs.reverse()[v][w]["length"])
    
    # (5-2) Hs(step2)の作成
    for v in S + U:
        H["s"]["step2"][v] = Heap_Edge()
    for v in Gs.reverse():
        for u in path_s[v]:
            # print(v, path_s[v])
            if u in H["s"]["step1"].keys():
                if len(H["s"]["step1"][u].data) > 0:
                    val, (h, t) = H["s"]["step1"][u].data[0]
                    H["s"]["step2"][v].insert(h, t, val)

    # (5-3) Hs(step3)の作成
    H["s"]["step3"] = deepcopy(H["s"]["step2"])
    for v, p in path_s.items():
        for u in p:
            for (val, (h, t)) in H["s"]["step1"][u].data:
                if (val, (h, t)) not in H["s"]["step3"][v].data:
                    H["s"]["step3"][v].insert(h, t, val)

    # (5-4) Ht(step1)の作成
    for v in T:
        H["t"]["step1"][v] = Heap_Edge()
    for v in Gt:
        sidetracks = sidetrack(v, Gt, SPT_t)
        for (_, w) in sidetracks:
            if v in path_shortest and w in path_shortest:
                continue
            if v == dst: # dstから出るsidetrackは不適
                continue
            H["t"]["step1"][v].insert(v, w, Gt[v][w]["length"])
    
    # (5-5) Ht(step2)の作成
    for v in T:
        H["t"]["step2"][v] = Heap_Edge()
    for v in Gt:
        for u in path_t[v]:
            if u in H["t"]["step1"].keys():
                if len(H["t"]["step1"][u].data) > 0:
                    val, (h, t) = H["t"]["step1"][u].data[0]
                    H["t"]["step2"][v].insert(h, t, val)
    # (5-6) Ht(step3)の作成
    H["t"]["step3"] = deepcopy(H["t"]["step2"])
    for v, p in path_t.items():
        for u in p:
            for (val, (h, t)) in H["t"]["step1"][u].data:
                if (val, (h, t)) not in H["t"]["step3"][v].data:
                    H["t"]["step3"][v].insert(h, t, val)

    # dataへの追加
    data["H"] = H

    # (6) PathGraphの作成
    print(f"    --- (6) make PathGraph ---")
    P = nx.DiGraph()
    sidetrack_weight = {}
    Hs = H["s"]["step3"]
    Ht = H["t"]["step3"]
    
    # (6-1) Rootの追加
    RootNodeName = "Root-Node"
    P.add_node(RootNodeName)

    # (6-2) Root - Hmid's root
    child_val, child_node = Hmid.data[0] # Hmidのrootを取り出す
    w = child_val
    t = "Root-Hmid(root)"
    P.add_edge(RootNodeName, child_node, weight=w, type=t)

    for i, (parent_val, parent_name) in enumerate(Hmid.data):
        # (6-3) Hmid - Hmid
        # t = "Hmid"
        P.add_node(parent_name)
        if 2*(i+1)-1 < len(Hmid.data):
            child_val, child_name = Hmid.data[2*(i+1)-1]
            w = 0.0 # Hmid - Hmid間はポテンシャル同じ
            t = "Hmid-Hmid"
            P.add_edge(parent_name, child_name, weight=w, type=t)
        if 2*(i+1) < len(Hmid.data):
            child_val, child_name = Hmid.data[2*(i+1)]
            w = 0.0 # Hmid - Hmid間はポテンシャル同じ
            t = "Hmid-Hmid"
            P.add_edge(parent_name, child_name, weight=w, type=t)
    
        # (6-4) Hmid - Hs's root
        if len(Hs[parent_name].data) > 0:
            sidetrack_val, sidetrack_edge = Hs[parent_name].data[0]
            child_node = (parent_name, sidetrack_edge)
            w = sidetrack_val
            t = "Hmid-Hs(root)"
            P.add_edge(parent_name, child_node, weight=w, type=t)

        # (6-5) Hmid - Ht's root
        if len(Ht[parent_name].data) > 0:
            sidetrack_val, sidetrack_edge = Ht[parent_name].data[0]
            child_node = (parent_name, sidetrack_edge)
            w = sidetrack_val
            t = "Hmid-Ht(root)"
            P.add_edge(parent_name, child_node, weight=w, type=t)
        
        # (6-6) Hs - Hs
        for j, (child_val, (child_u, child_v)) in enumerate(Hs[parent_name].data):
            # Hsを順に調べる．まず，ノードを追加
            child_node = (parent_name, (child_u, child_v))
            P.add_node(child_node)
            # サイドトラックの重みを記録
            sidetrack_weight[(child_u, child_v)] = child_val
            # 次に，辺(Hs-Hs)を追加
            if 2*(j+1)-1 < len(Hs[parent_name].data):
                g_child_val, (g_child_u, g_child_v) = Hs[parent_name].data[2*(j+1)-1]
                g_child_node =(parent_name, (g_child_u, g_child_v))
                w = g_child_val - child_val
                t = "Hs-Hs"
                P.add_edge(child_node, g_child_node, weight=w, type=t)
            if 2*(j+1) < len(Hs[parent_name].data):
                g_child_val, (g_child_u, g_child_v) = Hs[parent_name].data[2*(j+1)]
                g_child_node =(parent_name, (g_child_u, g_child_v))
                w = g_child_val - child_val
                t = "Hs-Hs"
                P.add_edge(child_node, g_child_node, weight=w, type=t)
                
            # Hsのノードchild_node:(child_u, child_v)について，Ht(child_v)が存在するとき，
            if child_v in Ht.keys():
                if len(Ht[child_v].data) > 0:
                    # (6-7) child_node - Ht(child_v)'s root
                    g_child_val, (g_child_u, g_child_v) = Ht[child_v].data[0] # root取り出し
                    g_child_node = (child_v, (g_child_u, g_child_v))
                    w = g_child_val
                    t = "Hs-Ht(root)"
                    P.add_edge(child_node, g_child_node, weight=w, type=t)
                    # (6-8) その後，Ht(child_v)内の辺を追加
                    for k, (g_child_val, (g_child_u, g_child_v)) in enumerate(Ht[child_v].data):
                        g_child_node = (child_v, (g_child_u, g_child_v))
                        P.add_node(g_child_node)
                        sidetrack_weight[(g_child_u, g_child_v)] = g_child_val
                        if 2*(k+1)-1 < len(Ht[child_v].data):
                            gg_child_val, (gg_child_u, gg_child_v) = Ht[child_v].data[2*(k+1)-1]
                            gg_child_node = (child_v, (gg_child_u, gg_child_v))
                            w = gg_child_val - g_child_val
                            t = "Ht-Ht"
                            P.add_edge(g_child_node, gg_child_node, weight=w, type=t)
                            
                        if 2*(k+1) < len(Ht[child_v].data):
                            gg_child_val, (gg_child_u, gg_child_v) = Ht[child_v].data[2*(k+1)]
                            gg_child_node = (child_v, (gg_child_u, gg_child_v))
                            w = gg_child_val - g_child_val
                            t = "Ht-Ht"
                            P.add_edge(g_child_node, gg_child_node, weight=w, type=t)

        # (6-7) Ht - Ht
        for j, (child_val, (child_u, child_v)) in enumerate(Ht[parent_name].data):
            # Htを順に調べる．まず，ノードを追加
            child_node = (parent_name, (child_u, child_v))
            P.add_node(child_node)
            # サイドトラックの重みを記録
            sidetrack_weight[(child_u, child_v)] = child_val
            # 次に，辺を追加
            if 2*(j+1)-1 < len(Ht[parent_name].data):
                g_child_val, (g_child_u, g_child_v) = Ht[parent_name].data[2*(j+1)-1]
                g_child_node =(parent_name, (g_child_u, g_child_v))
                w = g_child_val - child_val
                t = "Ht-Ht"
                P.add_edge(child_node, g_child_node, weight=w, type=t)
                
            if 2*(j+1) < len(Ht[parent_name].data):
                g_child_val, (g_child_u, g_child_v) = Ht[parent_name].data[2*(j+1)]
                g_child_node =(parent_name, (g_child_u, g_child_v))
                w = g_child_val - child_val
                t = "Ht-Ht"
                P.add_edge(child_node, g_child_node, weight=w, type=t)

    # dataへの追加
    data["P"] = {}
    data["P"]["graph"] = P
    data["sidetrack_weight"] = sidetrack_weight

    # (7) k-path
    print(f"    --- (7) find k-path ---")
    data["k-path"] = []
    q =  deque()
    q.append((RootNodeName, 0))
    while len(q) > 0:
        v, potential = q.popleft()
        for key, val in P[v].items():
            # print(key, val)
            if val["type"] == "Root-Hmid(root)":
                potential += val["weight"]
                q.append((key, potential))
            elif val["type"] == "Hmid-Hmid":
                potential += val["weight"]
                q.append((key, potential))
            else:
                (_, (u, v)) = key
                potential += val["weight"]
                q.append((key, potential))
                # 最短経路の計算
                if u in path_s.keys() and v in path_t.keys():
                    path_nodetype = path_s[u] + path_t[v] # 経路(nodetype)
                    path_edgetype = nodetype2edgetype(path_nodetype) # 経路(edgetype)
                    path_strtype = nodetype2strtype(path_nodetype) # 経路(strtype)
                    length = 0 # 経路長
                    # print(path_strtype)
                    for (u, v) in path_edgetype:
                        length += G[u][v]["length"]
                        # print(f"    {u}-{v}: {G[u][v]['length']}")

                    if length < data["shortest"]["length"] + threshold:
                        # 追加用のデータ(辞書)作成
                        tmp = {"length": length, "path": {}}
                        tmp["path"]["nodetype"] = path_nodetype
                        tmp["path"]["edgetype"] = path_edgetype
                        tmp["path"]["strtype"] = path_strtype
                        # まだ存在しない経路の場合追加
                        exist = False
                        for p in data["k-path"]:
                            if p["path"]["strtype"] == path_strtype:
                                exist = True
                        if not exist:
                            data["k-path"].append(tmp)
    # 標準出力
    for p in data["k-path"]:
        print(f"        {p['path']['strtype']}: {p['length']}")

    # (8) Gd = Gs + Gt の作成
    print(f"    --- (8) make Gd ---")
    Gd = nx.compose(Gs, Gt)

    # (9) overlapの計算
    print(f"    --- (9) calculate overlap ---")
    for p in data["k-path"]:
        p["overlap"] = None
        overlap = 0.0
        for e in p["path"]["edgetype"]:
            if e in data["shortest"]["path"]["edgetype"]:
                (u, v) = e
                overlap += Gd[u][v]["length"]
        p["overlap"] = overlap
    # 標準出力
    for p in data["k-path"]:
        print(f"        {p['path']['nodetype']}: {p['overlap']}")

    # (10) cycleの計算
    print(f"    --- (10) check cycle ---")
    for p in data["k-path"]:
        p["cycle"] = cycle(p["path"]["nodetype"])

    # (11) check detour
    print(f"    --- (11) check detour ---")
    detour_n = 1 # overlapのうち，小さい方からdetour_n番目までをdetourとする
    n_min_overlap = cal_n_min_overlap(data["k-path"], detour_n) # n番目に小さいoverlapの値
    for p in data["k-path"]:
        if p["overlap"] <= n_min_overlap:
            p["detour"] = True
        else:
            p["detour"] = False
    # 標準出力
    for p in data["k-path"]:
        if p["detour"]:
            print(f"        {p['path']['nodetype']}: {p['overlap']} (detour)")
        else:
            print(f"        {p['path']['nodetype']}: {p['overlap']}")

    
    # (12) 出力 (detour(detour + shortest)の定義)
    print(f"    --- (12) output ---")
    detour = []
    # (12-1) 迂回経路の追加
    for p in data["k-path"]:
        if p["detour"] and not p["cycle"]:
            tmp = {}
            tmp["length"] = p["length"]
            tmp["path"] = p["path"]
            detour.append(tmp)
    # (12-2) 最短経路の追加
    detour.append(data["shortest"])
    # (12-3) sort
    detour = sorted(detour, key=lambda x: x["length"], reverse=False)
    # (12-4) dataへの追加
    data["detour"] = detour
    # 標準出力
    for p in detour:
        print(f"        {p['path']['nodetype']}: {p['length']}")

    # グラフ出力
    for extension in data["draw_params"]["extensions"]:
        detour_dir_path = os.path.join(data["output_path"], "Imgs", extension, "detour")
        if not os.path.exists(detour_dir_path):
            os.mkdir(detour_dir_path)
        
        for i, p in enumerate(detour):
            fname = f"detour[{i}]"
            detour_graph(p, fname, extension, data)

    print(f"------ end cal_detour ------\n")

    # # 出力先を標準出力に戻す
    # sys.stdout = sys.__stdout__
    return detour

# detourの描画
def detour_graph(p, fname, extension, data):

    params = data["draw_params"]["cal_detour"]["detour_graph"]

    fig = plt.figure(figsize=params["figsize"])
    # title
    plt.title(f"{p['path']['strtype']} (delta:{data['threshold']} len:{p['length']})")
    # 枠線消去
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    # G描画
    # 辺
    nx.draw_networkx_edges(
        data["G"], 
        data["pos"],
        edge_color=params["G"]["edge"]["color"],
        width=params["G"]["edge"]["width"],
        alpha=params["G"]["edge"]["alpha"])
    # 辺のラベル
    edge_labels = {(i, j): round(w["length"], params["G"]["edge"]["digit"]) for i, j, w in data["G"].edges(data=True)}
    nx.draw_networkx_edge_labels(
        data["G"], 
        data["pos"], 
        edge_labels=edge_labels,
        font_color=params["G"]["edge"]["font"]["color"],
        font_size=params["G"]["edge"]["font"]["size"],
        alpha=params["G"]["edge"]["font"]["alpha"],
        bbox={"fc": (0, 0, 0, 0), "ec": (0, 0, 0, 0)},
        rotate=False)
    # node
    node_color = []
    for v in data["G"].nodes():
        if v == data["src"]:
            node_color.append(params["G"]["node"]["color"]["src"])
        elif v == data["dst"]:
            node_color.append(params["G"]["node"]["color"]["dst"])
        else:
            node_color.append(params["G"]["node"]["color"]["default"])
    nx.draw_networkx_nodes(
        data["G"], 
        data["pos"], 
        node_color=node_color,
        node_size=params["G"]["node"]["size"], 
        alpha=params["G"]["node"]["alpha"])

    # nodeのラベル
    nx.draw_networkx_labels(
        data["G"], 
        data["pos"], 
        font_color=params["G"]["node"]["font"]["color"], 
        font_size=params["G"]["node"]["font"]["size"],
        alpha=params["G"]["node"]["font"]["alpha"])

    # detour描画
    # 辺
    nx.draw_networkx_edges(
        data["G"], 
        data["pos"], 
        edgelist=p["path"]["edgetype"],
        edge_color=params["detour"]["edge"]["color"],
        arrowsize=params["detour"]["edge"]["arrowsize"],
        width=params["detour"]["edge"]["width"],
        alpha=params["detour"]["edge"]["alpha"])
    

    fpath = os.path.join(data["output_path"], "Imgs", extension, "detour", fname + "." + extension)
    plt.savefig(fpath, bbox_inches='tight')


# # detourの描画
# def detour_graph(p, fname, extension, data):
#     figsize = (8, 5)
#     params = {}
#     # G描画設定
#     params["G"] = {"edge": {}, "node": {}}
#     params["G"]["edge"]["digit"] = 3
#     params["G"]["edge"]["color"] = "gray"
#     params["G"]["edge"]["width"] = 3
#     params["G"]["edge"]["alpha"] = 0.6
#     params["G"]["edge"]["font"] = {"color": "black", "size": 12, "alpha": 0.9}

#     params["G"]["node"]["color"] = {"src": "red", "dst": "blue", "default": "black"}
#     params["G"]["node"]["size"] = 800
#     params["G"]["node"]["alpha"] = 0.9

#     params["G"]["node"]["font"] = {"color": "white", "size": 20, "alpha": 0.9}

#     # detour 描画設定
#     params["detour"] = {"edge": {}}
#     params["detour"]["edge"]["color"] = "red"
#     params["detour"]["edge"]["arrowsize"] = 15
#     params["detour"]["edge"]["width"] = 9
#     params["detour"]["edge"]["alpha"] = 0.5

#     fig = plt.figure(figsize=figsize)
#     # title
#     plt.title(f"{p['path']['strtype']} (delta:{data['threshold']} len:{p['length']})")
#     # 枠線消去
#     plt.gca().spines['right'].set_visible(False)
#     plt.gca().spines['top'].set_visible(False)
#     plt.gca().spines['bottom'].set_visible(False)
#     plt.gca().spines['left'].set_visible(False)

#     # G描画
#     # 辺
#     nx.draw_networkx_edges(
#         data["G"], 
#         data["pos"],
#         edge_color=params["G"]["edge"]["color"],
#         width=params["G"]["edge"]["width"],
#         alpha=params["G"]["edge"]["alpha"])
#     # 辺のラベル
#     edge_labels = {(i, j): round(w["length"], params["G"]["edge"]["digit"]) for i, j, w in data["G"].edges(data=True)}
#     nx.draw_networkx_edge_labels(
#         data["G"], 
#         data["pos"], 
#         edge_labels=edge_labels,
#         font_color=params["G"]["edge"]["font"]["color"],
#         font_size=params["G"]["edge"]["font"]["size"],
#         alpha=params["G"]["edge"]["font"]["alpha"],
#         bbox={"fc": (0, 0, 0, 0), "ec": (0, 0, 0, 0)},
#         rotate=False)
#     # node
#     node_color = []
#     for v in data["G"].nodes():
#         if v == data["src"]:
#             node_color.append(params["G"]["node"]["color"]["src"])
#         elif v == data["dst"]:
#             node_color.append(params["G"]["node"]["color"]["dst"])
#         else:
#             node_color.append(params["G"]["node"]["color"]["default"])
#     nx.draw_networkx_nodes(
#         data["G"], 
#         data["pos"], 
#         node_color=node_color,
#         node_size=params["G"]["node"]["size"], 
#         alpha=params["G"]["node"]["alpha"])

#     # nodeのラベル
#     nx.draw_networkx_labels(
#         data["G"], 
#         data["pos"], 
#         font_color=params["G"]["node"]["font"]["color"], 
#         font_size=params["G"]["node"]["font"]["size"],
#         alpha=params["G"]["node"]["font"]["alpha"])

#     # detour描画
#     # 辺
#     nx.draw_networkx_edges(
#         data["G"], 
#         data["pos"], 
#         edgelist=p["path"]["edgetype"],
#         edge_color=params["detour"]["edge"]["color"],
#         arrowsize=params["detour"]["edge"]["arrowsize"],
#         width=params["detour"]["edge"]["width"],
#         alpha=params["detour"]["edge"]["alpha"])
    

#     fpath = os.path.join(data["output_path"], "Imgs", extension, "detour", fname + "." + extension)
#     plt.savefig(fpath, bbox_inches='tight')

def cal_n_min_overlap(k_path, n):
    # k_path: data["k-path"]を与える.
    # n: 小さい方からn番目に小さいoverlapの値
    l = []
    for p in k_path:
        l.append(p["overlap"])
    n_min_overlap = sorted(set(l))[n-1]
    return n_min_overlap

def sidetrack(v, G, T):
    ret = []
    for (h, t) in G.edges():
        if (h, t) not in T.edges():
            if v == h:
                if (h, t) not in ret:
                    ret.append((h, t))
    return ret

def cycle(path_nodetype):
    if len(path_nodetype) != len(set(path_nodetype)):
        # 重複あり(cycleである)
        return True
    else:
        return False

def nodetype2edgetype(nodetype_path):
    return [e for e in zip(nodetype_path[0:], nodetype_path[1:])]

def edgetype2nodetype(edgetype_path):
    return [h for (h, t) in edgetype_path] + [edgetype_path[-1][1]]

def nodetype2strtype(nodetype_path):
    ret = "".join([str(_) for _ in nodetype_path])
    return ret 