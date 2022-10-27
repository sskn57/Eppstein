import networkx as nx

def bib_method(G, src, dst, weight_label="weight", verbose=False):
    S = {} #srcから探索した頂点集合
    T = {} #dstから探索した頂点集合
    ps = {}
    pt = {}
    V = G.nodes()
    path_s = {}
    path_t = {}
    for v in V:
        # Sのpotential初期化
        if v != src:
            ps[v] = float('inf')
        elif v == src:
            ps[v] = 0
            path_s[v] = [v]
        # Tのpotential初期化
        if v != dst:
            pt[v] = float('inf')
        elif v == dst:
            pt[v] = 0
            path_t[v] = [v]
    
    while True:

        # forward search #
        # V-Sのノードvで最小のps(v)をもつvを見つけ，vをSに追加する． #
        v0 = _min(V, S, ps)
        S[v0] = ps[v0]
        if v0 in T.keys():
            # go to step7.
            break
        if verbose:
            print("ps update")
            print(ps)
        for v in G[v0].items():
            # (v0, v)がEに含まれる
            if verbose:
                print(v0, ps[v0]) # v0, ps(v0)
                print(v[0], ps[v[0]]) # v, ps(v)
                print(v[1][weight_label]) # (v0, v)の重み(weight_labelに指定した重み)
            if ps[v0] + v[1][weight_label] < ps[v[0]]: # ps(v0) + l(v0, v) < ps(v)
                ps[v[0]] = ps[v0] + v[1][weight_label]
            if v[0] in S.keys():
                # 経路の更新 #
                tmp = path_s[v[0]].copy()
                tmp.append(v0)
                path_s[v0] = tmp
        if verbose:
            print(ps)
            print("")
        
        # backwark search #
        # V-Tのノードvで最小のps(v)をもつvを見つけ，vをTに追加する．
        v0 = _min(V, T, pt)
        T[v0] = pt[v0]
        if v0 in S.keys():
            # go to step7.
            break
        if verbose:
            print("pt update")
            print(pt)
        for v in G[v0].items():
            # (v0, v)がEに含まれる
            if verbose:
                print(v0, pt[v0]) # v0, ps(v0)
                print(v[0], pt[v[0]]) # v, ps(v)
                print(v[1][weight_label]) # (v0, v)の重み(weight_labelに指定した重み)
            if pt[v0] + v[1][weight_label] < pt[v[0]]: # ps(v0) + l(v0, v) < ps(v)
                pt[v[0]] = pt[v0] + v[1][weight_label]
            if v[0] in T.keys():
                # 経路の更新 #
                tmp = path_t[v[0]].copy()
                tmp.append(v0)
                path_t[v0] = tmp
        if verbose:
            print(pt)
            print("")
    
    # u in S, v in Tに対して ps(u) + l(u, v) + pt(v)を最小にする(u, v)を求める．
    m = float('inf')
    min_u = None
    min_v = None
    for u in S:
        for v in T:
            if (u, v) in G.edges():
                # print(u, v, G[u][v][weight_label])
                if(ps[u] + G[u][v][weight_label] + pt[v] < m):
                    m = ps[u] + G[u][v][weight_label] + pt[v]
                    min_u = u
                    min_v = v

    # srcからdstまでの最短距離
    length = ps[min_u]+G[min_u][min_v][weight_label]+pt[min_v] #float
    # srcからdstまでの最短経路
    path = path_s[min_u] + path_t[min_v][::-1] # list
    # print(length)
    # print(path)
    return [length, path, (S, T)]

# def uni_method(G, src, dst, weight_label="weight", verbose=False):
#     S = {} #srcから探索した頂点集合
#     ps = {}
#     path_s = {}
#     V = G.nodes()
#     for v in V:
#         # Sのpotential初期化
#         if v != src:
#             ps[v] = float('inf')
#         elif v == src:
#             ps[v] = 0
#             path_s[v] = [v]

#     while True:
#         # V-Sのノードvで最小のps(v)をもつvを見つけ，vをSに追加する． #
#         v0 = _min(V, S, ps)
#         S[v0] = ps[v0]
#         if v0 == dst:
#             break
#         if verbose:
#             print(ps)
#         for v in G[v0].items():
#             # (v0, v)がEに含まれる
#             if verbose:
#                 print(v0, ps[v0]) # v0, ps(v0)
#                 print(v[0], ps[v[0]]) # v, ps(v)
#                 print(v[1][weight_label]) # (v0, v)の重み(weight_labelに指定した重み)
#             if ps[v0] + v[1][weight_label] < ps[v[0]]: # ps(v0) + l(v0, v) < ps(v)
#                 ps[v[0]] = ps[v0] + v[1][weight_label]
#             if v[0] in S.keys():
#                 # print(f"{v0}, {v[0]} / {ps[v0]}, {ps[v[0]]}")
#                 # print(v[1][weight_label])
#                 if ps[v0] == ps[v[0]] + v[1][weight_label]:
#                     tmp = path_s[v[0]].copy()
#                     tmp.append(v0)
#                     path_s[v0] = tmp
#                     # print(tmp)
#                 # print("")
#         if verbose:
#             print(ps)
#             print("")

#     for v in G[dst].items():
#         if v[0] in S.keys():
#             tmp = path_s[v[0]].copy()
#             tmp.append(v0)
#             path_s[v0] = tmp
#             # print(f"{v0} add")

#     length = ps[dst]
#     path = path_s[dst]
#     # print(length)
#     # print(path)

#     return [length, path, S]

# V-Bの中で最初のpotentialをもつノードvを探す
def _min(V, B, potential):
    m = ["None", float('inf')]
    for v_name in V:
        # print(v_name)
        if v_name not in B.keys():
            # A-Bの中のvについて
            # print(v_name, B[v_name])
            if potential[v_name] <= m[1]:
                m[0] = v_name
                m[1] = potential[v_name]
    return m[0]

# def pred2path(G, src, dst):
#     pred, distance = nx.dijkstra_predecessor_and_distance(G.reverse(), source=dst)
#     # print(pred)
#     # print(pred[src])
    