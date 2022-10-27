import networkx as nx



# def bib_method(G, src, dst, weight_label="weight", verbose=False):
#     S = {} #srcから探索した頂点集合
#     T = {} #dstから探索した頂点集合
#     ps = {}
#     pt = {}
#     V = G.nodes()
#     path_s = {}
#     path_t = {}
#     ds = 0 # srcとSに追加した最後の頂点間の最短距離
#     dt = 0 # dstとTに追加した最後の頂点間の最短距離
#     for v in V:
#         # Sのpotential初期化
#         if v != src:
#             ps[v] = float('inf')
#         elif v == src:
#             ps[v] = 0
#             path_s[v] = [v]
#         # Tのpotential初期化
#         if v != dst:
#             pt[v] = float('inf')
#         elif v == dst:
#             pt[v] = 0
#             path_t[v] = [v]
    
#     while True:

#         # forward search #
#         # V-Sのノードvで最小のps(v)をもつvを見つけ，vをSに追加する． #
#         v0 = _min(V, S, ps)
#         S[v0] = ps[v0]
#         ds = ps[v0]
#         if v0 in T.keys():
#             # go to step7.
#             break
#         if verbose:
#             print("ps update")
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
#                 # 経路の更新 #
#                 tmp = path_s[v[0]].copy()
#                 tmp.append(v0)
#                 path_s[v0] = tmp
#         if verbose:
#             print(ps)
#             print("")
        
#         # backwark search #
#         # V-Tのノードvで最小のps(v)をもつvを見つけ，vをTに追加する．
#         v0 = _min(V, T, pt)
#         T[v0] = pt[v0]
#         dt = pt[v0]
#         if v0 in S.keys():
#             # go to step7.
#             break
#         if verbose:
#             print("pt update")
#             print(pt)
#         for v in G[v0].items():
#             # (v0, v)がEに含まれる
#             if verbose:
#                 print(v0, pt[v0]) # v0, ps(v0)
#                 print(v[0], pt[v[0]]) # v, ps(v)
#                 print(v[1][weight_label]) # (v0, v)の重み(weight_labelに指定した重み)
#             if pt[v0] + v[1][weight_label] < pt[v[0]]: # ps(v0) + l(v0, v) < ps(v)
#                 pt[v[0]] = pt[v0] + v[1][weight_label]
#             if v[0] in T.keys():
#                 # 経路の更新 #
#                 tmp = path_t[v[0]].copy()
#                 tmp.append(v0)
#                 path_t[v0] = tmp
#         if verbose:
#             print(pt)
#             print("")
    
#     # u in S, v in Tに対して ps(u) + l(u, v) + pt(v)を最小にする(u, v)を求める．
#     m = float('inf')
#     min_u = None
#     min_v = None
#     for u in S:
#         for v in T:
#             if (u, v) in G.edges():
#                 # print(u, v, G[u][v][weight_label])
#                 if(ps[u] + G[u][v][weight_label] + pt[v] < m):
#                     m = ps[u] + G[u][v][weight_label] + pt[v]
#                     min_u = u
#                     min_v = v

#     # srcからdstまでの最短距離
#     length = ps[min_u]+G[min_u][min_v][weight_label]+pt[min_v] #float
#     # srcからdstまでの最短経路
#     path = path_s[min_u] + path_t[min_v][::-1] # list
#     # print(length)
#     # print(path)
#     return [length, path, (S, T), (ps, pt), (ds, dt)]

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

# def bib_method(G, src, dst, threshold=0, weight_label="weight", verbose=False):
#     S = {} # srcから探索した頂点集合
#     T = {} # dstから探索した頂点集合
#     U = {}
#     V = {}
#     ps = {} # srcから探索したときのポテンシャル
#     pt = {} # dstから探索したときのポテンシャル
#     V = G.nodes()
#     path_s = {} # srcとv間の最短経路
#     path_t = {} # dstとv間の最短経路
#     ds = 0 # srcとSに追加した最後の頂点間の最短距離
#     dt = 0 # dstとTに追加した最後の頂点間の最短距離
#     min_u = None
#     min_v = None
#     count = 1
#     all_searched = False
#     path = None
#     length = None
#     for v in V:
#         # Sのpotential初期化
#         if v != src:
#             ps[v] = float('inf')
#         elif v == src:
#             ps[v] = 0
#             path_s[v] = [v]
#         # Tのpotential初期化
#         if v != dst:
#             pt[v] = float('inf')
#         elif v == dst:
#             pt[v] = 0
#             path_t[v] = [v]
    
#     while True:
#         print(f"count: {count}")
#         # if path == None:
#         while True:
#             # forward search #
#             # V-Sのノードvで最小のps(v)をもつvを見つけ，vをSに追加する． #
#             v0 = _min(V, S, ps)
#             if v0 == None:
#                 all_searched = True
#                 print("all node searched.")
#                 break
#             print(f"    [S] v0:{v0}")
#             S[v0] = ps[v0]
#             ds = ps[v0]
#             for v_name, v_val in G[v0].items():
#                 # (v0, v)がEに含まれるような点vについて
#                 if ps[v0] + v_val[weight_label] < ps[v_name]: # ps(v0) + l(v0, v) < ps(v)
#                     ps[v_name] = ps[v0] + v_val[weight_label] # ps(v) = ps(v0) + l(v0, v)
#                 if v_name in S.keys():
#                     # 経路の更新 #
#                     tmp = path_s[v_name].copy()
#                     tmp.append(v0)
#                     path_s[v0] = tmp
#             if path == None:
#                 if v0 in T.keys():
#                     # go to step7.
#                     break
            
#             # backwark search #
#             # V-Tのノードvで最小のps(v)をもつvを見つけ，vをTに追加する．
#             v0 = _min(V, T, pt)
#             if v0 == None:
#                 all_searched = True
#                 print("all node searched.")
#                 break
#             print(f"    [T] v0:{v0}")
#             T[v0] = pt[v0]
#             dt = pt[v0]
#             for v_name, v_val in G[v0].items():
#                 # (v0, v)がEに含まれるような点vについて
#                 if pt[v0] + v_val[weight_label] < pt[v_name]: # ps(v0) + l(v0, v) < ps(v)
#                     pt[v_name] = pt[v0] + v_val[weight_label]
#                 if v_name in T.keys():
#                     # 経路の更新 #
#                     tmp = path_t[v_name].copy()
#                     tmp.append(v0)
#                     path_t[v0] = tmp
#             if path == None:
#                 if v0 in S.keys():
#                     # go to step7.
#                     break
#             else:
#                 # path Noneでないとき(最短経路発見後F, U構築時)，SとTにv0を一つずつ追加
#                 break
        
#         if all_searched:
#             break

#         # STEP7
#         # u in S, v in Tに対して ps(u) + l(u, v) + pt(v)を最小にする(u, v)を求める．
#         if min_u == None or min_v == None:
#             m = float('inf')
#             for u in S:
#                 for v in T:
#                     if (u, v) in G.edges():
#                         if(ps[u] + G[u][v][weight_label] + pt[v] < m):
#                             m = ps[u] + G[u][v][weight_label] + pt[v]
#                             min_u = u
#                             min_v = v

#         # srcからdstまでの最短距離
#         length = ps[min_u]+G[min_u][min_v][weight_label]+pt[min_v] #float
#         # srcからdstまでの最短経路
#         path = path_s[min_u] + path_t[min_v][::-1] # list
#         print(f"    length: {length}")
#         print(f"    path: {path}")
#         print(f"    S: {S}")
#         print(f"    T: {T}")
#         print(f"    ps: {ps}")
#         print(f"    pt: {pt}")
#         # print(f"    ds: {ds}")
#         # print(f"    dt: {dt}")
#         print(f"    ds + dt = {ds} + {dt} = {ds+dt}")
#         print(f"    length(popt)+threshold = {length}+{threshold} = {length+threshold}")
#         print()
#         if count >= 2 and ds + dt > length + threshold:
#             break
#             # return [length, path, (S, T), (ps, pt), (ds, dt)]
#         count += 1

#     # F, Uの構築
#     F = []
#     U = []
#     for (u, v) in G.edges():
#         if u in S.keys():
#             if v in T.keys() and v not in S.keys():
#                 # print(v)
#                 F.append((u, v))
#                 if v not in U:
#                     U.append(v)
#         if v in S.keys():
#             if u in T.keys() and u not in S.keys():
#                 # print(u)
#                 F.append((v, u))
#                 if u not in U:
#                     U.append(u)

#     # return [length, path, (S, T), (ps, pt), (ds, dt)]
#     return [length, path, (list(S.keys()), list(T.keys())), F, U, (ps, pt)]

def bib_method(G, src, dst, threshold=0, weight_label="weight", verbose=False):
    print(f"--- start bib method ---")
    S = {} # srcから探索した頂点集合
    T = {} # dstから探索した頂点集合
    V = {}
    ps = {} # srcから探索したときのポテンシャル
    pt = {} # dstから探索したときのポテンシャル
    V = G.nodes()
    path_s = {} # srcとv間の最短経路
    path_t = {} # dstとv間の最短経路
    ds = 0 # srcとSに追加した最後の頂点間の最短距離
    dt = 0 # dstとTに追加した最後の頂点間の最短距離
    min_u = None
    min_v = None
    count = 1
    all_searched = False
    path = None # 最短経路
    length = None # 最短経路長
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
    
    while True: # popt構築(1回目)，S,Tに追加(2回目以降)
        if path == None:
            # poptの計算
            while True:
                # forward search
                v0 = _min(V, S, ps)
                if v0 == None:
                    print(f"    v0 in S: None")
                    pass
                else:
                    print(f"    v0 in S: {v0}")
                    S[v0] = ps[v0] # Sに追加
                    ds = ps[v0] # dsの計算
                    for v_name, v_val in G[v0].items():
                        # (v0, v)がEに含まれるような点vについて
                        if ps[v0] + v_val[weight_label] < ps[v_name]:
                            ps[v_name] = ps[v0] + v_val[weight_label] #ポテンシャルの更新
                        if v_name in S.keys():
                            # Sに含まれていた場合，経路の更新
                            tmp = path_s[v_name].copy()
                            tmp.append(v0)
                            path_s[v0] = tmp
                    if v0 in T.keys():
                        # v0がTに含まれる場合，STEP7へ
                        break
                
                # backward search
                v0 = _min(V, T, pt)
                if v0 == None:
                    print(f"    v0 in T: None")
                    break
                else:
                    print(f"    v0 in T: {v0}")
                    T[v0] = pt[v0] # Tに追加
                    dt = pt[v0] # dsの計算
                    for v_name, v_val in G[v0].items():
                        # (v0, v)がEに含まれるような点vについて
                        if pt[v0] + v_val[weight_label] < pt[v_name]:
                            pt[v_name] = pt[v0] + v_val[weight_label] #ポテンシャルの更新
                        if v_name in T.keys():
                            # Tに含まれていた場合，経路の更新
                            tmp = path_t[v_name].copy()
                            tmp.append(v0)
                            path_t[v0] = tmp
                    if v0 in S.keys():
                        # v0がSに含まれる場合，STEP7へ
                        break
            # STEP 7
            # u in S, v in Tに対して ps(u) + l(u, v) + pt(v)を最小にする(u, v)を求める．
            if min_u == None or min_v == None:
                m = float("inf")
                for u in S:
                    for v in T:
                        if (u, v) in G.edges():
                            if ps[u] + G[u][v][weight_label] + pt[v] < m:
                                m = ps[u] + G[u][v][weight_label] + pt[v]
                                min_u = u
                                min_v = v
            # srcからdstまでの最短距離
            length = ps[min_u] + G[min_u][min_v][weight_label] + pt[min_v]
            # srcからdstまでの最短経路
            path = path_s[min_u] + path_t[min_v][::-1]
            print(f"    length: {length}")
            print(f"    path: {path}")
            print(f"    S: {S.keys()}")
            print(f"    T: {T.keys()}")
            print(f"    ps: {ps}")
            print(f"    pt: {pt}")
            print(f"    ds + dt = {ds} + {dt} = {ds+dt}")
            print(f"    length(popt)+threshold = {length}+{threshold} = {length+threshold}")
            print()
        elif path != None:
            # S, Tに追加する処理
            if ds + dt > length + threshold:
                # S, Tへの追加終了
                print(f"    fin.")
                break
    
            v0_S = _min(V, S, ps)
            v0_T = _min(V, T, pt)

            print(f"    (v0 in S, v0 in T): ({v0_S}, {v0_T})")

            add_S_or_T = None
            # print(v0_S, v0_T)
            # print(type(v0_T))
            if v0_S == "None" and v0_T != "None":
                # v0 in Sが見つからないとき = Sに全て追加済
                add_S_or_T = "T"
            elif v0_S != "None" and v0_T == "None":
                # v0 in Tが見つからないとき = Tに全て追加済
                add_S_or_T = "S"
            elif v0_S != "None" and v0_T != "None":
                if ps[v0_S] <= pt[v0_T]:
                    add_S_or_T = "S"
                else:
                    add_S_or_T = "T"
            elif v0_S == "None" and v0_T == "None":
                print(f"    S full and T full")
                break

            if add_S_or_T == "S":
                # v0 in SをSに追加
                v0 = v0_S
                print(f"    v0 in S: {v0}")
                S[v0] = ps[v0] # Sに追加
                # ds = ps[v0] # dsの計算
                #ds, dtの計算
                ds = ps[src] # S.keys()の中で，最大のポテンシャルをもつvを探す
                for v in S.keys():
                    if ds <= ps[v]:
                        ds = ps[v]
                dt = pt[dst] # T.keys()の中で，最大のポテンシャルをもつvを探す
                for v in T.keys():
                    if dt <= pt[v]:
                        dt = pt[v]

                for v_name, v_val in G[v0].items():
                    # (v0, v)がEに含まれるような点vについて
                    if ps[v0] + v_val[weight_label] < ps[v_name]:
                        ps[v_name] = ps[v0] + v_val[weight_label] #ポテンシャルの更新
                    if v_name in S.keys():
                        # Sに含まれていた場合，経路の更新
                        print(path_s)
                        tmp = path_s[v_name].copy()
                        tmp.append(v0)
                        path_s[v0] = tmp
            elif add_S_or_T == "T":
                # v0 in TをTに追加
                v0 = v0_T
                print(f"    v0 in T: {v0}")
                T[v0] = pt[v0] # Tに追加
                # dt = pt[v0] # dtの計算
                #ds, dtの計算
                ds = ps[src] # S.keys()の中で，最大のポテンシャルをもつvを探す
                for v in S.keys():
                    if ds <= ps[v]:
                        ds = ps[v]
                dt = pt[dst] # T.keys()の中で，最大のポテンシャルをもつvを探す
                for v in T.keys():
                    if dt <= pt[v]:
                        dt = pt[v]
                for v_name, v_val in G[v0].items():
                    # (v0, v)がEに含まれるような点vについて
                    if pt[v0] + v_val[weight_label] < pt[v_name]:
                        pt[v_name] = pt[v0] + v_val[weight_label] #ポテンシャルの更新
                    if v_name in T.keys():
                        # Tに含まれていた場合，経路の更新
                        tmp = path_t[v_name].copy()
                        tmp.append(v0)
                        path_t[v0] = tmp

            # print(f"    S: {list(S.keys())}")
            # print(f"    S: {list(sorted(S.keys()))}")
            print(f"    S: {sorted(S.keys())}")
            # print(f"    T: {list(T.keys())}")
            # print(f"    T: {list(sorted(T.keys()))}")
            print(f"    T: {sorted(T.keys())}")
            print(f"    ps: {ps}")
            print(f"    pt: {pt}")
            print(f"    ds + dt = {ds} + {dt} = {ds+dt}")
            print(f"    length(popt)+threshold = {length}+{threshold} = {length+threshold}")
            print()

    # F, Uの構築
    U = []
    F = []
    for (u, v) in G.edges():
        if u in S.keys():
            if v in T.keys() and v not in S.keys():
                # print(v)
                F.append((u, v))
                if v not in U:
                    U.append(v)
        if v in S.keys():
            if u in T.keys() and u not in S.keys():
                # print(u)
                F.append((v, u))
                if u not in U:
                    U.append(u)
    print(f"--- fin. bib method ---")
    return [length, path, (sorted(S.keys()), sorted(T.keys())), sorted(F), sorted(U), (ps, pt)]
