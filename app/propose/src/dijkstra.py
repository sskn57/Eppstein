import networkx as nx

def _min(V, B, potential):
    m = ["None", float('inf')]
    for v_name in V:
        if v_name not in B:
            # A-Bの中のvについて
            if potential[v_name] <= m[1]:
                m[0] = v_name
                m[1] = potential[v_name]
    return m[0]

def bib_method(G, src, dst, threshold, weight_label, verbose=False):
    S = [] # srcから探索した頂点集合
    T = [] # dstから探索した頂点集合
    V = G.nodes() # G内の頂点集合
    ps = {} # srcから探索したときのポテンシャル
    pt = {} # dstから探索したときのポテンシャル
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
            path_s[v] = [v] # 経路path_sの初期化
        # Tのpotential初期化
        if v != dst:
            pt[v] = float('inf')
        elif v == dst:
            pt[v] = 0
            path_t[v] = [v] # 経路path_tの初期化
    
    while True: # popt構築(1回目)，S,Tに追加(2回目以降)
        if path == None:
            # poptの計算
            while True:
                # forward search
                v0 = _min(V, S, ps)
                if v0 == None:
                    if verbose:
                        print(f"    v0 in S: None")
                    pass
                else:
                    if verbose:
                        print(f"    v0 in S: {v0}")
                    S.append(v0) # Sに追加
                    ds = ps[v0] # dsの計算
                    in_S = []
                    for v_name, v_val in G[v0].items():
                        # (v0, v)がEに含まれるような点vについて
                        if ps[v0] + v_val[weight_label] < ps[v_name]:
                            ps[v_name] = ps[v0] + v_val[weight_label] #ポテンシャルの更新
                        if v_name in S:
                            # Sに含まれていた場合，経路の更新
                            in_S.append(v_name)
                    if len(in_S) > 0:
                        m = in_S[0]
                        for v in in_S:
                            if ps[v] + G[v][v0][weight_label] < ps[m] + G[m][v0][weight_label]:
                                m = v
                        # path_tの追加
                        tmp = path_s[m].copy()
                        tmp.append(v0)
                        path_s[v0] = tmp
                    if v0 in T:
                        # v0がTに含まれる場合，STEP7へ
                        break

                    
                
                # backward search
                v0 = _min(V, T, pt)
                if v0 == None:
                    if verbose:
                        print(f"    v0 in T: None")
                    break
                else:
                    if verbose:
                        print(f"    v0 in T: {v0}")
                    T.append(v0) # Tに追加
                    dt = pt[v0] # dtの計算
                    in_T = []
                    for v_name, v_val in G[v0].items():
                        # print(f"        v_name: {v_name}")
                        # (v0, v)がEに含まれるような点vについて
                        if pt[v0] + v_val[weight_label] < pt[v_name]:
                            pt[v_name] = pt[v0] + v_val[weight_label] #ポテンシャルの更新
                        if v_name in T:
                            # Tに含まれていた場合，点の記録
                            # print(f"            v_name in T: {v_name}")
                            in_T.append(v_name)                            
                    if len(in_T) > 0:
                        m = in_T[0]
                        for v in in_T:
                            if pt[v] + G[v][v0][weight_label] < pt[m] + G[m][v0][weight_label]:
                                m = v
                        # path_tの追加
                        tmp = path_t[m].copy()
                        tmp.append(v0)
                        path_t[v0] = tmp
                    
                    if v0 in S:
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
            if verbose:
                print(f"    length: {length}")
                print(f"    path: {path}")
                print(f"    S: {S}")
                print(f"    T: {T}")
                print(f"    ps: {ps}")
                print(f"    pt: {pt}")
                for v, val in path_s.items():
                    print(f"    {v}: {val}")
                for v, val in path_t.items():
                    print(f"    {v}: {val}")
                # print(f"    ps")
                # for name, val in sorted(ps.items()):
                #     if name in S:
                #         print(f"        ps({name}): {val}")
                # print(f"    pt")
                # for name, val in sorted(pt.items()):
                #     if name in T:
                #         print(f"        pt({name}): {val}")
                # print(f"    path_s: {path_s}")
                # print(f"    path_t: {path_t}")
                print(f"    ds + dt = {ds} + {dt} = {ds+dt}")
                print(f"    length(popt)+threshold = {length}+{threshold} = {length+threshold}")
                print()
        elif path != None:
            # S, Tに追加する処理
            if ds + dt > length + threshold:
                # S, Tへの追加終了
                break
    
            v0_S = _min(V, S, ps)
            v0_T = _min(V, T, pt)

            add_S_or_T = None
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
                if verbose:
                    print(f"    S full and T full")
                break

            if add_S_or_T == "S":
                # v0 in SをSに追加
                v0 = v0_S
                if verbose:
                    print(f"    v0 in S: {v0}")
                S.append(v0) # Sに追加
                #ds, dtの計算
                ds = ps[src] # S.keys()の中で，最大のポテンシャルをもつvを探す
                for v in S:
                    if ds <= ps[v]:
                        ds = ps[v]
                dt = pt[dst] # T.keys()の中で，最大のポテンシャルをもつvを探す
                for v in T:
                    if dt <= pt[v]:
                        dt = pt[v]

                for v_name, v_val in G[v0].items():
                    # (v0, v)がEに含まれるような点vについて
                    if ps[v0] + v_val[weight_label] < ps[v_name]:
                        ps[v_name] = ps[v0] + v_val[weight_label] #ポテンシャルの更新
                    if v_name in S:
                        # Sに含まれていた場合，経路の更新
                        tmp = path_s[v_name].copy()
                        tmp.append(v0)
                        path_s[v0] = tmp
            elif add_S_or_T == "T":
                # v0 in TをTに追加
                v0 = v0_T
                if verbose:
                    print(f"    v0 in T: {v0}")
                T.append(v0) # Tに追加
                #ds, dtの計算
                ds = ps[src] # S.keys()の中で，最大のポテンシャルをもつvを探す
                for v in S:
                    if ds <= ps[v]:
                        ds = ps[v]
                dt = pt[dst] # T.keys()の中で，最大のポテンシャルをもつvを探す
                for v in T:
                    if dt <= pt[v]:
                        dt = pt[v]
                for v_name, v_val in G[v0].items():
                    # (v0, v)がEに含まれるような点vについて
                    if pt[v0] + v_val[weight_label] < pt[v_name]:
                        pt[v_name] = pt[v0] + v_val[weight_label] #ポテンシャルの更新
                    if v_name in T:
                        # Tに含まれていた場合，経路の更新
                        tmp = path_t[v_name].copy()
                        tmp.append(v0)
                        path_t[v0] = tmp
            if verbose:
                print(f"    S: {sorted(S)}")
                print(f"    T: {sorted(T)}")
                print(f"    ps: {ps}")
                print(f"    pt: {pt}")
                # print(f"    ps")
                # for name, val in sorted(ps.items()):
                #     if val != float('inf'):
                #         if name in S:
                #             print(f"          ps({name}): {val}")
                #         else:
                #             print(f"        ps({name}): {val}")
                # print(f"    pt")
                # for name, val in sorted(pt.items()):
                #     if val != float('inf'):
                #         if name in T:
                #             print(f"          pt({name}): {val}")
                #         else:
                #             print(f"        pt({name}): {val}")
                print(f"    ds + dt = {ds} + {dt} = {ds+dt}")
                print(f"    length(popt)+threshold = {length}+{threshold} = {length+threshold}")
                print()

    # F, Uの構築
    U = []
    F = []
    for (u, v) in G.edges():
        if u in S:
            if v in T and v not in S:
                # print(v)
                F.append((u, v))
                if v not in U:
                    U.append(v)
        if v in S:
            if u in T and u not in S:
                # print(u)
                F.append((v, u))
                if u not in U:
                    U.append(u)
    return [length, path, (sorted(S), sorted(T)), sorted(F), sorted(U), (ps, pt)]
