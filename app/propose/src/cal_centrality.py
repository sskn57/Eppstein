import os
import matplotlib.pyplot as plt
import numpy as np

# def cal_centrality(detour, params, G):
def cal_centrality(data):

    print("------ cal_centrality ------")
    detour = data["detour"]

    # (1) detour内のノード集合
    nodes = nodes_in_detour(detour)
    
    # (2) ノードの種類(src/dst/between)の設定
    # (2-1) src, dstの設定
    nodes[data["src"]]["type"] = "src"
    nodes[data["dst"]]["type"] = "dst"
    # (2-2) 中間ノードの設定
    for v, v_info in nodes.items():        
        if v_info["type"] != "src" and v_info["type"] != "dst":
            v_info["type"] = "between"

    # (3) RPスコアの計算
    for p in detour:
        p["RPscore"] = RPscore(p["path"], data["G"])

    # (3) 媒介度の計算
    for v, v_info in nodes.items():
        if v_info["type"] == "between":
            v_info["Cbet"] = Cbet(v, detour)
    
    # (4) critical nodeが存在するか確認
    exist_critical_node = False
    for v, v_info in nodes.items():
        if v_info["type"] == "between":
            if v_info["Cbet"] == 1.0:
                exist_critical_node = True
                v_info["critical_node"] = True
            else:
                v_info["critical_node"] = False

    # (5) 中心性の計算(正規化前)
    for v, v_info in nodes.items():
        if v_info["type"] == "between":
            sum_RPscore = 0.0
            for p in detour:
                if v in p["path"]["nodetype"]:
                    # vが含まれる経路のRPscoreの加算
                    sum_RPscore += p["RPscore"]
            # 中心性c(正規化前)の計算
            c = v_info["Cbet"] * sum_RPscore
            v_info["C_without_normalize"] = c
    
    # (6) 正規化

    # (6-1) 最大の中心性c(正規化前)の計算
    if exist_critical_node:
        # critical_nodeが存在する場合，critical_node含む場合の最大値と含まない場合の最大値
        m = 2
    else:
        # critical_nodeが存在しない場合，最大値
        m = 1
    l = []
    for v, v_info in nodes.items():
        if v_info["type"] == "between":
            l.append(v_info["C_without_normalize"])
    cmax_list = sorted(set(l), reverse=True)[:m]
    
    # (6-2) 最大の中心性cmaxで割ることで正規化
    for v, v_info in nodes.items():
        if v_info["type"] == "between":
            C = []
            if v_info["critical_node"]:
                C.append(v_info["C_without_normalize"] / cmax_list[0])
                C.append(None)
            else:
                for cmax in cmax_list:
                    C.append(v_info["C_without_normalize"] / cmax)
            v_info["C"] = C

    data["nodes"] = nodes
    data["detour"] = detour
    data["exist_critical_node"] = exist_critical_node

    # 標準出力
    for p in detour:
        print(f"    {p['path']['nodetype']}: (len: {p['length']}, RPscore: {p['RPscore']})")
    print()
    for v, v_info in nodes.items():
        print(f"    {v} {v_info}")

    # 中心性算出結果出力
    # グラフにする
    # make_centrality_graph(output_path, nodes, detour, exist_critical_node)
    centrality_graph(data)


    print("------ end cal_centrality ------\n")
    return nodes


# 中心性のグラフ化
def centrality_graph(data):
    nodes = data["nodes"]
    detour = data["detour"]
    exist_critical_node = data["exist_critical_node"]

    params = data["draw_params"]["cal_centrality"]["centrality_graph"]

    fig = plt.figure(figsize=params["figsize"])
    plt.title(params["title"])
    plt.xlabel(params["xlabel"])
    plt.ylabel(params["ylabel"])

    # 軸データ
    x = []
    y_including_critical_node = []
    y_not_including_critical_node = []

    for v, v_info in nodes.items():
        if v_info["type"] == "between":
            x.append(v)
            y_including_critical_node.append(v_info["C"][0])
            if exist_critical_node:
                if v_info["C"][1] == None:
                    y_not_including_critical_node.append(0.0)
                else:
                    y_not_including_critical_node.append(v_info["C"][1])
    if exist_critical_node:
        left = np.arange(len(y_including_critical_node))
        width = params["width"] / 2.0
        plt.bar(left, y_including_critical_node, linewidth=0, align="center", color=params["color"]["color1"], width=width, label="including critical node")
        plt.bar(left+width, y_not_including_critical_node, linewidth=0, align="center", color=params["color"]["color2"], width=width, label="not including critical node")
        plt.xticks(left + width/2, x)
        # plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=2)
        plt.legend(loc="upper right")
    else:
        left = np.arange(len(y_including_critical_node))
        width = params["width"]
        plt.bar(left, y_including_critical_node, linewidth=0, align="center", color=params["color"]["color1"], width=width)
        plt.xticks(left, x)
    
    # fpath = output_path + "/Centrality.png"
    for extension in data["draw_params"]["extensions"]:
        fpath = os.path.join(data["output_path"], "Imgs", extension, "Centrality"+"."+extension)
        plt.savefig(fpath)

# RPscoreの計算
def RPscore(path, G):
    score = 0.0
    score = 1.0 / ETX(path, G)
    return score

# 媒介度の計算
def Cbet(v, detour):
    N = len(detour) # detourの合計経路数
    count_v_in_detour = Cbet_without_normalize(v, detour)
    return count_v_in_detour / N

# 媒介度の計算(正規化前)
def Cbet_without_normalize(v, detour):
    # vがdetour内の経路のうち何本の経路に含まれるか
    count_v_in_detour = 0
    for p in detour:
        if v in p["path"]["nodetype"]:
            count_v_in_detour += 1
    return count_v_in_detour


# detour内のノード集合nodesの取得
def nodes_in_detour(detour):
    nodes = {} # detour内のノード集合(辞書型)
    for p in detour:
        for v in p["path"]["nodetype"]:
            nodes[v] = {"type": None}
    return nodes

# ETX
def ETX(path, G):
    etx = 0.0
    for (u, v) in path["edgetype"]:
        etx += 1.0 / G[u][v]["rate"]
    return etx