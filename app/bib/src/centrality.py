import util, settings
import random
import numpy as np
from copy import deepcopy

# 通信成功率の割り当て
def cal_rate(G, settings):
    np.random.seed(settings.info["rate"]["seed"])
    for (u, v) in G.edges():
        # 分布の設定
        if settings.info["rate"]["distribution"] == "uniform":
            r = np.random.rand()
        elif settings.info["rate"]["distribution"] == "case1":
            r = np.random.normal(loc=0.9, scale=0.05)
            if r > 1.0:
                r = 1.0
        else:
            r = np.random.rand()
        G[u][v]["rate"] = round(r, settings.info["rate"]["digit"])

# RPスコアの計算
def RPscore(path, G):
    score = 0.0
    for e in util.nodetype2edgetype(path):
        u, v = e
        score += 1.0 / G[u][v]["rate"]
    return score

def cal_Cbet(nodes, detour, settings):
    N = len(detour)
    for v in nodes.keys():
        if nodes[v]["src"] != True and nodes[v]["dst"] != True:
            count = count_in_detour(v, detour)
            nodes[v]["Cbet"] = count / N
            if count == N:
                nodes[v]["critical_node"] = True
            else:
                nodes[v]["critical_node"] = False

def count_in_detour(v, detour):
    ret = 0
    for info in detour:
        if v in info["path"]:
            ret += 1
    return ret

def centrality(nodes, detour, settings):
    # max_C = 0
    list_1 = []
    exist_critical_node = False # critical nodeが存在するかどうか
    for v, v_info in nodes.items():
        if nodes[v]["src"] != True and nodes[v]["dst"] != True:
            tmp = 0.0 # sum(RP)
            for detour_info in detour:
                if v in detour_info["path"]:
                    tmp += 1.0 / detour_info["RP"]
            nodes[v]["C"] = {}
            nodes[v]["C"]["type0"] = nodes[v]["Cbet"] * tmp
            nodes[v]["C"]["type1"] = None
            nodes[v]["C"]["type2"] = None
            list_1.append(nodes[v]["C"]["type0"])
            if nodes[v]["critical_node"] == True:
                exist_critical_node = True

    print(exist_critical_node)
    list_2 = sorted(list(set(list_1)))[::-1]
    if exist_critical_node:
        max1, max2 = list_2[:2]
    else:
        max1 = list_2[0]
    # print(list_2)
    # print(max1, max2)
    for v, v_info in nodes.items():
        if nodes[v]["src"] != True and nodes[v]["dst"] != True:
            nodes[v]["C"]["type1"] = nodes[v]["C"]["type0"] / max1
            if exist_critical_node:
                if max2 > 0:
                    if nodes[v]["C"]["type0"] != max1:
                        nodes[v]["C"]["type2"] = nodes[v]["C"]["type0"] / max2
    return exist_critical_node
    

def centrality_v2(nodes, detour, settings, path_shortest):
    # max_C = 0
    list_1 = []
    exist_critical_node = False # critical nodeが存在するかどうか
    for v, v_info in nodes.items():
        if nodes[v]["src"] != True and nodes[v]["dst"] != True:
            tmp = 0.0 # sum(RP)
            for detour_info in detour:
                if v in detour_info["path"]:
                    tmp += 1.0 / detour_info["RP"]
            nodes[v]["C"] = {}
            if v in path_shortest:
                nodes[v]["C"]["type0"] = nodes[v]["Cbet"] * tmp * settings.info["cent"]["bias"]
            else:            
                nodes[v]["C"]["type0"] = nodes[v]["Cbet"] * tmp
            nodes[v]["C"]["type1"] = None
            nodes[v]["C"]["type2"] = None
            list_1.append(nodes[v]["C"]["type0"])
            if nodes[v]["critical_node"] == True:
                exist_critical_node = True

    print(exist_critical_node)
    list_2 = sorted(list(set(list_1)))[::-1]
    if exist_critical_node:
        max1, max2 = list_2[:2]
    else:
        max1 = list_2[0]
    # print(list_2)
    # print(max1, max2)
    for v, v_info in nodes.items():
        if nodes[v]["src"] != True and nodes[v]["dst"] != True:
            nodes[v]["C"]["type1"] = nodes[v]["C"]["type0"] / max1
            if exist_critical_node:
                if max2 > 0:
                    if nodes[v]["C"]["type0"] != max1:
                        nodes[v]["C"]["type2"] = nodes[v]["C"]["type0"] / max2
    return exist_critical_node
    