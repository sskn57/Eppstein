import networkx as nx
import matplotlib.pyplot as plt
# import random
import settings
import numpy as np

# WSモデル
# N:ノード数, k:次数, p:リンク張り替え確立, seed:シード値
def generate_WS_model(N, k, p, seed):
    G = nx.watts_strogatz_graph(N, k, p, seed)
    set_weight(G, settings)
    fig = plt.figure()
    nx.draw(G)
    fig.savefig("WS_model_sample.png")

# BAモデル
# N:ノード数, m:新規ノード接続リンク数
def generate_BA_model(N, m):
    G = nx.barabasi_albert_graph(N, m)
    set_weight(G, settings) # weightとrateの設定
    fig = plt.figure()
    nx.draw(G)
    fig.savefig("BA_model_sample.png")

# ERモデル
# HKモデル

def get_rand(settings):
    if settings.info["rate"]["distribution"] == "uniform":
        r = np.random.rand()
    elif settings.info["rate"]["distribution"] == "case1":
        r = np.random.normal(loc=0.9, scale=0.05)
        if r > 1.0:
            r = 1.0
    else:
        r = np.random.rand()
    return r

def set_weight(G, settings):
    np.random.seed(settings.info["rate"]["seed"])
    for (u, v) in G.edges():
        # print(u, v)
        # 経路長
        G[u][v]["weight"] = np.random.randint(1, 30)
        # 成功率
        r = get_rand(settings)
        # 
        G[u][v]["rate"] = round(r, settings.info["rate"]["digit"])
    for (u, v), val in G.edges().items():
        print(f"{u} - {v}: {val}")
    return G

if __name__ == "__main__":
    N = 30
    seed = 100
    # generate_WS_model(N, 4, 0.1, 100)
    generate_BA_model(N, 2)
