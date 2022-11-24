import os
import pickle
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from params import load_draw_params, load_model_params


# mould(鋳型): graph, posのセット
# model: mouldに辺の情報(length, rate)を付与したもの


def gen_model(mould_name, model_name, model_params, draw_params):

    logpath = os.path.join(model_name, "log")

    # mouldの読み込み
    srcpath = "propose/data/data_mould/" + mould_name + "/" + mould_name +".pickle"
    with open(srcpath, "rb") as f:
        data = pickle.load(f)
    
    # 各辺のlength, rateの設定
    length_params = model_params["length_params"]
    rate_params = model_params["rate_params"]
    np.random.seed(model_params["seed"])
    for (u, v) in data["G"].edges():
        data["G"][u][v]["length"] = _length(m=length_params["m"], M=length_params["M"])
        data["G"][u][v]["rate"] = _rate(ave=rate_params["ave"], scale=rate_params["scale"])

    # # modelパラメータをdataに記録
    data["model"]= {}
    data["model"]["params"] = model_params

    data["draw_params"]={}
    data["draw_params"]["path"] = draw_params["path"]
    
    # ログ
    # with open(logpath, "w") as f:
    #     f.write(f"n:{data['n']}\n")
    #     f.write(f"seed:{seed}\n")
    #     f.write(f"length\n")
    #     f.write(f"  m:{m}\n")
    #     f.write(f"  M:{M}\n")
    #     f.write(f"rate\n")
    #     f.write(f"  ave:{ave}\n")
    #     f.write(f"  scale:{scale}\n")
    #     if any(data["mould"]["params"]):
    #         f.write(f"mould_params:\n")
    #         for key, val in data["mould"]["params"].items():
    #             f.write(f"  {key}:{val}\n")
    
    # model出力先の設定
    dstpath = "propose/data/data_model/" + model_name
    data["model"]["path"] = dstpath
    if not os.path.exists(dstpath):
        os.mkdir(dstpath)

    # model(pickle)データの書き出し
    with open(os.path.join(dstpath, model_name+".pickle"), "wb") as f:
        pickle.dump(data, f)

    # Img書き出し
    imgspath = os.path.join(dstpath, "Imgs")
    if not os.path.exists(imgspath):
        os.mkdir(imgspath)


    extensions = draw_params["extensions"]
    labels = ["length", "rate"]

    for label in labels:
        for extension in extensions:
            fig = plt.figure(figsize=draw_params["gen_model"]["figsize"])
            # title
            plt.title(f"inputgraph({label})")
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
                edge_color=draw_params["gen_model"]["edge"]["color"],
                width=draw_params["gen_model"]["edge"]["width"],
                alpha=draw_params["gen_model"]["edge"]["alpha"])
            # 辺のラベル
            edge_labels = {(i, j): round(w[label], draw_params["gen_model"]["edge"]["digit"]) for i, j, w in data["G"].edges(data=True)}
            nx.draw_networkx_edge_labels(
                data["G"], 
                data["pos"], 
                edge_labels=edge_labels,
                font_color=draw_params["gen_model"]["edge"]["font"]["color"],
                font_size=draw_params["gen_model"]["edge"]["font"]["size"],
                alpha=draw_params["gen_model"]["edge"]["font"]["alpha"],
                bbox={"fc": (0, 0, 0, 0), "ec": (0, 0, 0, 0)},
                rotate=False)
            # node
            node_color = []
            for v in data["G"].nodes():
                node_color.append(draw_params["gen_model"]["node"]["color"]["default"])
            nx.draw_networkx_nodes(
                data["G"], 
                data["pos"], 
                node_color=node_color,
                node_size=draw_params["gen_model"]["node"]["size"], 
                alpha=draw_params["gen_model"]["node"]["alpha"])

            # nodeのラベル
            nx.draw_networkx_labels(
                data["G"], 
                data["pos"], 
                font_color=draw_params["gen_model"]["node"]["font"]["color"], 
                font_size=draw_params["gen_model"]["node"]["font"]["size"],
                alpha=draw_params["gen_model"]["node"]["font"]["alpha"])


            fname = f"inputgraph({label})" + "." + extension
            if not os.path.exists(os.path.join(imgspath, extension)):
                os.mkdir(os.path.join(imgspath, extension))
            plt.savefig(os.path.join(imgspath, extension, fname), bbox_inches='tight')

# 各辺のlength
def _length(m, M):
    # m: 最小値
    # M: 最大値
    r = np.random.randint(m, M)
    return r

# 各辺のrate
def _rate(ave, scale):
    # ave: 正規分布の平均値
    # scale: 正規分布の標準偏差
    r = np.random.normal(loc=ave, scale=scale)
    if r > 1.0:
        r = 1.0
    return r

def load_model(model_name):
    srcpath = "propose/data/data_model/" + model_name + "/" + model_name +".pickle"
    with open(srcpath, "rb") as f:
        ret = pickle.load(f)
        return ret
    return None

###############################################################

# BA mould
def gen_BA_mould(mould_name, seed, N, m):
    # mould_name: mould(ディレクトリ)
    # N: ノード数
    # m: 新規ノード接続リンク数

    ret = {}
    ret["G"] = nx.barabasi_albert_graph(N, m, seed=seed)
    # ret["pos"] = nx.shell_layout(ret["G"])
    ret["pos"] = nx.spring_layout(ret["G"])
    ret["N"] = N
    ret["mould"] = {"type": "ba", "params": {"seed": seed}}
    ret["mould"]["params"]["m"] = m

    dstpath = "propose/data/data_mould/" + mould_name
    ret["mould"]["path"] = dstpath
    if not os.path.exists(dstpath):
        os.mkdir(dstpath)
    with open(os.path.join(dstpath, mould_name+".pickle"), "wb") as f:
        pickle.dump(ret, f)

# WS mould
def gen_WS_mould(mould_name, seed, N, k, p):
    # mould_name: mould(ディレクトリ)
    # N: ノード数
    # k: 次数
    # p: リンク張り替え確率

    ret = {}
    ret["G"] = nx.watts_strogatz_graph(N, k, p, seed=seed)
    # ret["pos"] = nx.shell_layout(ret["G"])
    ret["pos"] = nx.spring_layout(ret["G"])
    ret["N"] = N
    ret["mould"] = {"type": "ws", "params": {"seed": seed}}
    ret["mould"]["params"]["k"] = k
    ret["mould"]["params"]["p"] = p

    dstpath = "propose/data/data_mould/" + mould_name
    ret["mould"]["path"] = dstpath
    if not os.path.exists(dstpath):
        os.mkdir(dstpath)
    with open(os.path.join(dstpath, mould_name+".pickle"), "wb") as f:
        pickle.dump(ret, f)

# HK mould

# SB mould
def gen_SB_mould(mould_name, seed, sizes, p1, p2):
    # mould_name: mould(ディレクトリ)

    props = []
    for i in range(len(sizes)):
        tmp = []
        for j in range(len(sizes)):
            if i == j:
                tmp.append(p1)
            else:
                tmp.append(p2)
        props.append(tmp)

    ret = {}
    ret["G"] = nx.stochastic_block_model(sizes, props)
    # ret["pos"] = nx.shell_layout(ret["G"])
    ret["pos"] = nx.spring_layout(ret["G"])
    ret["N"] = sum(sizes)
    ret["mould"] = {"type": "sb", "params": {"seed": seed}}
    ret["mould"]["params"]["sizes"] = sizes
    ret["mould"]["params"]["p1"] = p1
    ret["mould"]["params"]["p2"] = p2

    dstpath = "propose/data/data_mould/" + mould_name
    ret["mould"]["path"] = dstpath
    if not os.path.exists(dstpath):
        os.mkdir(dstpath)
    with open(os.path.join(dstpath, mould_name+".pickle"), "wb") as f:
        pickle.dump(ret, f)
# https://networkx.org/documentation/stable/reference/generated/networkx.generators.community.stochastic_block_model.html


# HK mould


# Refer
# https://buildersbox.corp-sansan.com/entry/2020/12/07/110000

# data_edgelistからdata_mouldの生成
def gen_from_edgelist_to_mould(srcpath, dstpath):
    # srcpath: edgelistデータディレクトリへのパス
    # dstpath: mouldデータへのパス
    edgelist_file_name = "edgelist.txt"
    pos_file_name = "pos.txt"
    ret = {}
    ret["G"] = nx.read_edgelist(os.path.join(srcpath, edgelist_file_name), nodetype=str)
    ret["pos"] = load_pos(os.path.join(srcpath, pos_file_name))
    ret["mould"] = {"params": {}}
    ret["n"] = len(ret["G"].nodes())
    with open(dstpath, "wb") as f:
        pickle.dump(ret, f)

##########################################################

# posの読み取り
def load_pos(srcpath, delim=" "):
    dic = {}
    with open(srcpath) as f:
        for line in f.readlines():
            data = line.split(delim)
            dic[data[0]] = (int(data[1].rstrip("\n")), int(data[2].rstrip("\n")))
        return dic

###############################################################

if __name__ == "__main__":
##### BA model #####
    # seed = 100
    # N = 500
    # m = 3

    # name = f"ba{N}-{m}"
    # mould_name = name + "_mould"
    # gen_BA_mould(mould_name, seed=seed, N=N, m=m)
    # # model作成時の 経路長/通信成功率 用のパラメータ

    # model_params = load_model_params("seed-100-len-1-20") # 辺のパラメータ
    # draw_params = load_draw_params(f"{N}nodes") # 描画用
    # model_name = name + "-1-20" + "_model"
    # gen_model(mould_name, model_name, model_params, draw_params)

##### WS model #####
    seed = 100
    N = 500
    k = 5
    p = 0.2

    name = f"ws{N}-{k}-{int(100*p)}"
    mould_name = name + "_mould"
    gen_WS_mould(mould_name, N=N, k=k, p=p, seed=seed)

    # model作成時の 経路長/通信成功率 用のパラメータ
    model_params = load_model_params("seed-100-len-1-20") # 辺のパラメータ
    draw_params = load_draw_params(f"{N}nodes") # 描画用
    model_name = name + "-1-20" + "_model"
    gen_model(mould_name, model_name, model_params, draw_params)


##### SB model #####
    # seed = 500
    # # sizes = [10, 10, 10]
    # sizes = [100, 100, 100, 50, 50, 50, 50]
    # p1 = 0.50
    # p2 = 0.05

    # N = sum(sizes)
    # name = f"SB{N}-{len(sizes)}-{int(100*p1)}-{int(100*p2)}"
    # mould_name = name + "_mould"
    # gen_SB_mould(mould_name, seed=seed, sizes=sizes, p1=p1, p2=p2)
    # # model作成時の 経路長/通信成功率 用のパラメータ
    # model_params = load_model_params("seed-100-len-1-100") # 辺のパラメータ
    # draw_params = load_draw_params(f"{N}nodes") # 描画用
    # model_name = name + "-1-100" + "_model"
    # gen_model(mould_name, model_name, model_params, draw_params)


##### el1 #####
    # srcpath = "propose/data/input/data_edgelist/" + "edgelist1"
    # dstpath = "propose/data/input/data_mould/" +  "el1_mould.pickle"
    # gen_from_edgelist_to_mould(srcpath, dstpath)

    # srcpath = "propose/data/input/data_mould/" + "el1_mould.pickle"
    # dstpath = "propose/data/input/data_model/" + "el1_model"
    # seed = 10
    # length_params = {"m": 5, "M": 30}
    # rate_params = {"ave": 0.9, "scale": 0.05}
    # draw_params = {"figsize": (6, 5), "node_size": 500}
    # gen_model(srcpath, dstpath, seed, length_params, rate_params, draw_params)
    
    pass
