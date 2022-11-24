import os, sys
import pickle
import dill
import networkx as nx
from cal_detour import cal_detour
from cal_centrality import cal_centrality
from graph import load_model
from params import load_input_params, load_draw_params, make_input_params
import json

# critical nodeなしパターン
# input_params_name = "ba30-3-1-20_model-1-2-10"
# input_params_name = "ba30-3-1-20_model-2-12-20"
# input_params_name = "ba30-5-1-20_model-18-26-20"
# input_params_name = "ba30-5-1-20_model-18-26-20"


# modelの読み込みとsrc, dst, thresholdの設定
def initialize(input_params_name):

    # input
    # inputパラメータの読み込み
    input_params = load_input_params(input_params_name)
    # modelの選択
    data = load_model(input_params["input"]["model_name"])

    # input_paramsをdataに追加
    set_params_to_data(data, input_params)

    # output_pathの設定
    data["output_path"] = "propose/data/output/" + input_params_name
    # outputディレクトリの初期化
    initialize_output(data)
    return data

def initialize_output(data):

    if not os.path.exists(data["output_path"]):
        os.mkdir(data["output_path"])
    # logディレクトリ
    log_path = os.path.join(data["output_path"], "log")
    if not os.path.exists(log_path):
        os.mkdir(log_path)

    logging = True
    if logging:
        sys.stdout = open(os.path.join(log_path, "all_log"), 'w')

    # Imgsディレクトリ
    Imgs_path = os.path.join(data["output_path"], "Imgs")
    if not os.path.exists(Imgs_path):
        os.mkdir(Imgs_path)

    for extension in data["draw_params"]["extensions"]:
        # extension(svg/png/pdf)ディレクトリ
        extension_path = os.path.join(Imgs_path, extension)
        if not os.path.exists(extension_path):
            os.mkdir(extension_path)

def set_params_to_data(data, input_params):
    data["src"] = input_params["input"]["src"]
    data["dst"] = input_params["input"]["dst"]
    data["threshold"] = input_params["input"]["threshold"]
    # draw_paramsの読み込み
    fname = os.path.splitext(os.path.basename(input_params["draw_params"]["path"]))[0]
    data["draw_params"] = load_draw_params(fname)
    data["input_params"] = input_params


def finalize(data):
    # dataの書き出し
    with open(os.path.join(data["output_path"], "data.dill"), "wb") as f:
        # pickle.dump(data, f)
        dill.dump(data, f)

    sys.stdout = sys.__stdout__

def main():

    # (1-A) 新規作成
    model_name = "ws30-4-50-1-20_model"
    src = 28
    dst = 8
    threshold = 25
    make_input_params(model_name, src, dst, threshold)
    input_params_name = f"{model_name}-{src}-{dst}-{threshold}"

    # (1-B) 既存の入力パターンの使用
    # input_params_name = "ba30-3-1-20_model-1-2-10"


    # (2) 初期化
    data = initialize(input_params_name)

    # (3) 迂回経路の算出
    detour = cal_detour(data)

    # (4) 中心性の計算
    cal_centrality(data)

    # (5) 終了処理
    finalize(data)

if __name__ == "__main__":
    main()