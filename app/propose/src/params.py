import json
import pickle


def save_params(fpath, params):
    with open(fpath, "w") as f:
        json.dump(params, f)

def load_params(fname):
    fpath = "propose/data/data_params/" + fname + ".json"
    with open(fpath, "r") as f:
        params = json.load(f)
    return params

#######################################################

# 描画用パラメータファイルの読み込み
def load_draw_params(fname):
    fpath = "propose/data/data_draw_params/" + fname + ".json"
    with open(fpath, "r") as f:
        params = json.load(f)
    return params

#########################################################

# モデル作成用パラメータファイルの保存
def save_model_params(fpath, model_params):
    with open(fpath, "w") as f:
        json.dump(model_params, f)

# モデル作成用パラメータファイルの読み込み
def load_model_params(fname):
    fpath = "propose/data/data_model_params/" + fname + ".json"
    with open(fpath, "r") as f:
        params = json.load(f)
    return params

# モデル作成用パラメータファイルの作成
def make_model_params(seed, length_params, rate_params):
    model_params = {
        "seed": seed,
        "length_params": length_params,
        "rate_params": rate_params
    }
    return model_params

#############################################################

def load_input_params(fname):
    fpath = "propose/data/input/" + fname + ".json"
    with open(fpath, "r") as f:
        params = json.load(f)
    return params

def make_input_params(model_name, src, dst, threshold):

    # output_name = f"{model_name}-{src}-{dst}-{threshold}"
    
    # paramsの設定
    input_params = {
        "input": {
            "model_name": model_name,
            "src": src,
            "dst": dst,
            "threshold": threshold
        }
    }

    # modelの読み込み
    srcpath = "propose/data/data_model/" + model_name + "/" + model_name +".pickle"
    with open(srcpath, "rb") as f:
        model = pickle.load(f)

    # print(model)
    input_params["N"] = model["N"]
    input_params["mould"] = model["mould"]
    input_params["model"] = model["model"]


    input_params["draw_params"] = model["draw_params"]


    input_params_name = f"{model_name}-{src}-{dst}-{threshold}"
    fpath = "propose/data/input/" + input_params_name + ".json"
    with open(fpath, "w") as f:
        json.dump(input_params, f)

#

if __name__ == "__main__":

    # model作成時の 経路長/通信成功率 用のパラメータファイルの作成
    # m = 1
    # M = 100
    # seed = 100
    # model_params = make_model_params(
    #     seed=seed,
    #     length_params={"m": 1, "M": 20},
    #     rate_params={"ave": 0.9, "scale": 0.05}
    # )
    # fpath = "propose/data/input/data_model_params/" + f"seed-{seed}-len-{m}-{M}" + ".json"
    # save_model_params(fpath, model_params)

    # make_input_params("ba30-3-1-20_model", 2, 12, 20)

    pass
