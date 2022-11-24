import os
import util
import networkx as nx

info = {}
all_format = "png"
# dataset
dataset_name = "DATASET002"
info["dataset_name"] = dataset_name
data_dir_path = os.path.join("bib", "data")
dataset_dir_path = os.path.join(data_dir_path, dataset_name)
# input
input_dir_name = "input"
input_dir_path = os.path.join(dataset_dir_path, input_dir_name)
# output
output_dir_name = "output"
output_dir_path = os.path.join(dataset_dir_path, output_dir_name)
info["output_dir_path"] = output_dir_path

# fromsrc
info["fromsrc"] = {}
fromsrc_dir_name = "fromsrc"
info["fromsrc"]["verbose"] = True
info["fromsrc"]["dir_path"] = os.path.join(output_dir_path, fromsrc_dir_name)
# info["fromsrc_dir_path"] = os.path.join(output_dir_path, fromsrc_dir)

# fromdst
info["fromdst"] = {}
fromdst_dir_name = "fromdst"
info["fromdst"]["verbose"] = True
info["fromdst"]["dir_path"] = os.path.join(output_dir_path, fromdst_dir_name)
# settings
settings_data_name = "settings.txt"
settings_data_path = os.path.join(input_dir_path, settings_data_name)
settings_dict = util.read_settings(settings_data_path)
# src, dst
info["src"] = settings_dict["src"]
info["dst"] = settings_dict["dst"]
# threshold
info["threshold"] = settings_dict["threshold"]
# pos
pos_data_name = "pos.txt"
pos_data_path = os.path.join(input_dir_path, pos_data_name)
info["default_pos"] = util.read_pos(pos_data_path)

# 双方向ダイクストラ法
info["bib_method_verbose"] = False
# find k-path
info["cycle"] = False # Cycleを含むか
info["overlap_verbose"] = False
info["k-path_verbose"] = False
# rate
info["rate"] = {}
info["rate"]["distribution"] = "case1"
info["rate"]["digit"] = 3
info["rate"]["seed"] = 400

# Original Graph(G)
graph_G_name = "graph.txt"
graph_G_path = os.path.join(input_dir_path, graph_G_name)
info["G"] = {}
info["G"]["verbose"] = True
info["G"]["data"] = nx.read_weighted_edgelist(graph_G_path, nodetype=str)
info["G"]["pos"] = info["default_pos"]
info["G"]["title"] = f"Original Graph[{dataset_name}]"
info["G"]["alpha"] = 0.7
info["G"]["nodesize"] = 800
info["G"]["figsize"] = (8, 8)
# info["G"]["format"] = all_format if all_format != None else "png"
info["G"]["format"] = ["png", "svg", "pdf"]
# Hmid
info["Hmid"] = {}
info["Hmid"]["verbose"] = False
Hmid_graph_data_name = "Hmid"
info["Hmid"]["data_path"] = os.path.join(output_dir_path, Hmid_graph_data_name)
info["Hmid"]["format"] = all_format if all_format != None else "png"
# Gs
info["Gs"] = {}
info["Gs"]["flag"] = True
info["Gs"]["title"] = f"Gs[{dataset_name}]"
info["Gs"]["alpha"] = 0.7
info["Gs"]["nodesize"] = 800
info["Gs"]["figsize"] = (4, 4)
info["Gs"]["format"] = all_format if all_format != None else "png"
# Gt
info["Gt"] = {}
info["Gt"]["flag"] = True
info["Gt"]["title"] = f"Gt[{dataset_name}]"
info["Gt"]["alpha"] = 0.7
info["Gt"]["nodesize"] = 800
info["Gt"]["figsize"] = (6, 6)
info["Gt"]["format"] = all_format if all_format != None else "png"
# SPT_s
info["SPT_s"] = {}
info["SPT_s"]["flag"] = True
info["SPT_s"]["title"] = f"SPT from {info['src']} & sidetrack in Gs [{dataset_name}]"
info["SPT_s"]["alpha"] = 0.7
info["SPT_s"]["nodesize"] = 800
info["SPT_s"]["figsize"] = (6, 6)
info["SPT_s"]["format"] = all_format if all_format != None else "png"
info["SPT_s"]["edge_color"] = "red"
info["SPT_s"]["edge_width"] = 3.0
info["SPT_s"]["arrowsize"] = 15
info["sidetrack_s"] = {}
info["sidetrack_s"]["flag"] = True
info["sidetrack_s"]["alpha"] = info["SPT_s"]["alpha"] * 0.3
info["sidetrack_s"]["edge_color"] = "green"
info["sidetrack_s"]["edge_width"] = 4.0
info["sidetrack_s"]["arrowsize"] = 20

# SPT_t
info["SPT_t"] = {}
info["SPT_t"]["flag"] = True
info["SPT_t"]["title"] = f"SPT from {info['dst']} & sidetrack in Gt[{dataset_name}]"
info["SPT_t"]["alpha"] = 0.7
info["SPT_t"]["nodesize"] = 800
info["SPT_t"]["figsize"] = (6, 6)
info["SPT_t"]["format"] = all_format if all_format != None else "png"
info["SPT_t"]["edge_color"] = "red"
info["SPT_t"]["edge_width"] = 3.0
info["SPT_t"]["arrowsize"] = 15
info["sidetrack_t"] = {}
info["sidetrack_t"]["flag"] = True
info["sidetrack_t"]["alpha"] = info["SPT_t"]["alpha"] * 0.3
info["sidetrack_t"]["edge_color"] = "green"
info["sidetrack_t"]["edge_width"] = 4.0
info["sidetrack_t"]["arrowsize"] = 20

info["H"] = {
    "s": {"step1": {}, "step2": {}, "step3": {}}, 
    "t": {"step1": {}, "step2": {}, "step3": {}}
    }
# Hs (step1)
info["H"]["s"]["step1"]["flag"] = True
Hs_step1_dir_name = f"Hs(step1)[{dataset_name}]"
info["H"]["s"]["step1"]["dir_path"] = os.path.join(info["fromsrc"]["dir_path"], Hs_step1_dir_name)
info["H"]["s"]["step1"]["format"] = all_format if all_format != None else "png"
# Hs (step2)
info["H"]["s"]["step2"]["flag"] = True
Hs_step2_dir_name = f"Hs(step2) [{dataset_name}]"
info["H"]["s"]["step2"]["dir_path"] = os.path.join(info["fromsrc"]["dir_path"], Hs_step2_dir_name)
info["H"]["s"]["step2"]["format"] = all_format if all_format != None else "png"
# Hs (step3)
info["H"]["s"]["step3"]["flag"] = True
Hs_step3_dir_name = f"Hs(step3) [{dataset_name}]"
info["H"]["s"]["step3"]["dir_path"] = os.path.join(info["fromsrc"]["dir_path"], Hs_step3_dir_name)
info["H"]["s"]["step3"]["format"] = all_format if all_format != None else "png"
# Ht (step1)
info["H"]["t"]["step1"]["flag"] = True
Hs_step1_dir_name = f"Ht(step1)[{dataset_name}]"
info["H"]["t"]["step1"]["dir_path"] = os.path.join(info["fromdst"]["dir_path"], Hs_step1_dir_name)
info["H"]["t"]["step1"]["format"] = all_format if all_format != None else "png"
# Ht (step2)
info["H"]["t"]["step2"]["flag"] = True
Hs_step2_dir_name = f"Ht(step2) [{dataset_name}]"
info["H"]["t"]["step2"]["dir_path"] = os.path.join(info["fromdst"]["dir_path"], Hs_step2_dir_name)
info["H"]["t"]["step2"]["format"] = all_format if all_format != None else "png"
# Ht (step3)
info["H"]["t"]["step3"]["flag"] = True
Hs_step3_dir_name = f"Ht(step3) [{dataset_name}]"
info["H"]["t"]["step3"]["dir_path"] = os.path.join(info["fromdst"]["dir_path"], Hs_step3_dir_name)
info["H"]["t"]["step3"]["format"] = all_format if all_format != None else "png"

# パスグラフ
info["P"] = {}
info["P"]["flag"] = False
info["P"]["title"] = f"PathGraph[{dataset_name}]"
info["P"]["alpha"] = 0.7
info["P"]["arrowalpha"] = 0.4
info["P"]["nodesize"] = 500
info["P"]["figsize"] = (30, 30)
info["P"]["format"] = all_format if all_format != None else "png"
info["P"]["edge_width"] = 3.0
info["P"]["arrowsize"] = 15
info["Q"] = {}
info["Q"]["flag"] = False
info["Q"]["dir_path"] = os.path.join(output_dir_path, f"Q[{dataset_name}]")
info["Q"]["alpha"] = 0.7
info["Q"]["arrowalpha"] = 0.4
info["Q"]["pos"] = "planar" # planar/shell
info["Q"]["nodesize"] = 500
info["Q"]["figsize"] = (15, 15)
info["Q"]["format"] = all_format if all_format != None else "png"
info["Q"]["edge_width"] = 3.0
info["Q"]["arrowsize"] = 15
#
info["P"]["RootNodeName"] = "Root"
info["P"]["Root-Hmid"] = {}
info["P"]["Root-Hmid"]["type"] = "Root-Hmid"
info["P"]["Root-Hmid"]["color"] = "red"
info["P"]["Hmid"] = {}
info["P"]["Hmid"]["type"] = "Hmid"
info["P"]["Hmid-Hmid"] = {}
info["P"]["Hmid-Hmid"]["type"] = "Hmid-Hmid"
info["P"]["Hmid-Hmid"]["color"] = "green"
info["P"]["Hmid-HsRoot"] = {}
info["P"]["Hmid-HsRoot"]["type"] = "Hmid-HsRoot"
info["P"]["Hmid-HsRoot"]["color"] = "blue"
info["P"]["Hmid-HtRoot"] = {}
info["P"]["Hmid-HtRoot"]["type"] = "Hmid-HtRoot"
info["P"]["Hmid-HtRoot"]["color"] = "orange"
info["P"]["Hs-Hs"] = {}
info["P"]["Hs-Hs"]["type"] = "Hs-Hs"
info["P"]["Hs-Hs"]["color"] = "gray"
info["P"]["Ht-Ht"] = {}
info["P"]["Ht-Ht"]["type"] = "Ht-Ht"
info["P"]["Ht-Ht"]["color"] = "pink"
info["P"]["Hs-HtRoot"] = {}
info["P"]["Hs-HtRoot"]["type"] = "Hs-HtRoot"
info["P"]["Hs-HtRoot"]["color"] = "black"

# detour
info["detour"] = {}
info["detour"]["flag"] = True
info["detour"]["dir_name"] = "detour"
info["detour"]["dir_path"] = os.path.join(output_dir_path, info["detour"]["dir_name"] )
info["detour"]["figsize"] = info["G"]["figsize"]
info["detour"]["nodesize"] = info["G"]["nodesize"]
info["detour"]["alpha"] = 0.4
info["detour"]["edge_color"] = "red"
info["detour"]["edge_width"] = 12
info["detour"]["arrowsize"] = 15
info["detour"]["format"] = ["png", "svg", "pdf"]

# cent
info["cent"] = {}
info["cent"]["dir_name"] = "detour"
info["cent"]["dir_path"] = os.path.join(output_dir_path, info["cent"]["dir_name"])
info["cent"]["format"] = ["png", "svg", "pdf"]
info["cent"]["bias"] = 3.0