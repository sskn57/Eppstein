import networkx as nx
import matplotlib.pyplot as plt
import os

# pos読み込み用
def read_pos(fpath, delim=" "):
  dic = {}
  with open(fpath) as f:
    for line in f.readlines():
      data = line.split(delim)
      dic[data[0]] = (int(data[1]), int(data[2]))
  return dic

def read_settings(fpath, delim=" "):
    with open(fpath) as f:
        # dat = f.readlines()
        ret = {}
        for line in f.readlines():
            dat = line.split(delim)
            if isfloat(dat[1].rstrip("\n")):
                ret[dat[0]] = float(dat[1].rstrip("\n"))
            else:
                ret[dat[0]] = dat[1].rstrip("\n")
        if len(ret.keys()):
            return ret
        else:
            return None

def isfloat(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True

class DrawGraph:
    def __init__(self, graph, settings):
        self.graph = graph
        self.settings = settings
        # グラフごとにsettings.pyから読み込む
        self.title = None
        self.alpha = None
        self.nodesize = None
        self.figsize = None

        self.fig = None
        self.ax = None
    
    def set_pos(self, pos):
        self.pos = pos

    def draw(self):
        pass

class DrawOriginalGraph(DrawGraph):
    def __init__(self, graph, settings):
        super().__init__(graph, settings)
        self.info = self.settings.info["G"]

        self.fig = plt.figure(figsize=self.info["figsize"])
        self.ax = self.fig.add_subplot()
        self.ax.set_title(self.info["title"])

        self.set_pos(self.info["pos"])

    def draw(self, weight_label):
        if self.info["flag"]:
            # 辺の重み
            edge_labels = {(i, j): w[weight_label] for i, j, w in self.graph.edges(data=True)}
            nx.draw_networkx_edge_labels(self.graph, self.pos, edge_labels=edge_labels)
            # 辺と頂点
            nx.draw_networkx(self.graph, self.pos, with_labels=True, alpha=self.info["alpha"], node_size=self.info["nodesize"])
            # 保存
            fname = self.info["title"] + f"({weight_label})" + "." + self.info["format"]
            output_path = os.path.join(self.settings.info["output_dir_path"], fname)
            plt.savefig(output_path)

class DrawGs(DrawGraph):
    def __init__(self, graph, settings, pos):
        super().__init__(graph, settings)
        self.info = self.settings.info["Gs"]

        self.fig = plt.figure(figsize=self.info["figsize"])
        self.ax = self.fig.add_subplot()
        self.ax.set_title(self.info["title"])

        self.set_pos(pos)

    def draw(self, G):
        if self.info["flag"]:
            nx.draw_networkx_nodes(
                G, 
                pos=self.settings.info["G"]["pos"], 
                alpha=self.settings.info["G"]["alpha"], 
                node_size=self.settings.info["G"]["nodesize"]
                ) # for G
            nx.draw_networkx_labels(
                G, 
                pos=self.settings.info["G"]["pos"], 
                alpha=self.settings.info["G"]["alpha"]
                ) # for G
            # 辺の重み
            edge_labels = {(i, j): w['weight'] for i, j, w in self.graph.edges(data=True)}
            nx.draw_networkx_edge_labels(self.graph, self.pos, edge_labels=edge_labels)
            # 辺と頂点
            nx.draw_networkx(self.graph, self.pos, with_labels=True, alpha=self.info["alpha"], node_size=self.info["nodesize"])
            # 保存
            fname = self.info["title"] + "." + self.info["format"]
            output_path = os.path.join(self.settings.info["fromsrc_dir_path"], fname)
            plt.savefig(output_path)



class DrawGt(DrawGraph):
    def __init__(self, graph, settings, pos):
        super().__init__(graph, settings)
        self.info = self.settings.info["Gt"]

        self.fig = plt.figure(figsize=self.info["figsize"])
        self.ax = self.fig.add_subplot()
        self.ax.set_title(self.info["title"])

        self.set_pos(pos)

    def draw(self, G):
        if self.info["flag"]:
            nx.draw_networkx_nodes(
                G, 
                pos=self.settings.info["G"]["pos"], 
                alpha=self.settings.info["G"]["alpha"], 
                node_size=self.settings.info["G"]["nodesize"]
                ) # for G
            nx.draw_networkx_labels(
                G, 
                pos=self.settings.info["G"]["pos"], 
                alpha=self.settings.info["G"]["alpha"]
                ) # for G
            # 辺の重み
            edge_labels = {(i, j): w['weight'] for i, j, w in self.graph.edges(data=True)}
            nx.draw_networkx_edge_labels(self.graph, self.pos, edge_labels=edge_labels)
            # 辺と頂点
            nx.draw_networkx(self.graph, self.pos, with_labels=True, alpha=self.info["alpha"], node_size=self.info["nodesize"])
            # 保存
            fname = self.info["title"] + "." + self.info["format"]
            output_path = os.path.join(self.settings.info["fromdst_dir_path"], fname)
            plt.savefig(output_path)



class DrawSPT_s(DrawGraph):
    def __init__(self, graph, settings, pos):
        super().__init__(graph, settings)
        self.info = self.settings.info["SPT_s"]

        self.fig = plt.figure(figsize=self.info["figsize"])
        self.ax = self.fig.add_subplot()
        self.ax.set_title(self.info["title"])

        self.set_pos(pos)

    def draw(self, G, Gs, path_shortest):
        if self.info["flag"]:
            # Gのノード描画
            nx.draw_networkx_nodes(
                G, 
                pos=self.settings.info["G"]["pos"], 
                alpha=self.settings.info["G"]["alpha"], 
                node_size=self.settings.info["G"]["nodesize"]
                )
            nx.draw_networkx_labels(
                G, 
                pos=self.settings.info["G"]["pos"], 
                alpha=self.settings.info["G"]["alpha"]
                )

            # 最短経路木 SPT_s
            edge_labels = {(i, j): w['weight'] for i, j, w in self.graph.edges(data=True)}
            nx.draw_networkx_edge_labels(
                self.graph, 
                self.pos, 
                edge_labels=edge_labels
                )
            nx.draw_networkx_edges(
                self.graph, 
                self.pos, 
                edgelist=self.graph.edges(), 
                edge_color=self.info["edge_color"], 
                alpha=self.info["alpha"], 
                width=self.info["edge_width"], 
                arrowsize=self.info["arrowsize"]
                )
            # サイドトラックの算出
            sidetrack_edge_list = []
            for (u, v) in Gs.edges():
                if (u, v) not in self.graph.edges():
                    if u in path_shortest and v in path_shortest:
                        continue
                    sidetrack_edge_list.append((u, v))
            # サイドトラックの描画
            nx.draw_networkx_edges(
                Gs, 
                self.pos, 
                edgelist=sidetrack_edge_list, 
                edge_color=self.settings.info["sidetrack_s"]["edge_color"], 
                alpha=self.settings.info["sidetrack_s"]["alpha"], 
                width=self.settings.info["sidetrack_s"]["edge_width"],
                arrowsize=self.settings.info["sidetrack_s"]["arrowsize"]
                )
            # 辺と頂点
            edge_labels = {(i, j): w['weight'] for i, j, w in Gs.edges(data=True)}
            nx.draw_networkx_edge_labels(G, self.pos, edge_labels=edge_labels)
            nx.draw_networkx(self.graph, self.pos, with_labels=True, alpha=self.info["alpha"], node_size=self.info["nodesize"])
            # 保存
            fname = self.info["title"] + "." + self.info["format"]
            output_path = os.path.join(self.settings.info["fromsrc_dir_path"], fname)
            plt.savefig(output_path)

class DrawSPT_t(DrawGraph):
    def __init__(self, graph, settings, pos):
        super().__init__(graph, settings)
        self.info = self.settings.info["SPT_t"]

        self.fig = plt.figure(figsize=self.info["figsize"])
        self.ax = self.fig.add_subplot()
        self.ax.set_title(self.info["title"])

        self.set_pos(pos)

    def draw(self, G, Gt, path_shortest):
        if self.info["flag"]:
            # Gのノード描画
            nx.draw_networkx_nodes(
                G, 
                pos=self.settings.info["G"]["pos"], 
                alpha=self.settings.info["G"]["alpha"], 
                node_size=self.settings.info["G"]["nodesize"]
                )
            nx.draw_networkx_labels(
                G, 
                pos=self.settings.info["G"]["pos"], 
                alpha=self.settings.info["G"]["alpha"]
                )
            # 最短経路木 SPT_t
            edge_labels = {(i, j): w['weight'] for i, j, w in self.graph.edges(data=True)}
            nx.draw_networkx_edge_labels(
                self.graph, 
                self.pos, 
                edge_labels=edge_labels
                )
            nx.draw_networkx_edges(
                self.graph, 
                self.pos, 
                edgelist=self.graph.edges(), 
                edge_color=self.info["edge_color"], 
                alpha=self.info["alpha"], 
                width=self.info["edge_width"], 
                arrowsize=self.info["arrowsize"]
                )  
            sidetrack_edge_list = []
            for (u, v) in Gt.edges():
                if (u, v) not in self.graph.edges():
                    if u in path_shortest and v in path_shortest: # poptと一致する辺はsidetrackではない
                        continue
                    if u == self.settings.info["dst"]:# dstから出るsidetrackは不適
                        continue
                    sidetrack_edge_list.append((u, v))
            # サイドトラックの描画
            nx.draw_networkx_edges(
                Gt, 
                self.pos, 
                edgelist=sidetrack_edge_list, 
                edge_color=self.settings.info["sidetrack_t"]["edge_color"], 
                alpha=self.settings.info["sidetrack_t"]["alpha"], 
                width=self.settings.info["sidetrack_t"]["edge_width"],
                arrowsize=self.settings.info["sidetrack_t"]["arrowsize"]
                )
            # 辺と頂点
            edge_labels = {(i, j): w['weight'] for i, j, w in Gt.edges(data=True)}
            nx.draw_networkx_edge_labels(G, self.pos, edge_labels=edge_labels)
            nx.draw_networkx(self.graph, self.pos, with_labels=True, alpha=self.info["alpha"], node_size=self.info["nodesize"])
            # 保存
            fname = self.info["title"] + "." + self.info["format"]
            output_path = os.path.join(self.settings.info["fromdst_dir_path"], fname)
            plt.savefig(output_path)


class DrawP(DrawGraph):
    def __init__(self, graph, settings):
        super().__init__(graph, settings)
        self.info = self.settings.info["P"]
        self.fig = plt.figure(figsize=self.info["figsize"])
        self.ax = self.fig.add_subplot()
        self.ax.set_title(self.info["title"])
        pos = nx.shell_layout(graph)
        self.set_pos(pos)

    def draw(self):
        if self.info["flag"]:
            # 辺の重み
            edge_labels = {(i, j): w['weight'] for i, j, w in self.graph.edges(data=True)}
            nx.draw_networkx_edge_labels(self.graph, self.pos, edge_labels=edge_labels)
            # 辺
            edge_color = [v["color"] for v in self.graph.edges.values()]
            nx.draw_networkx_edges(
                self.graph, 
                self.pos, 
                edgelist=self.graph.edges(), 
                edge_color=edge_color, 
                alpha=self.info["arrowalpha"], 
                width=self.info["edge_width"], 
                arrowsize=self.info["arrowsize"]
                )
            # 頂点
            nx.draw_networkx(self.graph, self.pos, with_labels=True, alpha=self.info["alpha"], node_size=self.info["nodesize"])
            # 保存
            fname = self.info["title"] + "." + self.info["format"]
            output_path = os.path.join(self.settings.info["output_dir_path"], fname)
            plt.savefig(output_path)

class DrawQ(DrawGraph):
    def __init__(self, graph, settings):
        super().__init__(graph, settings)
        self.info = self.settings.info["Q"]

    def draw(self):
        if self.info["flag"]:
            for v, graph in self.graph.items():
                self.fig = plt.figure(figsize=self.info["figsize"])
                self.ax = self.fig.add_subplot()
                # タイトル
                title = f"Q({v}) [{self.settings.info['dataset_name']}]"
                self.ax.set_title(title)
                # pos
                if self.info["pos"] == "shell":
                    pos = nx.shell_layout(graph)
                elif self.info["pos"] == "planar":
                    pos = nx.planar_layout(graph)
                else:
                    pos = nx.shell_layout(graph)
                self.set_pos(pos)
                # 辺の重み
                edge_labels = {(i, j): w['weight'] for i, j, w in graph.edges(data=True)}
                nx.draw_networkx_edge_labels(graph, self.pos, edge_labels=edge_labels)
                # 辺
                edge_color = [v["color"] for v in graph.edges.values()]
                nx.draw_networkx_edges(
                    graph, 
                    self.pos, 
                    edgelist=graph.edges(), 
                    edge_color=edge_color, 
                    alpha=self.info["arrowalpha"], 
                    width=self.info["edge_width"], 
                    arrowsize=self.info["arrowsize"]
                    )
                # 頂点
                nx.draw_networkx(graph, self.pos, with_labels=True, alpha=self.info["alpha"], node_size=self.info["nodesize"])
                # 保存
                fname = title + "." + self.info["format"]
                output_path = os.path.join(self.info["dir_path"], fname)
                plt.savefig(output_path)

# グラフG-Tにおいてhead/tailがノードvである辺, sidetrack(v)を取得する
def sidetrack(v, G, T):
    ret = []
    for (h, t) in G.edges():
        if (h, t) not in T.edges():
            if v == h:
                if (h, t) not in ret:
                    ret.append((h, t))
    return ret

def get_parent(stracks, popt):
    _, (h, t) = stracks[-1]
    if h in popt:
        return "Root"
    else:
        return stracks[-2]

def cycle(path):
    if type(path) == list:
        if type(path[0]) == tuple:
            # edgetype
            return None
        elif type(path[0]) == str:
            # nodetype
            if len(path) != len(set(path)):
                # 重複あり(cycleあり)
                return True
            else:
                return False

    elif type(path) == str:
        # strtype
        return None
    

# nodetype: ["F", "G", "H", "I"]
# edgetype: [("F", "G"), ("G", "H"), ("H", "I")]
# strtype : FGHI
def nodetype2edgetype(nodetype_path):
    return [e for e in zip(nodetype_path[0:], nodetype_path[1:])]

def edgetype2nodetype(edgetype_path):
    return [h for (h, t) in edgetype_path] + [edgetype_path[-1][1]]

def nodetype2strtype(nodetype_path):
    ret = "".join([str(_) for _ in nodetype_path])
    return ret 

# p_pos = nx.spring_layout(P)
# p_pos = nx.spiral_layout(P)
# p_pos = nx.random_layout(P)
# p_pos = nx.spectral_layout(P) # 円形
# p_pos = nx.planar_layout(P) # 辺が交わらない


def make_PathGraph(Hs, Ht, Hmid, settings):
    # RootNodeName = "Root"
    info = settings.info["P"]
    # RootNodeName = settings.info["P"]["RootNodeName"]
    RootNodeName = info["RootNodeName"]
    P = nx.DiGraph()
    Q = {}
    sidetrack_weight = {}

    # Rootの追加
    P.add_node(RootNodeName)
    if len(Hmid.data) > 0:
        # Root - Hmid's root
        child_val, child_node = Hmid.data[0] # Hmidのrootを取り出す
        w = child_val
        typ = info["Root-Hmid"]["type"]
        color = info["Root-Hmid"]["color"]
        P.add_edge(RootNodeName, child_node, weight=w, type=typ, color=color)

        # Hmid - Hmid
        for i, (parent_val, parent_name) in enumerate(Hmid.data):
            typ = info["Hmid"]["type"]
            P.add_node(parent_name, type=typ)
            # Qにも追加
            Q[parent_name] = nx.DiGraph()
            Q[parent_name].add_node(parent_name, type=typ)

            if 2*(i+1)-1 < len(Hmid.data):
                child_val, child_name = Hmid.data[2*(i+1)-1]
                w = 0.0 # Hmid - Hmid間はポテンシャル同じ
                typ = info["Hmid-Hmid"]["type"]
                color = info["Hmid-Hmid"]["color"]
                P.add_edge(parent_name, child_name, weight=w, type=typ, color=color)
            if 2*(i+1) < len(Hmid.data):
                child_val, child_name = Hmid.data[2*(i+1)]
                w = 0.0 # Hmid - Hmid間はポテンシャル同じ
                typ = info["Hmid-Hmid"]["type"]
                color = info["Hmid-Hmid"]["color"]
                P.add_edge(parent_name, child_name, weight=w, type=typ, color=color)
        
            # Hmid - Hs(step3)'s root
            if len(Hs[parent_name].data) > 0:
                sidetrack_val, sidetrack_edge = Hs[parent_name].data[0]
                child_node = (parent_name, sidetrack_edge)
                w = sidetrack_val
                typ = info["Hmid-HsRoot"]["type"]
                color = info["Hmid-HsRoot"]["color"]
                P.add_edge(parent_name, child_node, weight=w, type=typ, color=color)
                Q[parent_name].add_edge(parent_name, child_node, weight=w, type=typ, color=color)

            # Hmid - Ht(step3)'s root
            if len(Ht[parent_name].data) > 0:
                sidetrack_val, sidetrack_edge = Ht[parent_name].data[0]
                child_node = (parent_name, sidetrack_edge)
                w = sidetrack_val
                typ = info["Hmid-HtRoot"]["type"]
                color = info["Hmid-HtRoot"]["color"]
                P.add_edge(parent_name, child_node, weight=w, type=typ, color=color)
                Q[parent_name].add_edge(parent_name, child_node, weight=w, type=typ, color=color)
            
            # Hs(step3) - Hs(step3)
            # print(f"parent_name:{parent_name}")
            for j, (child_val, (child_u, child_v)) in enumerate(Hs[parent_name].data):
                # Hsを順に調べる．まず，ノードを追加
                child_node = (parent_name, (child_u, child_v))
                P.add_node(child_node)
                Q[parent_name].add_node(child_node)
                # print(f"    child_node:{child_node}")
                # サイドトラックの重みを記録
                sidetrack_weight[(child_u, child_v)] = child_val
                # 次に，辺(Hs-Hs)を追加
                if 2*(j+1)-1 < len(Hs[parent_name].data):
                    g_child_val, (g_child_u, g_child_v) = Hs[parent_name].data[2*(j+1)-1]
                    g_child_node =(parent_name, (g_child_u, g_child_v))
                    w = g_child_val - child_val
                    typ = info["Hs-Hs"]["type"]
                    color = info["Hs-Hs"]["color"]
                    P.add_edge(child_node, g_child_node, weight=w, type=typ, color=color)
                    Q[parent_name].add_edge(child_node, g_child_node, weight=w, type=typ, color=color)
                    # print(f"        g_child_node:{g_child_node}")
                if 2*(j+1) < len(Hs[parent_name].data):
                    g_child_val, (g_child_u, g_child_v) = Hs[parent_name].data[2*(j+1)]
                    g_child_node =(parent_name, (g_child_u, g_child_v))
                    w = g_child_val - child_val
                    typ = info["Ht-Ht"]["type"]
                    color = info["Ht-Ht"]["color"]
                    P.add_edge(child_node, g_child_node, weight=w, type=typ, color=color)
                    Q[parent_name].add_edge(child_node, g_child_node, weight=w, type=typ, color=color)
                    # print(f"        g_child_node:{g_child_node}")
                # Hsのノードchild_node:(child_u, child_v)について，Ht(child_v)が存在するとき，
                if child_v in Ht.keys():
                    if len(Ht[child_v].data) > 0:
                        # child_node - Ht(child_v)'s root
                        g_child_val, (g_child_u, g_child_v) = Ht[child_v].data[0] # root取り出し
                        g_child_node = (child_v, (g_child_u, g_child_v))
                        w = g_child_val
                        typ = info["Hs-HtRoot"]["type"]
                        color = info["Hs-HtRoot"]["color"]
                        P.add_edge(child_node, g_child_node, weight=w, type=typ, color=color)
                        Q[parent_name].add_edge(child_node, g_child_node, weight=w, type=typ, color=color)
                        # その後，Ht(child_v)内の辺を追加
                        for k, (g_child_val, (g_child_u, g_child_v)) in enumerate(Ht[child_v].data):
                            g_child_node = (child_v, (g_child_u, g_child_v))
                            P.add_node(g_child_node)
                            Q[parent_name].add_node(g_child_node)
                            sidetrack_weight[(g_child_u, g_child_v)] = g_child_val
                            if 2*(k+1)-1 < len(Ht[child_v].data):
                                gg_child_val, (gg_child_u, gg_child_v) = Ht[child_v].data[2*(k+1)-1]
                                gg_child_node = (child_v, (gg_child_u, gg_child_v))
                                w = gg_child_val - g_child_val
                                typ = info["Ht-Ht"]["type"]
                                color = info["Ht-Ht"]["color"]
                                P.add_edge(g_child_node, gg_child_node, weight=w, type=typ, color=color)
                                Q[parent_name].add_edge(g_child_node, gg_child_node, weight=w, type=typ, color=color)
                            if 2*(k+1) < len(Ht[child_v].data):
                                gg_child_val, (gg_child_u, gg_child_v) = Ht[child_v].data[2*(k+1)]
                                gg_child_node = (child_v, (gg_child_u, gg_child_v))
                                w = gg_child_val - g_child_val
                                typ = info["Ht-Ht"]["type"]
                                color = info["Ht-Ht"]["color"]
                                P.add_edge(g_child_node, gg_child_node, weight=w, type=typ, color=color)
                                Q[parent_name].add_edge(g_child_node, gg_child_node, weight=w, type=typ, color=color)

            # Ht(step3) - Ht(step3)
            # print(f"parent_name: {parent_name}")
            for j, (child_val, (child_u, child_v)) in enumerate(Ht[parent_name].data):
                # Htを順に調べる．まず，ノードを追加
                child_node = (parent_name, (child_u, child_v))
                # print(f"    child_node:{child_node}")
                P.add_node(child_node)
                Q[parent_name].add_node(child_node)
                # サイドトラックの重みを記録
                sidetrack_weight[(child_u, child_v)] = child_val
                # 次に，辺を追加
                if 2*(j+1)-1 < len(Ht[parent_name].data):
                    g_child_val, (g_child_u, g_child_v) = Ht[parent_name].data[2*(j+1)-1]
                    g_child_node =(parent_name, (g_child_u, g_child_v))
                    w = g_child_val - child_val
                    typ = info["Ht-Ht"]["type"]
                    color = info["Ht-Ht"]["color"]
                    P.add_edge(child_node, g_child_node, weight=w, type=typ, color=color)
                    Q[parent_name].add_edge(child_node, g_child_node, weight=w, type=typ, color=color)
                    # print(f"        g_chold_node:{g_child_node}")
                if 2*(j+1) < len(Ht[parent_name].data):
                    g_child_val, (g_child_u, g_child_v) = Ht[parent_name].data[2*(j+1)]
                    g_child_node =(parent_name, (g_child_u, g_child_v))
                    w = g_child_val - child_val
                    typ = info["Ht-Ht"]["type"]
                    color = info["Ht-Ht"]["color"]
                    P.add_edge(child_node, g_child_node, weight=w, type=typ, color=color)
                    Q[parent_name].add_edge(child_node, g_child_node, weight=w, type=typ, color=color)
                    # print(f"        g_chold_node:{g_child_node}")
        
        return P, Q, sidetrack_weight
    else:
        # Hmidが空(Sを用いてパスグラフ作成)
        pass

