# pos読み込み用
def read_pos(fpath, delim=" "):
  dic = {}
  with open(fpath) as f:
    for line in f.readlines():
      data = line.split(delim)
      dic[data[0]] = (int(data[1]), int(data[2]))
  return dic

# src, dst読み込み用
def read_src_dst(fpath):
    with open(fpath) as f:
        dat = f.readlines()
        if len(dat) != 2:
            return None
        else:
            return (dat[0].rstrip('\n'), dat[1].rstrip('\n'))

# threshold読み込み用
def read_threshold(fpath):
    with open(fpath) as f:
        dat = f.readlines()
        if len(dat) != 1:
            return None
        else:
            return float(dat[0].rstrip('\n'))

# グラフG-Tにおいてhead/tailがノードvである辺, sidetrack(v)を取得する
def sidetrack(v, G, T):
    # ret = []
    # for h in G[v]:
    #     if h not in T[v]:
    #         ret.append((v, h))
    # return ret
    ret = []
    # print(G.edges())
    for (h, t) in G.edges():
        if (h, t) not in T.edges():
        # if (h, t) not in T.edges() and (t, h) not in T.edges():
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