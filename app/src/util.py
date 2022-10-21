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

# グラフG-Tにおいてheadがノードvである辺, sidetrack(v)を取得する
def sidetrack(v, G, T):
    ret = []
    for h in G[v]:
        if h not in T[v]:
            ret.append((v, h))
    return ret