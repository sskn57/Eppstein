import heapq
from math import log2, floor, log
# import networkx as nx
import os
import graphviz

# A = [3, 5, 1, 2, 0, 10, 54, 2, 22, 4]
data = [
    (5, "A"),
    (4, "B"),
    (2, "C"),
    (7, "D"),
    (10, "E"),
    (3, "F"),
    (6, "G"),
    (9, "H"),
    (12, "I")
]

# Heapクラス #
class Heap:
    def __init__(self):
        self.data = []
        heapq.heapify(self.data)
    
    # 挿入: :重みvalueのsidetrack(u,v)をheapに追加 #
    def insert(self, u, v, value):
        heapq.heappush(self.data, (value, (u, v)))

    # Heapの描画(graphvizを使うことで木構造で描画可能) #
    def show(self, fpath):
        size = len(self.data)
        G = graphviz.Digraph(format="png")
        G.attr('node', shape='circle')
        edges = []
        for i, _ in enumerate(self.data):
            G.node(self.nodename(i))
            if (i-1)//2 >= 0:
                edges.append((self.nodename((i-1)//2), self.nodename(i)))
        G.edges(edges)
        G.render(fpath)
        os.remove(fpath)

    # self.dataとindexからヒープノードの名前を算出 #
    def nodename(self, index):
        h = self.data[index][1][0]
        t = self.data[index][1][1]
        val = self.data[index][0]
        nodename = f"({h},{t}) {val}"
        return nodename


# if __name__ == "__main__":
#     print(data)
#     heapq.heapify(data)
#     print(data)
#     show_tree(data, 1, 0)