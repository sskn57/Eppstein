import heapq
from math import log2, floor, log
# import networkx as nx
import os
import graphviz


# Heapクラス #
class Heap_Edge:
    def __init__(self):
        self.data = []
        heapq.heapify(self.data)
    
    # 挿入: :重みvalueのsidetrack(u,v)をheapに追加 #
    def insert(self, u, v, value):
        heapq.heappush(self.data, (value, (u, v)))

    # Heapの描画(graphvizを使うことで木構造で描画可能) #
    def show(self, fpath, format):
        size = len(self.data)
        G = graphviz.Digraph(format=format)
        G.attr('node', shape='circle')
        edges = []
        for i, _ in enumerate(self.data):
            G.node(self.nodename(i))
            if (i-1)//2 >= 0:
                edges.append((self.nodename((i-1)//2), self.nodename(i)))
        G.edges(edges)
        G.render(fpath)
        os.remove(fpath)

    # heapとindexからヒープノードの名前を算出 #
    def nodename(self, index):
        h = self.data[index][1][0]
        t = self.data[index][1][1]
        val = self.data[index][0]
        nodename = f"({h},{t}) {val}"
        return nodename


class Heap_Node:
    def __init__(self):
        self.data = []
        heapq.heapify(self.data)
    
    # 挿入: :重みvalueのsidetrack(u,v)をheapに追加 #
    def insert(self, node, value):
        heapq.heappush(self.data, (value, node))

    # Heapの描画(graphvizを使うことで木構造で描画可能) #
    def show(self, fpath, format):
        size = len(self.data)
        G = graphviz.Digraph(format=format)
        G.attr('node', shape='circle')
        edges = []
        for i, _ in enumerate(self.data):
            G.node(self.nodename(i))
            if (i-1)//2 >= 0:
                edges.append((self.nodename((i-1)//2), self.nodename(i)))
        G.edges(edges)
        G.render(fpath)
        os.remove(fpath)

    # heapとindexからヒープノードの名前を算出 #
    def nodename(self, index):
        node = self.data[index][1]
        val = self.data[index][0]
        nodename = f"q({node})={val}"
        return nodename
