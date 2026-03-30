from graphviz import Digraph
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'

class Show:
    """ Class for visualising, uses graphviz for nodes and edges and adding relations """
    def __init__(self, root):
        self.root = root

    def trace(self):
        nodes, edges = set(), set()
        def build(v):
            if v not in nodes:
                nodes.add(v)
                for child in v._children:
                    edges.add((child, v))
                    build(child)
        build(self.root)
        return nodes, edges

    def show(self, format='svg', rankdir = 'LR'):
        # format = svg | png
        # rankdir = rank direction = TB (top to bottom graph) | LR (left to right)
        assert rankdir in ['LR', 'TB']
        nodes, edges = self.trace()
        dot = Digraph(format=format, graph_attr={'rankdir': rankdir})

        for n in nodes:
            dot.node(name=str(id(n)), label = "{data %.4f} | {grad %.4f}"%(n.data, n.grad), shape='record')
            if n._op:
                dot.node(name=str(id(n))+n._op, label=n._op)
                dot.edge(str(id(n))+n._op, str(id(n)))

        for e1, e2 in edges:
            dot.edge(str(id(e1)), str(id(e2))+e2._op)

        return dot