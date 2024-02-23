import torch
from dgl import DGLGraph

edge_types = ["AST","CFG","DDG","CDG"]
max_etype = len(edge_types)

class DataEntry:
    def __init__(self, num_nodes, features, edges):
        self.num_nodes = num_nodes
        self.graph = DGLGraph()
        self.features = torch.FloatTensor(features)
        self.graph.add_nodes(self.num_nodes, data={'features': self.features})
        for s, _type, t in edges:
            etype_number = int(get_edge_type_number(edge_types,_type))
            self.graph.add_edges(int(s),int(t), data={'etype': torch.LongTensor([etype_number])})


def get_edge_type_number(edge_types, _type):
    if _type not in edge_types:
        edge_types.append(_type)
        max_etype += 1
    return edge_types.index(_type)