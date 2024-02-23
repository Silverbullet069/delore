import torch
from dgl.nn import GatedGraphConv
from torch import nn
import torch.nn.functional as f 
import torch
from dgl import DGLGraph


class Graph:
    def __init__(self):
        self.graph = DGLGraph()
        self.number_of_nodes = 0
        self.graphid_to_nodeids = {}
        self.num_of_subgraphs = 0

    def add_subgraph(self, _g):
        assert isinstance(_g, DGLGraph)
        num_new_nodes = _g.number_of_nodes()
        self.graphid_to_nodeids[self.num_of_subgraphs] = torch.LongTensor(
            list(range(self.number_of_nodes, self.number_of_nodes + num_new_nodes)))
        self.graph.add_nodes(num_new_nodes, data=_g.ndata)
        sources, dests = _g.all_edges()
        sources += self.number_of_nodes
        dests += self.number_of_nodes
        self.graph.add_edges(sources, dests, data=_g.edata)
        self.number_of_nodes += num_new_nodes
        self.num_of_subgraphs += 1

    def cuda(self, device=None):
        for k in self.graphid_to_nodeids.keys():
            self.graphid_to_nodeids[k] = self.graphid_to_nodeids[k].cuda(device=device)

    def de_batchify_graphs(self, features=None):
        if features is None:
            features = self.graph.ndata['features']
        assert isinstance(features, torch.Tensor)
        vectors = [features.index_select(dim=0, index=self.graphid_to_nodeids[gid]) for gid in
                   self.graphid_to_nodeids.keys()]
        lengths = [f.size(0) for f in vectors]
        max_len = max(lengths)
        for i, v in enumerate(vectors):
            vectors[i] = torch.cat(
                (v, torch.zeros(size=(max_len - v.size(0), *(v.shape[1:])), requires_grad=v.requires_grad,
                                device=v.device)), dim=0)
        output_vectors = torch.stack(vectors)
        lengths = torch.LongTensor(lengths).to(device=output_vectors.device)
        return output_vectors, lengths

    def get_network_inputs(self, cuda=False):
        raise NotImplementedError('Must be implemented by subclasses.')


class GGNNGraph(Graph):
    def get_network_inputs(self, cuda=False, device=None):
        features = self.graph.ndata['features']
        edge_types = self.graph.edata['etype']
        if cuda:
            self.cuda(device=device)
            return self.graph, features.cuda(device=device), edge_types.cuda(device=device)
        else:
            return self.graph, features, edge_types
        pass


class GGNNSum(nn.Module):
    def __init__(self, input_dim, output_dim, max_edge_types, num_steps=8):
        super(GGNNSum, self).__init__()
        self.inp_dim = input_dim
        self.out_dim = output_dim
        self.max_edge_types = max_edge_types
        self.num_timesteps = num_steps
        self.ggnn = GatedGraphConv(in_feats=input_dim, out_feats=output_dim, n_steps=num_steps,
                                   n_etypes=max_edge_types)
        self.classifier = nn.Linear(in_features=output_dim, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, _g, cuda=False):
        _Graph = GGNNGraph()
        _Graph.add_subgraph(_g)
        graph, features, edge_types = _Graph.get_network_inputs(cuda=cuda)
        outputs = self.ggnn(graph, features, edge_types)
        h_i, _ = _Graph.de_batchify_graphs(outputs)
        ggnn_sum = self.classifier(h_i.sum(dim=1))
        result = self.sigmoid(ggnn_sum).squeeze(dim=-1)
        return result