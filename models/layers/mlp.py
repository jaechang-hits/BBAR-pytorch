import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

"""
Linear Function for Graph
It combines dropout, linear, activation layers.

Args:
    node_input_size: the number features of node in the linear layer inputs (required)
    edge_input_size: the number features of edge in the linear layer inputs (optional)
    cond_size: the number of feature of condition vector (optional)
    node_output_size: the number features of node in the linear layer outputs (required)
    edge_output_size: the number features of edge in the linear layer outputs (optional)
    activation: the activation function (required)
    dropout: the dropout value (required)

Input:
    nodes: the nodes' feature (required)
    edges: the edges' feature (optional)
    condition: the condition vector's feature (optional)

Shape:
    nodes: (N, V, Fv)
    edges: (N, V, V, Fe)
    condition: (N, Fl) or (N, V, Fl)

Note:
    if a condition vector exists, it is combined with the nodes feature vector.
        >> condition = condition.repeat(1, num_nodes, 1) (when the dim of condition is (N, F))
        >> nodes = torch.cat([nodes, condition], -1)

Output:
    nodes: (N, V, Fv')
    edges: (N, V, V, Fe') (optional)
"""

ACT_LIST = {
    'relu': F.relu, 
    'tanh': torch.tanh, 
    'sigmoid': torch.sigmoid, 
    'leaky_relu': F.leaky_relu
}

class GraphLinear(nn.Module) :
    def __init__(self, node_input_size: int, edge_input_size: Optional[int], cond_input_size: Optional[int],
                 node_output_size: int, edge_output_size: Optional[int], activation: str, dropout: float) :
        super(GraphLinear, self).__init__()
        if cond_input_size is None :
            cond_input_size = 0
        self.node_linear = nn.Linear(node_input_size + cond_input_size, node_output_size)
        if edge_input_size is not None and edge_input_size != 0 :
            self.edge_linear = nn.Linear(edge_input_size, edge_output_size)
        self.activation = ACT_LIST.get(activation, None)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, nodes: Tensor, edges: Optional[Tensor] = None, condition: Optional[Tensor] = None) :
        assert (edges is None) ^ (hasattr(self, 'edge_linear'))
        if condition is not None :
            cs = condition.size()
            if len(cs) == 2 :   # condition: [B, F]
                num_nodes = nodes.size(1)
                condition = condition.unsqueeze(1)
                condition = condition.repeat(1, num_nodes, 1)
            elif len(cs) == 3 : # condition: [B, N, F]
                pass
            else :
                print("ERROR: Argument 3 of GraphEmbedding layer should be [batch, num_node, feature] or [batch, feature]")
                exit(-1)
            nodes = torch.cat([nodes, condition], -1)

        _nodes = self.dropout(nodes)
        _nodes = self.node_linear(_nodes)
        if self.activation is not None :
            _nodes = self.activation(_nodes)

        if edges is not None :
            _edges = self.edge_linear(edges)
            if self.activation is not None :
                _edges = self.activation(_edges)
            return _nodes, _edges
        else :
            return _nodes

class Linear(nn.Module) :
    def __init__(self, input_size: int, output_size:int, activation: str, bias: bool, dropout: float) :
        super(Linear, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = ACT_LIST.get(activation, None)
    
    def forward(self, x: Tensor) :
        _x = self.dropout(x)
        y = self.linear(_x)
        if self.activation is not None :
            y = self.activation(y)
        return y
