import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import FloatTensor, BoolTensor

class GConv(nn.Module) :
    def __init__(self, node_size: int, hidden_size: int, n_layer: int, dropout: float) :
        super(GConv, self).__init__()
        layers = []
        for i in range(n_layer) :
            if i == 0 :
                layers.append(GConvLayer(node_size, hidden_size, dropout))
            else :
                layers.append(GConvLayer(hidden_size, hidden_size, dropout))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: FloatTensor, adj: BoolTensor) :
        _x = x
        _adj = adj.float().requires_grad_(False)
        for layer in self.layers :
            _x = layer(_x, _adj)

        return _x

class GConvLayer(nn.Module) :
    def __init__(self, node_size: int, hidden_size: int, dropout: float):
        super(GConvLayer, self).__init__()
        self.W = nn.Linear(node_size, hidden_size)
        self.gate = nn.Linear(node_size + hidden_size, 1)
        self.leakyrelu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: FloatTensor, adj: BoolTensor):
        h = self.W(x)
        h = self.dropout(h)
        h = F.relu(torch.einsum('xjk,xkl->xjl', (adj, h)))
        coeff = torch.sigmoid(self.gate(torch.cat([x,h], -1))).repeat(1,1,x.size(-1))
        retval = coeff*x+(1-coeff)*h
        return retval
