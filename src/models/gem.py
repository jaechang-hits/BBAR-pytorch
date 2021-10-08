import torch.nn as nn
from torch import FloatTensor, BoolTensor
from typing import Optional
from . import layers

class GraphEncodingModel(nn.Module):
    def __init__(
        self,
        node_input_size: int, 
        cond_input_size: Optional[int] = None, 
        node_hidden_size: int = 64, 
        hidden_size: int = 64, 
        n_layer: int = 4, 
        dropout: float = 0.1
    ) :

        super(GraphEncodingModel, self).__init__()
        self.embedding = layers.GraphLinear(
            node_input_size = node_input_size,
            edge_input_size = 0,
            cond_input_size = cond_input_size,
            node_output_size = node_hidden_size,
            edge_output_size = 0,
            activation = 'relu',
            dropout = 0.0
        )
        self.encoder = layers.GConv(
            node_size = node_hidden_size,
            hidden_size = hidden_size,
            n_layer = n_layer,
            dropout = dropout
        )

    def forward(self, v: FloatTensor, adj: BoolTensor, cond: Optional[FloatTensor] = None) :
        _v= self.embedding(v, None, cond)
        V = self.encoder(_v, adj)
        node_mask = adj.sum(2)
        V = V.masked_fill(node_mask.unsqueeze(-1)==False, 0)
        return V
