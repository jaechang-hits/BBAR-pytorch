import torch
import torch.nn as nn
from torch import FloatTensor
from typing import Optional
from . import layers

class Graph2Vec(nn.Module) :
    """
    Return a graph-representation vector of shape.
    See Eq. (4) of Yujia Li et al. 2018.

    Input
    nodes : batch, n_node, input_size

    Output(Graph Vector)
    retval : batch, output_size
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        cond_size: int = 0,
        dropout: float = 0.1
    ) :

        super(Graph2Vec, self).__init__()
        self.cond = (cond_size > 0)
        self.linear1 = layers.Linear(
            input_size = input_size,
            output_size = hidden_size,
            activation = None,
            bias = False,
            dropout = dropout)

        self.linear2 = layers.Linear(
            input_size = input_size,
            output_size = hidden_size,
            activation = None,
            bias = True,
            dropout = dropout)

        self.linear3 = layers.Linear(
            input_size = hidden_size + cond_size,
            output_size = output_size,
            activation = None,
            bias = True,
            dropout = dropout)

    def forward(self, v: FloatTensor, cond: Optional[FloatTensor] = None) :
        """
        v:      N, V, F
        cond:   N, Fcond
        """
        assert (cond is None) ^ (self.cond)
        v1 = self.linear1(v)
        v2 = torch.sigmoid(self.linear2(v))
        Z = (v1 * v2).sum(1)
        if cond is not None :
            Z = torch.cat ([Z, cond], dim=-1)
        return self.linear3(Z)
