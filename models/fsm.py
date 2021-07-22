import torch
import torch.nn as nn
from torch import FloatTensor
from . import layers

class FragmentSelectionModel(nn.Module) :
    def __init__(
        self,
        input_size1: int, 
        input_size2: int,
        hidden_size: int, 
        dropout: float = 0.1
    ) :

        super(FragmentSelectionModel, self).__init__()
        self.linear1 = layers.Linear(
            input_size = input_size1 + input_size2,
            output_size = hidden_size,
            activation = 'relu',
            bias = True,
            dropout = dropout)

        self.linear2 = layers.Linear(
            input_size = hidden_size,
            output_size = 1,
            activation = 'sigmoid',
            bias = True,
            dropout = dropout)

    def forward(self, gv1: FloatTensor, gv2: FloatTensor) :
        gv_concat = torch.cat([gv1, gv2], dim=-1)               # (N, F1 + F2)
        Y = self.linear1(gv_concat)                             # (N, F1 + F2) -> (N, Fhid)
        Y = self.linear2(Y).squeeze(-1)                        # (N, Fhid) -> (N)
        return Y
