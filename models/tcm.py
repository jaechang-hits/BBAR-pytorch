import torch
import torch.nn as nn
from torch import Tensor
from . import layers

class TerminationCheckModel(nn.Module) :
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        dropout: float = 0.1
    ) :

        super(TerminationCheckModel, self).__init__()    
        self.linear1 = layers.Linear(
            input_size = input_size,
            output_size = hidden_size,
            activation = 'tanh',
            bias = True,
            dropout = dropout
        )
        self.linear2 = layers.Linear(
            input_size = hidden_size,
            output_size = 1,
            activation = 'sigmoid',
            bias = True,
            dropout = dropout
        )

    def forward(self, v1: Tensor) :
        _v1 = self.linear1(v1)                  # N, F
        Y = self.linear2(_v1).squeeze(-1)       # N
        return Y
