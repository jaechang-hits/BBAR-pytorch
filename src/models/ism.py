import torch
import torch.nn as nn
from torch import Tensor
from . import layers

"""
Calculate attention between nodes of two graphs.
Input:
    X1: node feature of first graph
    X2: node feature of second graph
    sbidx1: single bond idx of first graph
    sbidx2: single bond idx of second graph
    dbidx1: double bond idx of first graph
    dbidx2: double bond idx of second graph

Shape:
    X1: (N, V1, Fv)
    X2: (N, V2, Fv)
    sbidx1: (N, V1)
    sbidx2: (N, V2)
    dbidx1: (N, V1)
    dbidx2: (N, V2)

Output:
    Y: probability distribution. Y[#, v1, v2] means It means the probability of a connection
    between v1 of graph 1 and v2 of graph 2.

Shape:
    Y: (N, V1, V2)

Note:
    The Brics algorithm breaks a single bond or a double bond to obtain a substructure.
    In order to recover the original structure from the obtained substructure, it is
    necessary to distinguish between a single bond and a double bond.

    Our goal is to predict through which nodes the two graphs will be connected.
    Therefore, we have to calculate the probability distribution according to v1 and v2,
    v1 in Graph 1 and v2 in Graph 2.

Example :
    Molecule 1 :
        *COCC*
        012345
    >> sbidx1 = [[1, 0, 0, 0, 0, 1]]
    >> dbidx1 = [[0, 0, 0, 0, 0, 0]]
    
    Molecule 2: 
        *C(=O)C(=*)CC*
        01  2 3  4 567
    >> sbidx2 = [[1, 0, 0, 0, 0, 0, 0, 1]]
    >> dbidx2 = [[0, 0, 0, 0, 1, 0, 0, 0]]

    avaliable connection: (0, 0), (0, 7), (5, 0), (5, 7)
    probability distribution :
        *COCC-C(=O)C(=*)CC*  probability: Y[0, 0, 0]
        *COCC-CC(=*)C(=O)C*  probability: Y[0, 0, 7]
        *CCOC-C(=O)C(=*)CC*  probability: Y[0, 5, 0]
        *CCOC-CC(=*)C(=O)C*  probability: Y[0, 5, 7]
     
    >> Y[0, 0, 0] + Y[0, 0, 7] + Y[0, 5, 0] + Y[0, 5, 7] = 1.0
    >> torch.sum(Y[0]) = 1.0
"""


class IndexSelectionModel(nn.Module) :
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        dropout: float = 0.1
    ) :

        super(IndexSelectionModel, self).__init__()    
        self.linear1 = layers.Linear(
            input_size = input_size,
            output_size = hidden_size,
            activation = 'leaky_relu',
            bias = True,
            dropout = dropout
        )
        self.linear2 = layers.Linear(
            input_size = hidden_size,
            output_size = 1,
            activation = None,
            bias = True,
            dropout = dropout
        )

    def forward(self, v1: Tensor) :
        _v1 = self.linear1(v1)              # N, V, F
        Y = self.linear2(_v1).squeeze(-1)   # N, V
        return Y
