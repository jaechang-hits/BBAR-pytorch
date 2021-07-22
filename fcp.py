import torch
import torch.nn as nn
from torch import FloatTensor, BoolTensor
from typing import Tuple, Dict
from models import GraphEncodingModel, Graph2Vec, TerminationCheckModel, FragmentSelectionModel, IndexSelectionModel
from utils.feature import NUM_ATOM_FEATURES, NUM_ATOM_FEATURES_BRICS

use_lib = 2000 #69364

class FCP(nn.Module) :
    def __init__ (self,
                  cond_scale,
                  hidden_size11=64,     # embedding size of node feature
                  hidden_size12=64,     # output size of GConv
                  hidden_size13=128,    # hidden size in readout layer
                  gv_size1=128,
                  gv_size2=128,
                  hidden_size21=128,    # hidden size of Termination Check Model
                  hidden_size31=128,    # hidden size of Fragment Selection Model
                  hidden_size41=64,     # embedding size of node feature (Connection Part)
                  hidden_size42=64,     # output size of GConv (Connection Part)
                  hidden_size51=64,     # hidden size of Connection Selection Model
                  n_layer = 4,
                  dropout = 0.0
                  ) :
        super(FCP, self).__init__()

        self.cond_scale = cond_scale
        self.cond_size = len(self.cond_scale)

        self.gv_lib = None
        self.gv_lib_batch = None

        # Graph To Vec
        self.gem1_1 = GraphEncodingModel(NUM_ATOM_FEATURES, self.cond_size, hidden_size11, hidden_size12, n_layer, dropout)
        self.readout1 = Graph2Vec(hidden_size12, hidden_size13, gv_size1, self.cond_size, dropout)

        self.gem1_2 = GraphEncodingModel(NUM_ATOM_FEATURES_BRICS, None, hidden_size11, hidden_size12, n_layer, dropout)
        self.readout2 = Graph2Vec(hidden_size12, hidden_size13, gv_size2, 0, dropout)

        # Terminate Check
        self.tcm = TerminationCheckModel(gv_size1, hidden_size21, dropout)

        # Fragment Section
        self.fsm = FragmentSelectionModel(gv_size1, gv_size2, hidden_size31, dropout)

        # Index Selection
        cond_size = self.cond_size + gv_size1 + gv_size2
        self.gem2_1 = GraphEncodingModel(hidden_size12, cond_size, hidden_size41, hidden_size42, n_layer, dropout)
        self.ism = IndexSelectionModel(NUM_ATOM_FEATURES + hidden_size42, hidden_size51, dropout)

    def g2v1 (self, h1, adj1, cond) :
        _h1 = self.gem1_1(h1, adj1, cond)
        gv1 = self.readout1(_h1, cond)
        return _h1, gv1

    def g2v2 (self, h2, adj2) :
        _h2 = self.gem1_2(h2, adj2)
        gv2 = self.readout2(_h2)
        return _h2, gv2

    def calculate_prob(self, gv1, gv2) :
        return self.fsm(gv1, gv2)

    def save_gv_lib (self, gv_lib) :
        if gv_lib is not None :
            self.gv_lib = nn.Parameter(gv_lib, requires_grad = False)    # (N_lib, F)
        else :
            self.gv_lib = None

    def predict_termination(self, gv1) :
        return self.tcm(gv1)
        

    def predict_fid(self, gv1, y_mask = None, force = False, probs = False) :
        """
        gv1     N, Fout+F'
        y_mask  N, N_lib
        """
        batch_size = gv1.size(0)
        if force or self.gv_lib_batch is None or self.gv_lib_batch.size(0) < batch_size :
            self.gv_lib_batch = self.gv_lib[:use_lib].unsqueeze(0).repeat(batch_size, 1, 1)
                                                                            # (N, N_lib, F)
        gv1 = gv1.unsqueeze(1).repeat(1, use_lib, 1)                        # (N, N_lib, F+F')
        y = self.fsm(gv1, self.gv_lib_batch[:batch_size]).log()

        if y_mask is not None :
            y.masked_fill_(y_mask[:, :use_lib], '-inf')                     # (N, N_lib)

        if probs :
            y = torch.softmax(y, dim=-1)

        return y                                                            # (N, N_lib)

    def predict_idx(self, h1, adj1, _h1, gv1, gv2, cond, mask = None, probs = False) :
        """
        h1      (N, V1, Fin)
        adj1    (N, V1, V1)
        _h1     (N, V1, Fhid)
        gv1     (N, Fgv1)

        gv2     (N, Fgv2)

        cond    (N, Fcond)
        mask    (N, V1)
        """
        _cond = torch.cat([cond, gv1, gv2], dim=-1)                 # N, Fcond + Fgv1 + Fgv2
        _h1 = self.gem2_1(_h1, adj1, _cond)                         # N, V, Fhid
        h11 = torch.cat([h1, _h1], dim=-1)                          # N, V, Fin + Fhid
        Y = self.ism(h11)                                           # N, V
        node_mask = torch.logical_not(adj1.sum(2).bool())
        Y.masked_fill_(node_mask[:, :Y.size(1)], float('-inf'))
        if mask is not None :
            Y.masked_fill_(mask[:, :h11.size(1)], float('-inf'))
        if probs :
            Y = torch.softmax(Y, dim=-1)
        return Y
        
    @torch.no_grad()
    def sampling (self, h1, adj1, cond, y_mask) :
        _h1, gv1 = self.g2v1(h1, adj1, cond)
        fid_probs = self.calculate_prob_dist (gv1, y_mask, probs=True)
        valid = (torch.sum(fid_probs, dim=-1) > 0)
        fid_probs = fid_probs[valid]
        h1, adj1, _h1, gv1, cond = h1[valid], adj1[valid], _h1[valid], gv1[valid], cond[valid]
        m = Categorical(probs = fid_probs)
        fid = m.sample()
        gv2 = self.gv_lib[fid]
        idx_probs = self.predict_index(h1, adj1, _h1, gv2, None, probs=True)
        m = Categorical(probs = idx_probs)
        idx = m.sample()
        return valid, fid, idx

    def get_cond (self, target: Dict[str, float]) :
        assert target.keys() == self.cond_scale.keys()
        ret = []
        for desc, std_var in self.cond_scale.items() :
            ret.append (self.standardize(target[desc], std_var))
        return torch.Tensor(ret)

    @staticmethod
    def standardize(var, std_var) :
        return (var - std_var.mean) / std_var.std
