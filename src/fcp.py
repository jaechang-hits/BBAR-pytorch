import sys
import torch
import torch.nn as nn
from torch import FloatTensor, BoolTensor
from typing import Tuple, Dict

from .models import GraphEncodingModel, Graph2Vec, TerminationCheckModel, FragmentSelectionModel, IndexSelectionModel
from .utils.feature import NUM_ATOM_FEATURES, NUM_ATOM_FEATURES_BRICS

class FCP(nn.Module) :
    def __init__ (self, cond_scale, cfg) :
        super(FCP, self).__init__()
        self._cfg = cfg
        if cond_scale is not None :
            self.cond_scale = cond_scale
            self.cond_size = len(self.cond_scale)
        else :
            self.cond_scale = {}
            self.cond_size = 0

        self.gv_lib = None
        gv_size1 = cfg.Readout1.output_size
        gv_size2 = cfg.Readout2.output_size

        # Graph To Vec
        self.gem1_1 = GraphEncodingModel(NUM_ATOM_FEATURES, self.cond_size, **cfg.GraphEncodingModel1)
        self.readout1 = Graph2Vec(cond_size = self.cond_size, **cfg.Readout1)

        self.gem1_2 = GraphEncodingModel(NUM_ATOM_FEATURES_BRICS, None, **cfg.GraphEncodingModel2)
        self.readout2 = Graph2Vec(cond_size = 0, **cfg.Readout2)

        # Terminate Check
        self.tcm = TerminationCheckModel(**cfg.TerminationCheckModel)

        # Fragment Section
        self.fsm = FragmentSelectionModel(**cfg.FragmentSelectionModel)

        # Index Selection
        cond_size = self.cond_size + gv_size1 + gv_size2
        self.gem2_1 = GraphEncodingModel(cond_input_size=cond_size, **cfg.GraphEncodingModel3)
        self.ism = IndexSelectionModel(NUM_ATOM_FEATURES + cfg.GraphEncodingModel3.hidden_size, **cfg.IndexSelectionModel)

    def initialize_parameters(self) :
        for param in self.parameters() :
            if param.dim() == 1 :
                continue
            else :
                nn.init.xavier_normal_(param)

    def g2v1 (self, h1, adj1, cond) :
        _h1 = self.gem1_1(h1, adj1, cond)
        gv1 = self.readout1(_h1, cond)
        return _h1, gv1

    def g2v2 (self, h2, adj2) :
        _h2 = self.gem1_2(h2, adj2)
        gv2 = self.readout2(_h2)
        return gv2

    def calculate_prob(self, gv1, gv2) :
        return self.fsm(gv1, gv2)

    def save_gv_lib (self, gv_lib) :
        if gv_lib is not None :
            self.gv_lib = nn.Parameter(gv_lib, requires_grad = False)    # (N_lib, F)
        else :
            self.gv_lib = None

    def predict_termination(self, gv1) :
        return self.tcm(gv1)
        
    def predict_fid(self, gv1, use_lib = None, probs = False) :
        """
        gv1     N, Fout+F'
        y_mask  N, N_lib
        """
        batch_size = gv1.size(0)
        if use_lib is None :
            gv_lib_batch = self.gv_lib.unsqueeze(0).repeat(batch_size, 1, 1)
        else :
            gv_lib_batch = self.gv_lib[use_lib]                             # (N, N_lib, F)
        n_lib = gv_lib_batch.size(1)

        gv1 = gv1.unsqueeze(1).repeat(1, n_lib, 1)                # (N, N_lib, F+F')
        y = self.fsm(gv1, gv_lib_batch)

        if probs :
            y = y / y.sum(-1, keepdim = True)
        else :
            return y.log()

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
        if cond is not None :
            _cond = torch.cat([cond, gv1, gv2], dim=-1)                 # N, Fcond + Fgv1 + Fgv2
        else :
            _cond = torch.cat([gv1, gv2], dim=-1)
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

    def save(self, save_file) :
        torch.save({'model_state_dict': self.state_dict(),
                    'config': self._cfg,
                    'cond_scale': self.cond_scale,
                    'gv_lib_size': self.gv_lib.size()}, save_file)

    @classmethod
    def load(cls, save_file, map_location='cuda') :
        sys.path.insert(0, './src')
        checkpoint = torch.load(save_file, map_location = map_location)
        model = cls(checkpoint['cond_scale'], checkpoint['config'])
        model.gv_lib = nn.Parameter(torch.empty(checkpoint['gv_lib_size']))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(map_location)
        return model
