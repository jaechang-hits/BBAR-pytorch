import sys
import torch
import torch.nn as nn
from torch import FloatTensor, BoolTensor
from typing import Tuple, Dict

from .layer import GraphEncodingModel, Graph2Vec, TerminationCheckModel, FragmentSelectionModel, IndexSelectionModel
from .utils.feature import NUM_ATOM_FEATURES, NUM_ATOM_FEATURES_BRICS

class BlockConnectionPredictor(nn.Module) :
    def __init__ (self, cfg, cond_scale) :
        super(BlockConnectionPredictor, self).__init__()
        self._cfg = cfg
        if cond_scale is not None :
            self.cond_scale = cond_scale
            self.cond_keys = list(self.cond_scale.keys())
            self.cond_size = len(self.cond_scale)
        else :
            self.cond_scale = {}
            self.cond_keys = []
            self.cond_size = 0

        self.Z_lib = None

        # Graph To Vec
        self.gem_mol = GraphEncodingModel(NUM_ATOM_FEATURES, self.cond_size, **cfg.GraphEncodingModel_Mol)
        self.readout_mol = Graph2Vec(cond_size = self.cond_size, **cfg.Readout_Mol)

        self.gem_frag = GraphEncodingModel(NUM_ATOM_FEATURES_BRICS, None, **cfg.GraphEncodingModel_Frag)
        self.readout_frag = Graph2Vec(cond_size = 0, **cfg.Readout_Frag)

        # Terminate Check
        self.tcm = TerminationCheckModel(**cfg.TerminationCheckModel)

        # Fragment Section
        self.fsm = FragmentSelectionModel(**cfg.FragmentSelectionModel)

        # Index Selection
        Z_mol_dim = cfg.Readout_Mol.output_size
        Z_frag_dim = cfg.Readout_Frag.output_size
        Z_pair_dim = Z_mol_dim + Z_frag_dim
        self.gem_pair = GraphEncodingModel(cond_input_size=Z_pair_dim, **cfg.GraphEncodingModel_Pair)
        self.ism = IndexSelectionModel(NUM_ATOM_FEATURES + cfg.GraphEncodingModel_Pair.hidden_size, **cfg.IndexSelectionModel)

    def initialize_parameters(self) :
        for param in self.parameters() :
            if param.dim() == 1 :
                continue
            else :
                nn.init.xavier_normal_(param)

    def graph_embedding_mol (self, h, adj, cond) :
        _h = self.gem_mol(h, adj, cond)
        Z = self.readout_mol(_h, cond)
        return _h, Z

    def graph_embedding_frag (self, h, adj) :
        _h = self.gem_frag(h, adj)
        Z = self.readout_frag(_h)
        return Z

    def calculate_prob(self, Z_mol, Z_frag) :
        return self.fsm(Z_mol, Z_frag)

    def predict_termination(self, Z_mol) :
        return self.tcm(Z_mol)
        
    def predict_frag_id(self, Z_mol, use_lib = None, probs = False) :
        """
        Z_mol       N, Fout+F'
        y_mask      N, N_lib
        """
        batch_size = Z_mol.size(0)
        if use_lib is None :
            Z_lib_batch = self.Z_lib.unsqueeze(0).repeat(batch_size, 1, 1)
        else :
            Z_lib_batch = self.Z_lib[use_lib]                             # (N, N_lib, F)
        n_lib = Z_lib_batch.size(1)

        Z_mol = Z_mol.unsqueeze(1).repeat(1, n_lib, 1)                # (N, N_lib, F+F')
        y = self.fsm(Z_mol, Z_lib_batch)

        if probs :
            y = y / y.sum(-1, keepdim = True)
        else :
            return y.log()

        return y                                                            # (N, N_lib)

    def predict_idx(self, h, adj, _h, Z_mol, Z_frag, mask = None, probs = False) :
        """
        h       (N, V, Fin)
        adj     (N, V, V)
        _h      (N, V, Fhid)
        Z_mol   (N, Fz_mol)
        Z_frag  (N, Fz_frag)

        mask    (N, V)
        """
        Z_pair = torch.cat([Z_mol, Z_frag], dim=-1)
        _h_pair = self.gem_pair(_h, adj, Z_pair)                    # N, V, Fhid
        h_pair = torch.cat([h, _h_pair], dim=-1)                    # N, V, Fin + Fhid
        Y = self.ism(h_pair)                                         # N, V
        node_mask = torch.logical_not(adj.sum(2).bool())
        Y.masked_fill_(node_mask[:, :Y.size(1)], float('-inf'))
        if mask is not None :
            Y.masked_fill_(mask[:, :h_pair.size(1)], float('-inf'))
        if probs :
            Y = torch.softmax(Y, dim=-1)
        return Y
        
    def get_cond (self, target: Dict[str, float]) :
        def standardize(var, std_var) :
            return (var - std_var.mean) / std_var.std
        assert target.keys() == self.cond_scale.keys(), \
            f"Input Keys is not valid\n" \
            f"\tInput:      {set(target.keys()) if len(target)>0 else '{}'}\n" \
            f"\tRequired:   {set(self.cond_scale.keys())}"
        ret = []
        for desc, std_var in self.cond_scale.items() :
            ret.append (standardize(target[desc], std_var))
        return torch.Tensor(ret)

    def set_Z_lib(self, Z_lib) :
        self.Z_lib = Z_lib

    def save(self, save_file) :
        torch.save({'model_state_dict': self.state_dict(),
                    'config': self._cfg,
                    'cond_scale': self.cond_scale}, save_file)

    @classmethod
    def load(cls, save_file, map_location='cuda') :
        checkpoint = torch.load(save_file, map_location = map_location)
        model = cls(checkpoint['config'],checkpoint['cond_scale'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(map_location)
        return model
