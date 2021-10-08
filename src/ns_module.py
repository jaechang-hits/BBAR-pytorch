import torch
import torch.nn as nn
import numpy as np
from torch import LongTensor, BoolTensor, FloatTensor
from typing import Tuple, Union

from .utils.feature import get_library_feature

bceloss = nn.BCELoss()
celoss = nn.CrossEntropyLoss()

class NS_Trainer(nn.Module) :
    def __init__(self, model, library_file: str, n_sample: int, alpha: float, device: Union[str, torch.device]) :
        super(NS_Trainer, self).__init__()
        self.model = model
        self.n_sample = n_sample
        h, adj, freq = get_library_feature(library_path = library_file, device = device)
        self.library_h, self.library_adj, self.library_freq = h, adj, freq
        self.lib_size = self.library_h.size(0)
        self.lib_node_size = self.library_h.size(1)
        self.library_h.requires_grad_(False)
        self.library_adj.requires_grad_(False)
        self.model.to(device)

    def forward(self, h: FloatTensor, adj: BoolTensor, cond: FloatTensor, y_fid: LongTensor, y_idx: LongTensor, train=True) :
        """
        h       N, Fin
        adj     N, N
        cond    N, F'
        y_fid   N
        y_idx   N
        """
        if cond.size(-1) == 0 :
            cond = None
        _h, gv1 = self.model.g2v1(h, adj, cond)

        y_term = (y_fid == -1)
        y_not_term = torch.logical_not(y_term)

        p_term = self.model.predict_termination(gv1)
        term_loss = bceloss(p_term, y_term.float())

        if cond is not None :
            h, adj, cond = h[y_not_term], adj[y_not_term], cond[y_not_term]
        else :
            h, adj = h[y_not_term], adj[y_not_term]
        y_fid, y_idx = y_fid[y_not_term], y_idx[y_not_term]
        _h, gv1 = _h[y_not_term], gv1[y_not_term]
   
        if h.size(0) == 0 :
            return term_loss, 0.0, 0.0, 0.0

        y_fid[y_fid<0] = 0
        if train :
            hp, adjp = self.get_sample(y_fid)
            gvp = self.model.g2v2(hp, adjp)
        else :
            gvp = self.model.gv_lib[y_fid]
        prob_p = self.model.calculate_prob(gv1, gvp)                # (N)
        fid_ploss = (prob_p + 1e-12).log().mean().neg()

        fid_nloss = 0
        for _ in range(self.n_sample) :
            y_neg = self.get_neg_sample(y_fid)
            if train :
                hn, adjn = self.get_sample(y_neg)                # (N)
                gvn = self.model.g2v2(hn, adjn)
            else :
                gvn = self.model.gv_lib[y_neg]
            prob_n = self.model.calculate_prob(gv1, gvn)      # (N)
            fid_nloss += (1. - prob_n + 1e-12).log().mean().neg()
        fid_nloss/=self.n_sample

        logit_idx = self.model.predict_idx(h, adj, _h, gv1, gvp, cond, probs=False)
        idx_loss = celoss(logit_idx, y_idx)

        return term_loss, fid_ploss, fid_nloss, idx_loss
        
    """
    get_neg_sample: select negative sample according to the frequent distribution of library.
    Correct fragments(y) and fragments couldn't be connected to target(y_mask) are masked.
    """
    @torch.no_grad()
    def get_neg_sample(self, y: LongTensor) -> LongTensor:
        batch_size = y.size(0)
        freq = self.library_freq.repeat(batch_size, 1)
        freq.scatter_(1, y.unsqueeze(1), 0)
        neg_idxs = torch.multinomial(freq, 1, True).view(-1)
        return neg_idxs

    @torch.no_grad()
    def get_sample(self, sample: LongTensor) -> Tuple[FloatTensor, BoolTensor] :
        """
        sample: LongTensor, (N)
        """
        h = self.library_h[sample]
        adj = self.library_adj[sample]
        return h, adj

    @torch.no_grad()
    def model_save_gv_lib(self) :
        gv_out_lib = self.model.g2v2(self.library_h, self.library_adj)
        self.model.save_gv_lib(gv_out_lib)

    def model_remove_gv_lib(self) :
        self.model.save_gv_lib(None)
