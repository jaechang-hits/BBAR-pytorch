import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
from torch import LongTensor, BoolTensor, FloatTensor

from typing import Tuple, Union, List
import os
import logging
import time

from .model import BlockConnectionPredictor
from .cond_module import Cond_Module
from .dataset import MolBlockPairDataset
from .utils.feature import get_library_feature

bceloss = nn.BCELoss()
celoss = nn.CrossEntropyLoss()

class Trainer() :
    def __init__(self, trainer_config, model_config, data_config, properties: List[str], save_dir: str) :
        self.setup_trainer(trainer_config)
        self.setup_properties(data_config.property_path, properties)
        self.setup_model(model_config)
        self.setup_library(data_config.library_path)
        self.setup_dataset(data_config)
        self.save_dir = save_dir
        if not os.path.exists(save_dir) :
            os.mkdir(save_dir)

    def setup_trainer(self, trainer_cfg):
        self.device = 'cuda:0' if trainer_cfg.gpus > 0 else 'cpu'
        self.num_workers = trainer_cfg.num_workers
        self.lr = trainer_cfg.lr
        self.n_sample = trainer_cfg.num_negative_samples
        self.alpha = trainer_cfg.alpha
        self.train_batch_size = trainer_cfg.train_batch_size
        self.val_batch_size = trainer_cfg.val_batch_size

        self.max_step = trainer_cfg.max_step
        self.val_interval = trainer_cfg.val_interval
        self.save_interval = trainer_cfg.save_interval
        self.log_interval = trainer_cfg.log_interval

    def setup_properties(self, property_path, properties) :
        self.properties = properties
        if len(properties) > 0 :
            self.cond_module = Cond_Module(property_path, properties)
            self.cond_scale = self.cond_module.scale
        else :
            self.cond_module = None
            self.cond_scale = None

    def setup_model(self, model_cfg) :
        model = BlockConnectionPredictor(model_cfg, self.cond_scale)
        model.initialize_parameters()
        self.model = model.to(self.device)
        logging.info(f"number of parameters : {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}")

    def setup_library(self, library_path) :
        h, adj, freq = get_library_feature(library_path = library_path, device = self.device)
        self.library_h, self.library_adj, self.library_freq = h, adj, freq ** self.alpha

    def setup_dataset(self, data_cfg) :
        self.train_ds = MolBlockPairDataset(data_cfg.train_data_path, self.cond_module, data_cfg.train_max_atoms) 
        self.val_ds = MolBlockPairDataset(data_cfg.val_data_path, self.cond_module, data_cfg.val_max_atoms) 

        weight = torch.from_numpy(np.load(data_cfg.train_weight_path)).double()
        sampler = WeightedRandomSampler(weight, self.max_step*self.train_batch_size)
        self.train_dl = DataLoader(self.train_ds, self.train_batch_size, num_workers=self.num_workers, sampler=sampler)
        self.val_dl = DataLoader(self.val_ds, self.val_batch_size, num_workers=self.num_workers)

        logging.info(f'num of train data: {len(self.train_ds)}')
        logging.info(f'num of val data: {len(self.val_ds)}\n')

    def fit(self) :
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.global_step = 0
        self.min_loss = float('inf')
        self.model.train()
        logging.info('Train Start')
        optimizer.zero_grad()
        metrics_storage = self.get_metrics_storage()
        for h_in, adj_in, cond, y_frag, y_idx in self.train_dl :
            metrics = self.run_train_step(h_in, adj_in, cond, y_frag, y_idx, optimizer)
            self.log_metrics(metrics, metrics_storage)

            self.global_step += 1
            if self.global_step % self.log_interval == 0 :
                metrics = self.aggregate_metrics(metrics_storage)
                self.print_metrics(metrics, 'TRAIN')
                self.clear_metrics_storage(metrics_storage)
            if self.global_step % self.save_interval == 0 :
                save_path = os.path.join(self.save_dir, f'ckpt_{self.global_step}.tar')
                self.model.save(save_path)
            if self.global_step % self.val_interval == 0 :
                self.validation()

        save_path = os.path.join(self.save_dir, 'last.tar')
        self.model.save(save_path)

    @torch.no_grad()
    def validation(self) :
        self.model.eval()
        self.model_set_Z_lib()
        metrics_storage = self.get_metrics_storage()
        for h_in, adj_in, cond, y_frag, y_idx in self.val_dl :
            metrics = self.run_val_step(h_in, adj_in, cond, y_frag, y_idx)
            self.log_metrics(metrics, metrics_storage)
        metrics = self.aggregate_metrics(metrics_storage)
        loss = metrics['loss']
        self.print_metrics(metrics, 'VAL  ')

        self.model_remove_Z_lib()
        self.model.train()

        if loss < self.min_loss :
            self.min_loss = loss
            save_path = os.path.join(self.save_dir, 'best.tar')
            self.model.save(save_path)

    def run_train_step(self, h, adj, cond, y_frag, y_idx, optimizer) :
        loss, metrics = self._step(h, adj, cond, y_frag, y_idx, train=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        return metrics

    def run_val_step(self, h, adj, cond, y_frag, y_idx) :
        loss, metrics = self._step(h, adj, cond, y_frag, y_idx, train=False)
        return metrics

    def _step(self, h: FloatTensor, adj: BoolTensor, cond: FloatTensor, y_frag: LongTensor, y_idx: LongTensor, train: bool = True) :
        """
        h       N, Fin
        adj     N, N
        cond    N, F'
        y_frag   N
        y_idx   N
        """
        h = h.to(self.device).float()
        adj = adj.to(self.device).bool()
        cond = cond.to(self.device).float()
        y_frag = y_frag.to(self.device).long()
        y_idx = y_idx.to(self.device).long()

        if cond.size(-1) == 0 :
            cond = None
        _h, Z_mol = self.model.graph_embedding_mol(h, adj, cond)

        y_term = (y_frag == -1)
        y_not_term = torch.logical_not(y_term)
        p_term = self.model.predict_termination(Z_mol)
        term_loss = bceloss(p_term, y_term.float())

        if cond is not None :
            h, adj, cond = h[y_not_term], adj[y_not_term], cond[y_not_term]
        else :
            h, adj = h[y_not_term], adj[y_not_term]
        y_frag, y_idx = y_frag[y_not_term], y_idx[y_not_term]
        _h, Z_mol = _h[y_not_term], Z_mol[y_not_term]
   
        if h.size(0) == 0 :
            loss = term_loss
            metrics = {'loss' : loss.item(), 'term_loss' : term_loss.item()}
            return term_loss, metrics

        if train :
            h_frag, adj_frag = self.get_sample(y_frag)
            Z_frag_pos = self.model.graph_embedding_frag(h_frag, adj_frag)
        else :
            Z_frag_pos = self.model.Z_lib[y_frag]
        prob_p = self.model.calculate_prob(Z_mol, Z_frag_pos)       # (N)
        frag_ploss = (prob_p + 1e-12).log().mean().neg()

        frag_nloss = 0
        for _ in range(self.n_sample) :
            y_frag_neg = self.get_neg_sample(y_frag)
            if train :
                h_frag, adj_frag = self.get_sample(y_frag_neg)
                Z_frag_neg = self.model.graph_embedding_frag(h_frag, adj_frag)
            else :
                Z_frag_neg = self.model.Z_lib[y_frag_neg]
            prob_n = self.model.calculate_prob(Z_mol, Z_frag_neg)   # (N)
            frag_nloss += (1. - prob_n + 1e-12).log().mean().neg()
        frag_nloss/=self.n_sample

        logit_idx = self.model.predict_idx(h, adj, _h, Z_mol, Z_frag_pos, probs=False)
        idx_loss = celoss(logit_idx, y_idx)

        loss = term_loss + frag_ploss + frag_nloss + idx_loss
        metrics = {
            'loss' : loss.item(), 'term_loss' : term_loss.item(),
            'frag_ploss' : frag_ploss.item(), 'frag_nloss' : frag_nloss.item(), 'idx_loss' : idx_loss.item()
        }
        return loss, metrics

    """
    get_neg_sample: select negative sample according to the frequent distribution of library.
    Correct fragments(y) and fragments couldn't be connected to target(y_mask) are masked. """
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
    def model_set_Z_lib(self) :
        Z_lib = self.model.graph_embedding_frag(self.library_h, self.library_adj)
        self.model.set_Z_lib(Z_lib)

    def model_remove_Z_lib(self) :
        self.model.set_Z_lib(None)

    def get_metrics_storage(self) :
        return {'tloss': 0.0, 'fploss': 0.0, 'fnloss': 0.0, 'iloss': 0.0, 'cnt': 0, 'cnt_add' : 0}

    def clear_metrics_storage(self, metrics_storage) :
        for key in metrics_storage.keys () :
            metrics_storage[key] = 0

    def log_metrics(self, metrics, metrics_storage) :
        metrics_storage['cnt'] += 1
        metrics_storage['tloss'] += metrics['term_loss']
        if 'frag_ploss' in metrics :
            metrics_storage['cnt_add'] += 1
            metrics_storage['fploss'] += metrics['frag_ploss']
            metrics_storage['fnloss'] += metrics['frag_nloss']
            metrics_storage['iloss'] += metrics['idx_loss']

    def aggregate_metrics(self, metrics_storage) :
        tloss = metrics_storage['tloss'] / metrics_storage['cnt']
        fploss = metrics_storage['fploss'] / metrics_storage['cnt_add']
        fnloss = metrics_storage['fnloss'] / metrics_storage['cnt_add']
        iloss = metrics_storage['iloss'] / metrics_storage['cnt_add']
        loss = tloss + iloss + fploss + fnloss
        return {'tloss': tloss, 'fploss': fploss, 'fnloss': fnloss, 'iloss': iloss, 'loss': loss}

    def print_metrics(self, metrics, mode = 'TRAIN') :
        loss, tloss, fploss, fnloss, iloss = metrics['loss'], metrics['tloss'], \
                                             metrics['fploss'], metrics['fnloss'], metrics['iloss']
        logging.info(
            f'STEP {self.global_step}\t'
            f'{mode}\t'
            f'loss: {loss:.3f}\t'
            f'tloss: {tloss:.3f}\t'
            f'ploss: {fploss:.3f}\t'
            f'nloss: {fnloss:.3f}\t'
            f'iloss: {iloss:.3f}\t'
        )
