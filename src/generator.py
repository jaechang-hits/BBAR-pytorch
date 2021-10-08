import os
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem, RDLogger
from rdkit.Chem import Mol
from typing import Dict, Union, Any, List, Optional, Callable

from utils import brics, feature

RDLogger.DisableLog('rdApp.*')

class MoleculeBuilder() :
    def __init__(self, cfg, device, filter_fn : Optional[Callable] = None) :
        self.cfg = cfg
        self.device = device

        self.model = torch.load(cfg.model_path, map_location = self.device)
        self.model.eval()

        self.library = brics.BRICSLibrary(cfg.library_path, save_mol = True)
        library_freq = torch.from_numpy(self.library.freq.to_numpy()).to(self.device) ** cfg.alpha
        self.library_freq = library_freq / library_freq.sum()
        self.lib_size = len(self.library)
        self.n_lib_sample = min(self.lib_size, cfg.n_library_sample)
        if cfg.update_gv_lib or len(getattr(self.model, 'gv_lib', [])) != self.lib_size :
            h, adj = self.get_library_feature()
            with torch.no_grad() :
                gv_lib = self.model.g2v2(h, adj)
                self.model.save_gv_lib(gv_lib)

        self.batch_size = cfg.batch_size
        self.num_workers = cfg.num_workers

        if filter_fn :
            self.filter_fn = filter_fn
        else :
            self.filter_fn = lambda x : True

        self.cond = self.model.get_cond (cfg.target).unsqueeze(0).repeat(self.batch_size, 1).to(self.device)

    @torch.no_grad()
    def generate(
        self,
        start_fragment: Optional[str],
        n_sample:int = 100,
        ) :

        if start_fragment is not None :
            smiles1_list = np.array([start_fragment for _ in range(n_sample)])
        if start_fragment is None :
            idxs = list(range(len(self.library)))
            draw = np.random.choice(idxs, n_sample, p=self.library_freq.to('cpu').numpy())
            smiles1_list = self.library.get_smiles(draw).tolist()
            smiles1_list = [brics.preprocess.remove_brics_label(smiles) for smiles in smiles1_list]
            smiles1_list = np.array(smiles1_list)

        result_list = []
        total_step = 0
        step = 0
        
        while len(smiles1_list) > 0 :
            step += 1 
            n_sample = len(smiles1_list)
            dataset = MPDataset(smiles1_list, self.library)
            dataloader = DataLoader(dataset, self.batch_size, shuffle=False, num_workers=self.num_workers)

            next_smiles1_list = []
            for idx1, h1, adj1 in dataloader :
                # Calculate Graph Vector
                h1 = h1.to(self.device)
                adj1 = adj1.to(self.device)
                smiles1_batch = smiles1_list[idx1] if idx1.size(0) > 1 else np.array([smiles1_list[idx1]])

                n = h1.size(0)
                _h1, gv1 = self.model.g2v1(h1, adj1, self.cond[:n, :])                  # (N, F + F')

                # Predict Termination
                p_term = self.model.predict_termination(gv1)
                # sampling
                m = Bernoulli(probs=p_term)
                term = m.sample().bool()
                result_list += smiles1_batch[term.cpu().numpy()].tolist()
                total_step += step * term.int().sum()

                # Remove Terminated Ones
                not_term = torch.logical_not(term)
                h1, adj1, gv1, _h1 = h1[not_term], adj1[not_term], gv1[not_term], _h1[not_term]
                idx1 = idx1[not_term]
                smiles1_batch = smiles1_batch[not_term.cpu().numpy()]
                if h1.size(0) == 0 :
                    break

                # Predict Fid2
                use_lib = self.get_sample(h1.size(0))
                prob_dist_fid2 = self.model.predict_fid(gv1, probs=True, use_lib = use_lib).squeeze(-1)     # (N, N_lib)
                # Check whether the prob dist is valid or not
                invalid = torch.isnan(prob_dist_fid2.sum(-1))
                valid = torch.logical_not(invalid)

                h1, adj1, gv1, _h1 = h1[valid], adj1[valid], gv1[valid], _h1[valid]
                prob_dist_fid2 = prob_dist_fid2[valid]
                idx1 = idx1[valid]
                smiles1_batch = smiles1_batch[valid.cpu().numpy()]
                if h1.size(0) == 0 :
                    break

                n = h1.size(0)
                num_atoms1 = h1.size(1)
                # Sampling
                m = Categorical(probs = prob_dist_fid2)
                if use_lib is not None:
                    fid2 = torch.gather(use_lib, 1, m.sample().unsqueeze(-1)).squeeze(-1)
                else :
                    fid2 = m.sample()
                gv2 = self.model.gv_lib[fid2]
                
                # Predict Index
                prob_dist_idx = self.model.predict_idx(h1, adj1, _h1, gv1, gv2, self.cond[:n], probs=True)
                # Masking
                if self.cfg.idx_masking :
                    prob_dist_idx.masked_fill_(self.get_idx_mask(smiles1_batch, fid2, num_atoms1), 0)
                    valid = (torch.sum(prob_dist_idx, dim=-1) > 0)
                    valid_ = valid.tolist()
                    smiles1_batch, fid2 = smiles1_batch[valid_], fid2[valid_]
                    prob_dist_idx = prob_dist_idx[valid_]
                    if fid2.size(0) == 0 : break
                # Sampling
                m = Categorical(probs = prob_dist_idx)
                idx_batch = m.sample()

                # compose fragments
                frag1_batch = [Chem.MolFromSmiles(s) for s in smiles1_batch]
                frag2_batch = [self.library.get_mol(idx.item()) for idx in fid2]
                for frag1, frag2, idx in zip(frag1_batch, frag2_batch, idx_batch) :
                    try :
                        compose_smiles = brics.BRICSCompose.compose(frag1, frag2, int(idx), 0, force=self.cfg.compose_force)
                    except:
                        continue
                    if compose_smiles is None or len(compose_smiles) == 0 :
                        continue
                    if not self.filter_fn(compose_smiles) :
                        continue
                    if Chem.MolFromSmiles(compose_smiles) :
                        next_smiles1_list.append(compose_smiles)

            smiles1_list = np.array(next_smiles1_list)
        return result_list, total_step
    
    @staticmethod
    def sampling_connection(logits = None, probs = None) :
        assert (logits is None) ^ (probs is None)
        if logits is not None :
            batch_size, num_atoms1, num_atoms2 = logits.size()
            m = Categorical(logits = logits.reshape(batch_size, -1))
        else :
            batch_size, num_atoms1, num_atoms2 = probs.size()
            m = Categorical(probs = probs.reshape(batch_size, -1))
        
        y = m.sample()
        idx1 = y // num_atoms2
        idx2 = y % num_atoms2

        return idx1, idx2

    def get_idx_mask(self, frag1_list: List[Union[str, Mol]], fid2_list: List[int], max_atoms:int):
        #possible_brics_type = self.library.get_possible_brics_type(fid2)
        idx_mask = torch.ones((len(frag1_list), max_atoms), dtype=torch.bool, device=self.device)
        for i, (frag1, fid2) in enumerate(zip(frag1_list, fid2_list)) :
            if isinstance(frag1, str) :
                frag1 = Chem.MolFromSmiles(frag1)
            frag2 = self.library.get_mol(fid2)
            idxs = brics.BRICSCompose.get_possible_indexs(frag1, frag2)
            for idx, _ in idxs :
                idx_mask[i, idx] = False
        return idx_mask

    def get_library_feature(self) :
        library_feature_path = os.path.splitext(self.cfg.library_path)[0] + '.npz'
        if os.path.exists(library_feature_path) :
            f = np.load(library_feature_path)
            v = torch.from_numpy(f['h']).float().to(self.device)
            adj = torch.from_numpy(f['adj']).bool().to(self.device)
            f.close()
        else:
            max_atoms = max([m.GetNumAtoms() for m in self.library.mol])
            v, adj = [], []
            for m in self.library.mol :
                v.append(feature.get_atom_features(m, max_atoms, True))
                adj.append(feature.get_adj(m, max_atoms))

            v = torch.stack(v)
            adj = torch.stack(adj)
            np.savez(library_feature_path, h=v.numpy(), adj=adj.numpy().astype('?'), \
                     freq=self.library.freq.to_numpy())
            v = v.float().to(self.device)
            adj = adj.bool().to(self.device)
            
        return v, adj

    def get_sample(self, n_sample) :
        if self.n_lib_sample == self.lib_size :
            return None
        else :
            freq = self.library_freq.expand(n_sample, self.lib_size)
            idxs = torch.multinomial(freq, self.n_lib_sample, False)
            return idxs

class MPDataset(Dataset) :
    def __init__(self, dataset, library) :
        super(MPDataset, self).__init__()
        self.frag1 = dataset
        self.library = library
        self.n_atoms = max([Chem.MolFromSmiles(s).GetNumAtoms() for s in self.frag1])
        
    def __len__(self) :
        return len(self.frag1)

    def __getitem__(self, idx: int) :
        frag1_s = self.frag1[idx]
        frag1_m = Chem.MolFromSmiles(frag1_s)
        v = feature.get_atom_features(frag1_m, self.n_atoms, brics=False)
        adj = feature.get_adj(frag1_m, self.n_atoms)
        return idx, v, adj
