import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from torch.utils.data import Dataset, DataLoader
from rdkit import Chem, RDLogger
from typing import Dict, Union, Any, Optional, Callable
import gc

from utils import brics, common, feature

RDLogger.DisableLog('rdApp.*')

class MoleculeBuilder() :
    def __init__(
        self,
        model: str,     # Model path
        library: str,   # Library path
        library_npz: str,   #Library feature path
        target: Dict[str, float],
        batch_size: int = 32,
        num_workers: int = 0,
        filter_fn : Optional[Callable] = None,
        device : Union[str, torch.device] = 'cuda:0'
        ) :

        self.model = torch.load(model, map_location = device)
        self.model.eval()
        self.library = brics.BRICSLibrary(library, save_mol = True)
        self.batch_size = batch_size
        self.num_workers = num_workers

        if filter_fn :
            self.filter_fn = filter_fn
        else :
            self.filter_fn = lambda x : True

        self.device = device
        self.lib_size = len(self.library)
        self.cond = self.model.get_cond (target).unsqueeze(0).repeat(batch_size, 1).to(device)

        library_npz = np.load(library_npz)
        self.library_h = torch.from_numpy(library_npz['h']).to(device)
        self.library_adj = torch.from_numpy(library_npz['adj']).to(device)
        #self.library_freq = torch.from_numpy(library_npz['freq']).unsqueeze(0).to(device)
        self.lib_size = self.library_h.size(0)
        self.lib_node_size = self.library_h.size(1)
        self.library_h.requires_grad_(False)
        self.library_adj.requires_grad_(False)
        library_npz.close()
        gc.collect()

    @torch.no_grad()
    def generate(
        self,
        start_fragment: str,
        n_sample:int = 100,
        ) :

        smiles1_list = np.array([start_fragment for _ in range(n_sample)])
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
                prob_dist_fid2 = self.model.predict_fid(gv1, probs=True).squeeze(-1)     # (N, N_lib)
                # check whether the prob dist is valid or not
                valid = (torch.sum(prob_dist_fid2, dim=-1) > 0)
                h1, adj1, gv1, _h1 = h1[valid], adj1[valid], gv1[valid], _h1[valid]
                prob_dist_fid2 = prob_dist_fid2[valid]
                idx1 = idx1[valid]
                smiles1_batch = smiles1_batch[valid.cpu().numpy()]
                frag1_batch = [Chem.MolFromSmiles(s) for s in smiles1_batch]
                n = h1.size(0)
                num_atoms1 = h1.size(1)
                # sampling
                m = Categorical(probs = prob_dist_fid2)
                fid2 = m.sample()
                frag2_batch = [self.library.get_mol(idx.item()) for idx in fid2]
                gv2 = self.model.gv_lib[fid2]
                
                # Predict Index
                prob_dist_idx = self.model.predict_idx(h1, adj1, _h1, gv1, gv2, self.cond[:n], probs=True)
                m = Categorical(probs = prob_dist_idx)
                idx_batch = m.sample()

                # compose fragments
                for frag1, frag2, idx in zip(frag1_batch, frag2_batch, idx_batch) :
                    compose_smiles = brics.BRICSCompose.compose(frag1, frag2, int(idx), 0)
                    if compose_smiles is None :
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
