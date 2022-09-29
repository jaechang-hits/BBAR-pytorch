import os
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from torch.utils.data import Dataset, DataLoader
from torch import BoolTensor
from rdkit import Chem, RDLogger
from rdkit.Chem import Mol
from typing import Dict, Union, Any, List, Optional, Callable

from .fcp import FCP
from .utils import brics, feature

RDLogger.DisableLog('rdApp.*')

class MoleculeBuilder() :
    def __init__(self, cfg, filter_fn : Optional[Callable] = None) :
        self.cfg = cfg

        self.model = FCP.load(cfg.model_path, map_location = 'cpu')
        self.model.eval()

        self.library = brics.BRICSLibrary(cfg.library_path, save_mol = True)
        library_freq = torch.from_numpy(self.library.freq) ** cfg.alpha
        self.library_freq = library_freq / library_freq.sum()
        self.lib_size = len(self.library)
        self.n_lib_sample = min(self.lib_size, cfg.n_library_sample)
        if cfg.update_gv_lib or len(getattr(self.model, 'gv_lib', [])) != self.lib_size :
            h, adj = self.get_library_feature()
            with torch.no_grad() :
                gv_lib = self.model.g2v2(h, adj)
                self.model.save_gv_lib(gv_lib)

        self.max_iteration = cfg.max_iteration
        self.max_try = cfg.max_try

        if filter_fn :
            self.filter_fn = filter_fn
        else :
            self.filter_fn = lambda x : True

        self.cond = None
        self.target_properties = self.model.cond_keys

    def setup(self, condition) :
        self.cond = self.model.get_cond(condition).unsqueeze(0)

    @torch.no_grad()
    def generate(
        self,
        scaffold: Optional[Mol],
        verbose: bool = True
        ) :
        assert self.cond is not None, \
            'MoleculeBuilder is not setup. Please call MoleculeBuilder.setup(condition: Dict[property_name, target_value])\n' + \
            f'required property: {list(self.target_properties)}' 

        step = 0

        if scaffold is None :
            idxs = list(range(len(self.library)))
            fragment_idx = np.random.choice(idxs, 1, p=self.library_freq.numpy())[0]
            scaffold = self.library.get_mol(fragment_idx)
            scaffold = brics.preprocess.remove_brics_label(scaffold, returnMol = True)
            if verbose :
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                print(f"Step {step}: random select from library ({fragment_idx})\n"
                      f"\t{scaffold_smiles}")
                step += 1
        else :
            if verbose :
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                print(f"Start\t'{scaffold_smiles}'")

        mol = scaffold
        while step < self.max_iteration :
            h1 = feature.get_atom_features(mol, brics=False).unsqueeze(0)
            adj1 = feature.get_adj(mol).unsqueeze(0)

            # Predict Termination
            _h1, gv1 = self.model.g2v1(h1, adj1, self.cond)
            p_term = self.model.predict_termination(gv1)
            termination = Bernoulli(probs=p_term).sample().bool().item()
            if termination :
                self.print_log(verbose, 'FINISH', step, mol)
                return mol

            # Predict Fragment
            use_lib = self.get_fragment_sample()        # (N_lib)
            if use_lib is not None :
                use_lib = use_lib.unsqueeze(0)
            prob_dist_fragment = self.model.predict_fid(gv1, probs=True, use_lib=use_lib).squeeze(0)
                                                                                                    # (N_lib)

            valid = not (torch.isnan(prob_dist_fragment.sum()))
            if not valid :
                self.print_log(verbose, 'FAIL', step, mol, 'NO_APPROPRIATE_FRAGMENT')
                return None

            success_connect = False
            for _ in range(self.max_try) :
                idx = Categorical(probs = prob_dist_fragment).sample().item()
                if use_lib is None :
                    fragment_idx = idx
                else :
                    fragment_idx = use_lib[0, idx].item()
                fragment = self.library.get_mol(fragment_idx)
                
                # Predict Index
                gv2 = self.model.gv_lib[fragment_idx].unsqueeze(0)
                prob_dist_idx = self.model.predict_idx(h1, adj1, _h1, gv1, gv2, self.cond, probs=True).squeeze(0)
                                                                                                        # (N_atom)

                # Predict Index
                # Masking
                if self.cfg.idx_masking :
                    prob_dist_idx.masked_fill_(self.get_idx_mask(mol, fragment), 0)
                    valid = (torch.sum(prob_dist_idx).item() > 0)
                    if not valid :
                        continue
                # Sampling
                atom_idx = Categorical(probs = prob_dist_idx).sample().item()

                # compose fragments
                try :
                    composed_mol = brics.BRICSCompose.compose(mol, fragment, atom_idx, 0, 
                                                        returnMol=True, force=self.cfg.compose_force)
                    if composed_mol is None or not(self.filter_fn(composed_mol)):
                        continue
                    else :
                        success_connect = True
                        break
                except Exception as e:
                    continue
            
            if success_connect :
                mol = composed_mol
                self.print_log(verbose, 'ADD', step, mol, fragment=fragment, fragment_idx=fragment_idx, atom_idx=atom_idx)
                step += 1
            else :
                self.print_log(verbose, 'FAIL', step, mol, log = 'FAIL_TO_CONNECT_FRAGMENT')
                return None
        # Max Iteration Error
        return None

    __call__ = generate

    @staticmethod
    def print_log(verbose, state, step, mol, **kwargs) :
        if verbose is False :
            return

        mol_smi = Chem.MolToSmiles(mol)
        if state == 'FINISH' :
            print(f"Step {step}: Termination\n"
                  f"\t{mol_smi}"
            )
        elif state == 'ADD' :
            fragment_smi = Chem.MolToSmiles(kwargs['fragment'])
            fragment_idx = kwargs['fragment_idx']
            atom_idx = kwargs['atom_idx']
            print(f"Step {step}: Add '{fragment_smi}' ({fragment_idx}) at index {atom_idx}\n"
                  f"\t{mol_smi}"
            )
        elif state == 'FAIL' :
            print(f"Step {step}: FAIL ({kwargs['log']})")

    def get_library_feature(self) :
        library_feature_path = os.path.splitext(self.cfg.library_path)[0] + '.npz'
        if os.path.exists(library_feature_path) :
            f = np.load(library_feature_path)
            v = torch.from_numpy(f['h']).float()
            adj = torch.from_numpy(f['adj']).bool()
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
                     freq=self.library.freq)
            v = v.float()
            adj = adj.bool()
            
        return v, adj

    def get_fragment_sample(self) :
        if self.n_lib_sample == self.lib_size :
            return None
        else :
            freq = self.library_freq
            idxs = torch.multinomial(freq, self.n_lib_sample, False)
            return idxs

    def get_idx_mask(self, mol: Mol, fragment: Mol) -> BoolTensor:
        idx_mask = torch.ones((mol.GetNumAtoms(),), dtype=torch.bool)
        idxs = brics.BRICSCompose.get_possible_indexs(mol, fragment)
        for idx, _ in idxs :
            idx_mask[idx] = False
        return idx_mask

