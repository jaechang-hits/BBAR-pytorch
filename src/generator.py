import os
import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.bernoulli import Bernoulli
from torch.utils.data import Dataset, DataLoader
from torch import BoolTensor
from rdkit import Chem, RDLogger
from rdkit.Chem import Mol, Descriptors
from typing import Dict, Union, Any, List, Optional, Callable

from .model import BlockConnectionPredictor
from .utils import brics, feature

RDLogger.DisableLog('rdApp.*')

class MoleculeBuilder() :
    def __init__(self, cfg, filter_fn : Optional[Callable] = None) :
        self.cfg = cfg
        self.max_iteration = cfg.max_iteration

        if filter_fn :
            self.filter_fn = filter_fn
        else :
            self.filter_fn = lambda x : True

        library_builtin_model_path = self.cfg.get('library_builtin_model_path', None)
        # Load Model & Library
        if library_builtin_model_path is not None and os.path.exists(library_builtin_model_path) :
            self.model, self.library = self.load_library_builtin_model(library_builtin_model_path)
        else :
            self.model, self.library = self.load_model(cfg.model_path), self.load_library(cfg.library_path)
            self.embed_model_with_library(library_builtin_model_path)

        # Setup Library Information
        library_freq = torch.from_numpy(self.library.freq)
        self.library_freq = library_freq / library_freq.sum()
        self.library_freq_weighted = library_freq ** cfg.alpha
        self.library_allow_brics_list = self.get_library_allow_brics_list()
        self.n_lib_sample = min(len(self.library), cfg.n_library_sample)

        # Setup after self.setup()
        self.target_properties = self.model.cond_keys
        self.cond = None

    def setup(self, condition) :
        self.cond = self.model.get_cond(condition).unsqueeze(0)

    @torch.no_grad()
    def generate(
        self,
        scaffold: Union[Mol, str, None],
        verbose: bool = True
        ) :
        assert self.cond is not None, \
            'MoleculeBuilder is not setup. Please call MoleculeBuilder.setup(condition: Dict[property_name, target_value])\n' + \
            f'required property: {list(self.target_properties)}' 

        step = 0
        if scaffold is None :
            fragment_idx, scaffold = self.get_random_scaffold(max_try = 10)
            if verbose :
                if scaffold is not None :
                    scaffold_smiles = Chem.MolToSmiles(scaffold)
                    print(f"Step {step}: random select from library ({fragment_idx})\n"
                        f"\t{scaffold_smiles}")
                    step += 1
                else :
                    print(f"Fail to select valid scaffold from library")
                    return None
        else :
            if isinstance(scaffold, str) :
                scaffold = Chem.MolFromSmiles(scaffold)
            if verbose :
                scaffold_smiles = Chem.MolToSmiles(scaffold)
                if self.cfg.idx_masking :
                    valid = (len(brics.BRICSCompose.get_possible_labels(scaffold)) > 0)
                else :
                    valid = True
                if valid :
                    print(f"Start\t'{scaffold_smiles}'")
                else :
                    print(f"Invalid Scaffold: '{scaffold_smiles}'")

        mol = scaffold
        while step < self.max_iteration :
            h = feature.get_atom_features(mol, brics=False).unsqueeze(0)
            adj = feature.get_adj(mol).unsqueeze(0)

            # Predict Termination
            _h, Z_mol = self.model.graph_embedding_mol(h, adj, self.cond)
            p_term = self.model.predict_termination(Z_mol)
            termination = Bernoulli(probs=p_term).sample().bool().item()
            if termination :
                self.print_log(verbose, 'FINISH', step, mol)
                return mol

            # Predict Fragment
            use_lib = self.get_fragment_sample(mol)        # (N_lib)
            if use_lib is not None :
                if use_lib.size(0) == 0 :
                    self.print_log(verbose, 'FAIL', step, mol, log = 'NO_APPROPRIATE_FRAGMENT')
                    return None
                else :
                    use_lib = use_lib.unsqueeze(0)

            prob_dist_fragment = self.model.predict_frag_id(Z_mol, probs=True, use_lib=use_lib).squeeze(0)
                                                                                                    # (N_lib)
            valid = not (torch.isnan(prob_dist_fragment.sum()))
            if not valid :
                self.print_log(verbose, 'FAIL', step, mol, log = 'NO_APPROPRIATE_FRAGMENT')
                return None
            idx = Categorical(probs = prob_dist_fragment).sample().item()
            if use_lib is None :
                fragment_idx = idx
            else :
                fragment_idx = use_lib[0, idx].item()
            fragment = self.library.get_mol(fragment_idx)
            
            # Predict Index
            Z_frag = self.model.Z_lib[fragment_idx].unsqueeze(0)
            prob_dist_idx = self.model.predict_idx(h, adj, _h, Z_mol, Z_frag, probs=True).squeeze(0)
            # Masking
            if self.cfg.idx_masking :
                prob_dist_idx.masked_fill_(self.get_idx_mask(mol, fragment), 0)
                valid = (torch.sum(prob_dist_idx).item() > 0)
                if not valid :
                    continue
            atom_idx = Categorical(probs = prob_dist_idx).sample().item()

            # compose fragments
            try :
                composed_mol = brics.BRICSCompose.compose(mol, fragment, atom_idx, 0, 
                                                    returnMol=True, force=self.cfg.compose_force)
                assert composed_mol is not None
            except Exception as e:
                self.print_log(verbose, 'FAIL', step, mol, log = 'FAIL_TO_CONNECT_FRAGMENT\n'+str(e))
                return None
        
            mol = composed_mol
            self.print_log(verbose, 'ADD', step, mol, fragment=fragment, fragment_idx=fragment_idx, atom_idx=atom_idx)
            step += 1
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

    def get_library_allow_brics_list(self) :
        library_mask = torch.zeros((len(self.library), 17), dtype=torch.bool)
        for i, brics_label in enumerate(self.library.brics_label_list) : 
            allow_brics_label_list = brics.constant.BRICS_ENV_INT[brics_label]
            for allow_brics_label in allow_brics_label_list :
                library_mask[i, allow_brics_label] = True
        self.library_mask = library_mask.T  # (self.library, 17)
    
    def get_random_scaffold(self, max_try = 10) :
        fragment_idxs = torch.multinomial(self.library_freq, max_try).tolist()
        for fragment_idx in fragment_idxs :
            scaffold = self.library.get_mol(fragment_idx)
            scaffold = brics.preprocess.remove_brics_label(scaffold, returnMol = True)
            if self.cfg.idx_masking :
                valid = (len(brics.BRICSCompose.get_possible_brics_labels(scaffold)) > 0)
            else :
                valid = True
            if valid :
                return fragment_idx, scaffold
        return None, None

    def get_fragment_sample(self, mol) :
        brics_labels = brics.BRICSCompose.get_possible_brics_labels(mol)
        allow_fragment = torch.zeros((len(self.library),), dtype=torch.bool)
        for brics_label in brics_labels :
            allow_fragment += self.library_mask[int(brics_label)]
        
        use_lib = torch.arange(len(self.library))[allow_fragment]
        if self.n_lib_sample > len(use_lib) :
            return use_lib
        else :
            freq = self.library_freq_weighted[allow_fragment]
            idxs = torch.multinomial(freq, self.n_lib_sample, False)
            return use_lib[idxs]

    def get_idx_mask(self, mol: Mol, fragment: Mol) -> BoolTensor:
        idx_mask = torch.ones((mol.GetNumAtoms(),), dtype=torch.bool)
        idxs = brics.BRICSCompose.get_possible_indexs(mol, fragment)
        for idx, bidx in idxs :
            idx_mask[idx] = False
        return idx_mask
   
    def load_model(self, model_path) :
        model = BlockConnectionPredictor.load(model_path, map_location = 'cpu')
        model.eval()
        return model

    def load_library(self, library_path) :
        return brics.BRICSLibrary(library_path, save_mol = True)

    def load_library_builtin_model(self, library_builtin_model_path) :
        print(f"Load {library_builtin_model_path}")
        checkpoint = torch.load(library_builtin_model_path, map_location = 'cpu')
        model = BlockConnectionPredictor(checkpoint['config'], checkpoint['cond_scale'])
        model.load_state_dict(checkpoint['model_state_dict'])
        model.set_Z_lib(checkpoint['Z_lib'])
        model.eval()
        
        library = brics.BRICSLibrary(smiles_list = checkpoint['library_smiles'], freq_list = checkpoint['library_freq'], save_mol = True)
        return model, library

    def embed_model_with_library(self, library_builtin_model_path) :
        print("Setup Library Fragments' Graph Vectors")
        with torch.no_grad() :
            h, adj = self.load_library_feature()
            Z_lib = self.model.graph_embedding_frag(h, adj)
            self.model.Z_lib = Z_lib
        print("Finish")
        if library_builtin_model_path is not None :
            print(f"Create Local File ({library_builtin_model_path})")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'config': self.model._cfg,
                'cond_scale': self.model.cond_scale,
                'library_smiles': self.library.smiles,
                'library_freq': self.library.freq,
                'Z_lib': Z_lib
            }, self.library_builtin_model_path)
        else :
            print("You can save graph vectors by setting generator_config.library_builtin_model_path")

        
    def load_library_feature(self) :
        # Load node feature / adjacency matrix
        library_feature_path = os.path.splitext(self.cfg.library_path)[0] + '.npz'
        if os.path.exists(library_feature_path) :
            f = np.load(library_feature_path)
            h = torch.from_numpy(f['h']).float()
            adj = torch.from_numpy(f['adj']).bool()
            f.close()
        else:
            max_atoms = max([m.GetNumAtoms() for m in self.library.mol])
            h, adj = [], []
            for m in self.library.mol :
                h.append(feature.get_atom_features(m, max_atoms, True))
                adj.append(feature.get_adj(m, max_atoms))

            h = torch.stack(h)
            adj = torch.stack(adj)
            np.savez(library_feature_path, h=h.numpy(), adj=adj.numpy().astype('?'), \
                     freq=self.library.freq)
            h = h.float()
            adj = adj.bool()
        return h, adj
