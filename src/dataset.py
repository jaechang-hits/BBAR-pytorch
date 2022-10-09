import numpy as np
import torch
from torch import FloatTensor, BoolTensor, LongTensor
from torch.utils.data import Dataset
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Mol
from typing import List, Tuple, Optional
import gc

from .utils import feature, brics
RDLogger.DisableLog('rdApp.*')

class MolBlockPairDataset(Dataset) :
    def __init__(self, data_file: str, cond_module, max_atoms: int) :
        super(MolBlockPairDataset, self).__init__()
        self.cond_module = cond_module
        self.max_atoms = max_atoms
        data = pd.read_csv(data_file)
        self.mol = data.SMILES.to_numpy()
        self.frag_id = data.FID.to_numpy()
        self.index = data.Idx.to_numpy()
        self.molID = data.MolID.to_numpy()
        del(data)
        gc.collect()

    def __len__(self) :
        return len(self.mol)

    def __getitem__(self, idx: int) :
        y_frag = self.frag_id[idx]
        y_idx = self.index[idx]

        mol_smiles = self.mol[idx]
        mol = Chem.MolFromSmiles(mol_smiles)
        v = feature.get_atom_features(mol, self.max_atoms, brics=False)
        adj = feature.get_adj(mol, self.max_atoms)

        molID = self.molID[idx]
        if self.cond_module :
            cond = torch.Tensor(self.cond_module[molID])
        else :
            cond = torch.Tensor([])
        
        return v, adj, cond, y_frag, y_idx 
