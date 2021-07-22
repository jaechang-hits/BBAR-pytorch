import numpy as np
import torch
from torch import FloatTensor, BoolTensor, LongTensor
from torch.utils.data import Dataset
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Mol
from typing import List, Tuple, Optional
import gc

from utils import feature, brics
RDLogger.DisableLog('rdApp.*')

__all__ = ['FCPDataset']

class FCPDataset(Dataset) :
    def __init__(self, data_file: str, cond_module, library: brics.BRICSLibrary, max_atoms: int) :
        super(FCPDataset, self).__init__()
        self.cond_module = cond_module
        self.max_atoms = max_atoms
        data = pd.read_csv(data_file)
        self.frag1 = data.SMILES.to_numpy()
        self.fid = data.FID.to_numpy()
        self.index = data.Index.to_numpy()
        self.molID = data.MolID.to_numpy()
        self.library = library
        del(data)
        gc.collect()

    def __len__(self) :
        return len(self.frag1)

    def __getitem__(self, idx: int) :
        y_fid = self.fid[idx]
        y_idx = self.index[idx]

        frag1 = self.frag1[idx]
        mol = Chem.MolFromSmiles(frag1)
        v = feature.get_atom_features(mol, self.max_atoms, brics=False)
        adj = feature.get_adj(mol, self.max_atoms)

        molID = self.molID[idx]
        if self.cond_module :
            cond = torch.Tensor(self.cond_module[molID])
        else :
            cond = torch.Tensor([])
        
        return v, adj, cond, y_fid, y_idx 
