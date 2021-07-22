from rdkit import Chem, RDLogger
from rdkit.Chem import Mol
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
import numpy as np
import torch
from torch import FloatTensor, BoolTensor
from typing import Union, Tuple, Optional, List

from .atom_feature import atom_features, NUM_ATOM_FEATURES, NUM_ATOM_FEATURES_BRICS
from .bond_feature import bond_features, NUM_BOND_FEATURES
from . import brics

RDLogger.DisableLog('rdApp.*')

def get_atom_features(mol: Union[Mol,str],
                      max_atoms: int,
                      brics: bool) -> FloatTensor :
    if isinstance(mol, str) :
        mol = Chem.MolFromSmiles(mol)
    if brics :
        af = torch.zeros((max_atoms, NUM_ATOM_FEATURES_BRICS), dtype=torch.float)
    else :
        af = torch.zeros((max_atoms, NUM_ATOM_FEATURES), dtype=torch.float)
    for idx, atom in enumerate(mol.GetAtoms()) :
        af[idx, :] = torch.Tensor(atom_features(atom, brics))
    return af

def get_adj(mol: Union[Mol,str],
            max_atoms: int) -> BoolTensor :
    if isinstance(mol, str) :
        mol = Chem.MolFromSmiles(mol)
    adj = GetAdjacencyMatrix(mol) + np.eye(mol.GetNumAtoms())
    padded_adj = np.zeros((max_atoms, max_atoms), dtype='b')
    n_atom = len(adj)
    padded_adj[:n_atom, :n_atom] = adj
    return torch.from_numpy(padded_adj)

def get_bond_features(mol: Union[Mol,str],
                      max_atoms: int) -> FloatTensor :
    if isinstance(mol, str) :
        mol = Chem.MolFromSmiles(mol)
    bf = torch.zeros((max_atoms, max_atoms, NUM_BOND_FEATURES), dtype=torch.float)
    for bond in mol.GetBonds() :
        idx1 = bond.GetBeginAtomIdx()
        idx2 = bond.GetEndAtomIdx()
        feature = torch.Tensor(bond_features(bond))
        bf[idx1, idx2, :] = feature
        bf[idx2, idx1, :] = feature
    return bf
