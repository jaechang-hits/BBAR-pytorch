from rdkit import Chem
from rdkit.Chem import Mol
from typing import Union, List, Tuple, Optional, Dict
import re

def remove_brics_label(mol: Union[str, Mol], idx: Optional[int] = None, returnMols: bool = False) \
                                -> Tuple[Union[str, Mol], Optional[int]] :
    if isinstance(mol, Mol) :
        smi = Chem.MolToSmiles(mol)
        mol = Chem.MolFromSmiles(smi)
    if isinstance(mol, str) :
        mol = Chem.MolFromSmiles(mol)
        smi = Chem.MolToSmiles(mol)

    assert (idx is None) or (idx < mol.GetNumAtoms())

    new_smi = re.sub('\[\d+\*\]', '[H]', smi)
    new_mol = Chem.MolFromSmiles(new_smi)
    new_smi = Chem.MolToSmiles(new_mol)
    new_mol = Chem.MolFromSmiles(new_smi)
    if idx :
        map_idx = mol.GetSubstructMatch(new_mol)
        for idx1, idx2 in enumerate(map_idx) :
            if (idx2 == idx) :
                idx = idx1
                break
        if returnMols:
            return new_mol, idx
        else :
            return new_smi, idx
    if returnMols:
        return new_mol
    else :
        return new_smi
