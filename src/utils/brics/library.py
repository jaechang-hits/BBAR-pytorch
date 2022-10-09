from rdkit import Chem
import pandas as pd
import numpy as np
from typing import Union, List, Optional
import gc
import re

p = re.compile('\[\d+\*\]')

class BRICSLibrary() : 
    def __init__(self, library_path: Optional[str] = None, smiles_list: Optional[List[str]] = None, freq_list: Optional[List[int]] = None, save_mol: bool = True) :
        self.smiles = None
        self._mol = None
        self.freq = None

        if library_path is not None :
            self.load_from_file(library_path)
        else :
            self.load_from_list(smiles_list, freq_list)

        if save_mol :
            self._mol = [Chem.MolFromSmiles(s) for s in self.smiles]

        self._brics_labels = None
    
    def load_from_file(self, library_path) :
        library = pd.read_csv(library_path)
        self.smiles = library.SMILES.to_list()
        try :
            self.freq = library.frequency.to_numpy()
        except :
            self.freq = np.full((len(self.smiles),), 1/len(self.smiles))

    def load_from_list(self, smiles_list, freq_list) :
        self.smiles = smiles_list
        if freq_list is not None :
            self.freq = np.array(freq_list)
        else :
            self.freq = np.full((len(self.smiles),), 1/len(self.smiles))

    def __getitem__(self, idx: Union[int, List[int]]) :
        return self.smiles[idx]

    def get_smiles(self, idx: Union[int, List[int]]) :
        return self[idx]
    
    def get_mol(self, idx: int) :
        if self._mol is not None :
            return self._mol[idx]
        else :
            return Chem.MolFromSmiles(self.smiles[idx])

    @property
    def mol(self) :
        if self._mol is not None :
            return self._mol
        else :
            return [Chem.MolFromSmiles(s) for s in self.smiles]

    def __len__(self) :
        return len(self.smiles)
    
    @property
    def brics_label_list(self)  :
        if self._brics_labels is None :
            self._brics_labels = [self.__get_brics_label(mol) for mol in self.mol]
        return self._brics_labels

    @staticmethod
    def __get_brics_label(mol) :
        return str(mol.GetAtomWithIdx(0).GetIsotope())
