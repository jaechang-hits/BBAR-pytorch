from rdkit import Chem
import pandas as pd
import numpy as np
from typing import Union, List
import gc
import re

from .constant import BRICS_TYPE_INT
p = re.compile('\[\d+\*\]')

class BRICSLibrary() : 
    def __init__(self, library_file: str, save_mol: bool = False) :
        self.smiles = None
        self._mol = None
        self.freq = None

        library = pd.read_csv(library_file)
        self.smiles = library.SMILES
        self.freq = library.frequency

        if save_mol :
            self._mol = [Chem.MolFromSmiles(s) for s in self.smiles]
    
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
