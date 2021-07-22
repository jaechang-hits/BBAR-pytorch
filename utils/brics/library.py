from rdkit import Chem
import pandas as pd
import numpy as np
import gc
import re
p = re.compile('\[\d+\*\]')

class BRICSLibrary() : 
    def __init__(self, library_file: str, save_mol: bool = False) :
        self.smiles = None
        self._mol = None
        self.freq = None

        library = pd.read_csv(library_file, index_col = False, names=['SMILES', 'freq'])
        self.smiles = library.SMILES
        self.freq = library.freq
        
        self.allow = np.zeros([17, len(self.smiles)], dtype='?')
        for fid2, frag2 in enumerate(self.smiles) :
            allow_set = set()
            brics_set = {t[1: -2] for t in p.findall(frag2)}
            for brics_type in brics_set :
                allow_set.update(BRICS_TYPE_INT[brics_type])
                self.allow[list(allow_set), fid2] = True

        if save_mol :
            self._mol = [Chem.MolFromSmiles(s) for s in self.smiles]
    
    def __getitem__(self, idx: int) :
        return self.smiles[idx]

    def get_smiles(self, idx: int) :
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
