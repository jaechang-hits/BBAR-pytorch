from utils import feature
import pandas as pd
from rdkit import Chem
import torch
import numpy as np

lib = pd.read_csv('data/library.csv')
a = lib.SMILES

ma = 34
v = []
adj = []
for smi in a :
    m = Chem.MolFromSmiles(smi)
    v.append(feature.get_atom_features(m, ma, True))
    adj.append(feature.get_adj(m, ma))

v = torch.stack(v).numpy()
adj = torch.stack(adj).numpy().astype('?')
freq = lib.frequency.to_numpy()

np.savez('data/library.npz', h=v, adj=adj, freq=freq)
