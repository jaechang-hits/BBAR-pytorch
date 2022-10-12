import random
import pandas as pd
import numpy as np
from collections import Counter
import os

random.seed(0)

src_dir = '../ZINC/'
src_train_data_path = os.path.join(src_dir, 'train.csv')
src_val_data_path = os.path.join(src_dir, 'val.csv')
src_library_path = os.path.join(src_dir, 'library.csv')
src_property_path = os.path.join(src_dir, 'property.db')

dst_train_data_path = './train.csv'
dst_weight_file_path = './train_weight.npy'
dst_val_data_path = './val.csv'
dst_library_path = './library.csv'
dst_library_unseen_path = './library_unseen.csv'
dst_library_unseen_hydrophilic_path = './library_unseen_hydrophilic.csv'
dst_library_unseen_hydrophobic_path = './library_unseen_hydrophobic.csv'
dst_property_path = './property.db'

os.symlink(src_property_path, dst_property_path)

# Create Library
with open(src_library_path) as f :
    lines = [l.strip().split(',') for l in f.readlines()[1:]]
lines = [(int(a),b,int(c)) for a, b, c in lines]
random.shuffle(lines)
cutoff = int(len(lines) * 2/3)
seen_lines = lines[:cutoff]
unseen_lines = lines[cutoff:]
seen_lines.sort()
unseen_lines.sort()

with open(dst_library_path, 'w') as w :
    w.write('FID,SMILES,frequency\n')
    for new_fragid,(old_fragid, smiles, freq) in enumerate(seen_lines) :
        w.write(f'{new_fragid},{smiles},{freq}\n')

with open(dst_library_unseen_path, 'w') as w :
    w.write('FID,SMILES\n')
    for new_fragid,(old_fragid, smiles, freq) in enumerate(unseen_lines) :
        w.write(f'{new_fragid},{smiles}\n')

from rdkit import Chem
from rdkit.Chem.Descriptors import TPSA
unseen_data_w_tpsa = [(TPSA(Chem.MolFromSmiles(smiles)), smiles) for _, smiles, _ in unseen_lines]
unseen_data_w_tpsa.sort()

with open(dst_library_unseen_hydrophobic_path, 'w') as w :
    w.write('FID,SMILES,TPSA\n')
    for new_fragid in range(2000) :
        tpsa, smiles = unseen_data_w_tpsa[new_fragid]
        w.write(f'{new_fragid},{smiles},{tpsa}\n')

with open(dst_library_unseen_hydrophilic_path, 'w') as w :
    w.write('FID,SMILES,TPSA\n')
    for new_fragid in range(2000) :
        tpsa, smiles = unseen_data_w_tpsa[-(new_fragid+1)]
        w.write(f'{new_fragid},{smiles},{tpsa}\n')

## Update Train/Val DataPoint
dic = {a:i for i, (a,_,_) in enumerate(seen_lines)}

with open(dst_train_data_path, 'w') as w :
    w.write('SMILES,FID,Idx,MolID\n')
    with open(src_train_data_path) as f :
        lines = f.readlines()
        for l in lines[1:] :
            smi, org_fid, idx, molid = l.split(',')
            fid = dic.get(int(org_fid), None)
            if fid is not None :
                w.write(f'{smi},{fid},{idx},{molid}')

smiles = pd.read_csv(dst_train_data_path, usecols = ['SMILES'])['SMILES'].tolist()

counter = Counter(smiles)
freq = np.array([counter[smi] for smi in smiles])
weight = 1. / freq
np.save(dst_weight_file_path, weight)

with open(dst_val_data_path, 'w') as w :
    w.write('SMILES,FID,Idx,MolID\n')
    with open(src_val_data_path) as f :
        lines = f.readlines()
        for l in lines[1:] :
            smi, org_fid, idx, molid = l.split(',')
            fid = dic.get(int(org_fid), None)
            if fid is not None :
                w.write(f'{smi},{fid},{idx},{molid}')
