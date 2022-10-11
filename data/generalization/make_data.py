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
dst_library_unuse_path = './library_unuse.csv'
dst_property_path = './property.db'

os.symlink(src_property_path, dst_property_path)

with open(src_library_path) as f :
    lines = [l.strip().split(',') for l in f.readlines()[1:]]
lines = [(int(a),b,int(c)) for a, b, c in lines]
random.shuffle(lines)
cutoff = int(len(lines) * 2/3)
use_lines = lines[:cutoff]
unuse_lines = lines[cutoff:]

use_lines.sort()
unuse_lines.sort()

with open(dst_library_path, 'w') as w :
    w.write('FID,SMILES,frequency\n')
    for i,(a,b,c) in enumerate(use_lines) :
        w.write(f'{i},{b},{c}\n')

with open(dst_library_unuse_path, 'w') as w :
    w.write('FID,SMILES\n')
    for i,(a,b,c) in enumerate(unuse_lines) :
        w.write(f'{i},{b}\n')

dic = {a:i for i, (a,_,_) in enumerate(use_lines)}

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
