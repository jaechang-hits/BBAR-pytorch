import pandas as pd
from rdkit import Chem
import re
import random

df = pd.read_csv('../data/train.csv')
train_smi = set(df.SMILES.tolist())
df = pd.read_csv('../data/val.csv')
val_smi = set(df.SMILES.tolist())
df = pd.read_csv('../data/test.csv')
test_smi = set(df.SMILES.tolist())

smi_ = train_smi.union(val_smi)
print(len(train_smi), len(val_smi), len(test_smi))
print(len(smi_))

smi_list = list(test_smi.difference(smi_))
random.shuffle(smi_list)
print(len(smi_list))

with open('start.smi', 'w') as w :
    for new_smi in smi_list :
        w.write(f'{new_smi}\n')
