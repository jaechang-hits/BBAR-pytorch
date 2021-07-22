import os
import pytorch_lightning as pl
import numpy as np
import logging
import time
logging.disable(logging.CRITICAL)
os.environ['OMP_NUM_THREADS'] = '1'
pl.seed_everything(0)

from generator import MoleculeBuilder
from utils import common
from rdkit import Chem

import sys

"""
To Avoid Too Large Molecule For Speed
"""
def filter_fn (smiles: str) -> bool:
    return True
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() > 50 :
        return False
    else :
        return True

# set parameters
model = sys.argv[1]
library = 'data/library.csv'
library_npz = 'data/library.npz'
start_frag_file = 'start_list/start_list.smi'
target = {'MolLogP': float(sys.argv[2])}
n_sample = int(sys.argv[3])
output_file = sys.argv[4]

os.system('mkdir -p ' + '/'.join(output_file.split('/')[:-1]))

st = time.time()
mb = MoleculeBuilder(model, library, library_npz, target, batch_size = 512, num_workers=0, filter_fn = filter_fn, device='cuda:0')
start_frag_list = common.load_txt(start_frag_file)
sample_dict = {}
total_step = 0
for i, start_frag in enumerate(start_frag_list) :
    sample_list, n_step = mb.generate(start_frag, n_sample = n_sample)
    sample_dict[start_frag] = sample_list
    total_step += n_step

validity = [len((sample_dict[start_frag])) / n_sample for start_frag in start_frag_list]
uniqueness = [len(set(sample_dict[start_frag])) / n_sample for start_frag in start_frag_list]
avg_validity = np.mean(np.array(validity))*100
avg_uniqueness = np.mean(np.array(uniqueness))*100
avg_step = total_step / (n_sample * len(start_frag_list))
end = time.time()
print(f'validity: {avg_validity:.1f}\t'
      f'uniqueness: {avg_uniqueness:.1f}\t'
      f'num step: {avg_step:.2f}\t'
      f'time: {(end-st):.5f}\t'
      f'time: {(end-st)/(n_sample*len(start_frag_list)):.5f}')

with open(output_file, 'w') as w :
    for start_frag in start_frag_list :
        w.write('\n'.join(sample_dict[start_frag]) + '\n')
