from rdkit import Chem
from rdkit.Chem.Descriptors import ExactMolWt
import random

with open('start.smi') as f :
    lines = f.readlines()

random.shuffle(lines)
t = 0
with open('start_docking.smi', 'w') as w :
    for l in lines :
        s = l.strip()
        m = Chem.MolFromSmiles(s)
        if ExactMolWt(m) < 250 :
            w.write(l)
            t+=1
            if t == 10000 :
                break
