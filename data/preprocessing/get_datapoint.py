import multiprocessing
import pandas as pd
import argparse
import random
import time
import itertools
from typing import Set, Tuple, Dict, List
from functools import partial
import os
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

from __init__ import brics

def split(row: Dict, lib_map: Dict, lib_set: Set) -> List[Tuple[str, int, int]]:
    i, row = row
    molID, smiles = row['MolID'], row['SMILES'] 
    splitter = brics.BRICSSplitter(smiles, setup=False)
    bbs = splitter.brics_bonds
    frag_pair_set = dict()
    for i in range(1, len(bbs)+1) :
        for bbs_subset in itertools.combinations(bbs, i) :
            splitter.initialize()
            splitter.setup(bbs_subset)
            for frag2 in splitter :
                if frag2.smiles not in lib_set :
                    continue
                else :
                    for cxn in frag2.connection :
                        fidx1, atomidx1, BRICSidx1 = cxn[1]
                        fidx2, atomidx2, BRICSidx2 = cxn[0]
                        frag1 = splitter[fidx1]
                        frag1_connect_part = frag1.mol.GetAtomWithIdx(atomidx1)
                        frag1_connect_part_neigh = frag1_connect_part.GetNeighbors()[0].GetIdx()
                        frag1_smi_nostar, mapped_idx1 = brics.preprocess.remove_brics_label(frag1.mol, frag1_connect_part_neigh)
                        fid2 = lib_map.get((frag2.smiles, atomidx2), None)
                        if fid2 is None :
                            continue
                        composed_mol = brics.BRICSCompose.compose(frag1_smi_nostar, frag2.mol, mapped_idx1, atomidx2)
                        key = (frag1_smi_nostar, fid2, composed_mol)
                        if key not in frag_pair_set :
                            frag_pair_set[key] = mapped_idx1
    
    frag_pair_set = [(k[0], k[1], v) for k, v in frag_pair_set.items()]

    return frag_pair_set, molID

def main(args) :
    random.seed(42)
    molecules_file = os.path.join(args.run_directory, 'smiles', args.mol)
    library_map_file = os.path.join(args.run_directory, args.library_map)
    db = pd.read_csv(molecules_file)
    smiles_list = db.to_dict('records')
    smiles_list = [(i, s) for i, s in enumerate(smiles_list)]
    library_map = pd.read_csv(library_map_file, index_col=['SMILES','Idx']).to_dict('dict')['FID']
    library_set = {k[0] for k in library_map.keys()}
    split_ = partial(split, lib_map = library_map, lib_set = library_set) 
    print(f'multiprocessing start')
    pool = multiprocessing.Pool(args.cpus)
    poolget = pool.map(split_, smiles_list)
    pool.terminate()
    pool.join()
    res = list(poolget)
    print(f'multiprocessing finish')
    num_frag = 0
    num_mol = 0
    output_file = os.path.join(args.run_directory, args.output)
    with open(output_file, 'w') as w :
        w.write('SMILES,FID,Idx,MolID\n')
        for pair_set, molID in res :
            for pair in pair_set :
                w.write(f'{pair[0]},{pair[1]},{pair[2]},{molID}\n')
                num_frag += 1
        for _, row in smiles_list :
            molID, smiles = row['MolID'], row['SMILES'] 
            w.write(f'{smiles},-1,0,{molID}\n')
            num_mol+=1
    
    print(f"number of fragments: {num_frag}")
    print(f"number of molecules: {num_mol}")

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('run_directory', type=str)
    parser.add_argument('--mol', type=str)
    parser.add_argument('--library_map', type=str, default='library_map.csv')
    parser.add_argument('--output', type=str)
    parser.add_argument('--cpus', type=int, default=72)
    args = parser.parse_args()
    main(args)
