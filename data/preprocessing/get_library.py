from rdkit import Chem
from rdkit.Chem import Mol
import multiprocessing
import random
import pandas as pd
import argparse
import os
from typing import Union, List, Tuple

from __init__ import brics

def reallocate_frag(frag: Union[str, Mol], returnMols: bool = False) -> List[Tuple[int, str]] :
    """
    Allocate one new fragment for each connecting part of input fragment.
    >>> reallocate_frag('[14*]c1ccco1')
    [(0, '[14*]c1ccco1')]
    >>> reallocate_frag('[5*]N1CCN([5*])CC1')
    [(0, '[5*]N1CCNCC1'), (5, '[5*]N1CCNCC1')]
    >>> reallocate_frag('[8*]C[8*]')
    [(0, '[8*]C'), (2, '[8*]C')]
    """
    if isinstance(frag, str) :
        frag = Chem.MolFromSmiles(frag)
    stars = []
    for atom in frag.GetAtoms() :
        if atom.GetAtomicNum() == 0 :
            stars.append(atom.GetIdx())
    
    if len(stars) == 1 :
        if returnMols :
            return [(0, frag)]
        else :
            return [(0, Chem.MolToSmiles(frag))]

    retval = []
    for idx in stars :
        t = 0
        emol = Chem.EditableMol(frag)
        H = Chem.Atom(1)
        for idx_ in stars :
            if idx == idx_ :
                continue
            emol.ReplaceAtom(idx_, H)
        new_frag = Chem.MolToSmiles(Chem.MolFromSmiles(Chem.MolToSmiles(emol.GetMol())))
        if returnMols: 
            retval.append((idx, Chem.MolFromSmiles(new_frag)))
        else :
            retval.append((idx, new_frag))
    return retval

def decompose(row) :
    molId, smiles= row['MolID'], row['SMILES']
    try:
        frags = brics.BRICSSplitter.decompose(smiles, returnMols=False)
    except:
        return None

    retval = []
    for s in frags :
        if '*' not in s :
            continue
        m = Chem.MolFromSmiles(s)
        if m is None : return None
        s = Chem.MolToSmiles(m)
        splitted_frags = []

        for idx, ss in reallocate_frag(s) :
            m = Chem.MolFromSmiles(ss)
            if m is None : return None
            ss = Chem.MolToSmiles(m)
            splitted_frags.append((idx, ss))
            
        retval.append((s, splitted_frags))
    return (molId, smiles, retval)

def run(args) :
    random.seed(42)
    molecule_file = os.path.join(args.run_directory, args.mol)
    db = pd.read_csv(molecule_file)
    smiles_list = db.to_dict('records')
    pool = multiprocessing.Pool(args.cpus)
    poolget = pool.map(decompose, smiles_list)
    pool.terminate()
    pool.join()
    poolget = list(poolget)

    splitted_frag_log = {}
    frag_map = {}
    print(f'input: {len(poolget)}')
    for s in poolget :
        if s is None : continue 
        idx, smiles, retval = s
        for frag, splitted_list in retval :
            for idx, splitted_frag in splitted_list :
                splitted_frag_log[splitted_frag] = splitted_frag_log.get(splitted_frag, 0) + 1
                if (frag, idx) not in frag_map :
                    frag_map[(frag, idx)] = splitted_frag

    splitted_frag_list = {}
    for splitted_frag, n in splitted_frag_log.items() :
        if n in splitted_frag_list :
            splitted_frag_list[n].append(splitted_frag)
        else :
            splitted_frag_list[n] = [splitted_frag]

    key = list(splitted_frag_list.keys())
    key.sort(reverse=True)
    n_library = 0
    frag_to_fid = {}
    library_path = os.path.join(args.run_directory, args.library)
    with open(library_path, 'w') as w :
        w.write(f'FID,SMILES,frequency\n')
        for n in key :
            splitted_frag_set = splitted_frag_list[n]
            random.shuffle(splitted_frag_set)
            for splitted_frag in splitted_frag_set :
                w.write(f'{n_library},{splitted_frag},{n}\n')
                frag_to_fid[splitted_frag] = n_library
                n_library+=1

    print(f'library: {n_library}')

    used_list = set()
    library_map_path = os.path.join(args.run_directory, args.library_map)
    with open(library_map_path, 'w') as w :
        w.write(f'SMILES,Idx,FID\n')
        for (frag, idx), splitted_frag in frag_map.items() :
            fid = frag_to_fid[splitted_frag]
            w.write(f'{frag},{idx},{fid}\n')

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('run_directory', type=str)
    parser.add_argument('--mol', type=str, default='property.db')
    parser.add_argument('--library', type=str, default='library.csv')
    parser.add_argument('--library_map', type=str, default='library_map.csv')
    parser.add_argument('--cpus', type=int, default=72)
    args = parser.parse_args()
    run(args)
