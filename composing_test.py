import pandas as pd
from rdkit import Chem
from rdkit.Chem import Mol
from typing import Union, Tuple
import re

BRICS_TYPE = {
    '1': ['3', '5', '10'],
    '3': ['1', '4', '13', '14', '15', '16'],
    '4': ['3', '5', '11'],
    '5': ['1', '4', '12', '13', '14', '15', '16'],
    '6': ['13', '14', '15', '16'],
    '7': ['7'],
    '8': ['9', '10', '13', '14', '15', '16'],
    '9': ['8', '13', '14', '15', '16'],
    '10': ['1', '8', '13', '14', '15', '16'],
    '11': ['4', '13', '14', '15', '16'],
    '12': ['5'],
    '13': ['3', '5', '6', '8', '9', '10', '11', '14', '15', '16'],
    '14': ['3', '5', '6', '8', '9', '10', '11', '13', '14', '15', '16'],
    '15': ['3', '5', '6', '8', '9', '10', '11', '13', '14', '16'],
    '16': ['3', '5', '6', '8', '9', '10', '11', '13', '14', '15', '16'],
}
BRICS_TYPE_INT = {k: [int(_) for _ in v] for k, v in BRICS_TYPE.items()}

BRICS_SMARTS_FRAG = {
  '1': ('[C;D3]([#0,#6,#7,#8])(=O)', '-'),
  '3': ('[O;D2]-;!@[#0,#6,#1]', '-'), 
  '4': ('[C;!D1;!$(C=*)]-;!@[#6]', '-'), 
  '5': ('[N;!D1;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)]', '-'),
  '6': ('[C;D3;!R](=O)-;!@[#0,#6,#7,#8]', '-'), 
  '7': ('[C;D2,D3]-[#6]', '='),
  '8': ('[C;!R;!D1;!$(C!-*)]', '-'), 
  '9': ('[n;+0;$(n(:[c,n,o,s]):[c,n,o,s])]', '-'),
  '10': ('[N;R;$(N(@C(=O))@[C,N,O,S])]', '-'), 
  '11': ('[S;D2](-;!@[#0,#6])', '-'), 
  '12': ('[S;D4]([#6,#0])(=O)(=O)', '-'), 
  '13': ('[C;$(C(-;@[C,N,O,S])-;@[N,O,S])]', '-'), 
  '14': ('[c;$(c(:[c,n,o,s]):[n,o,s])]', '-'),
  '15': ('[C;$(C(-;@C)-;@C)]', '-'), 
  '16': ('[c;$(c(:c):c)]', '-'),
}

BRICS_SMARTS_MOL = {
  '1': ('[C;D2]([#0,#6,#7,#8])(=O)', '-'),
  '3': ('[O;D1]-;!@[#0,#6,#1]', '-'), 
  '4': ('[C;!$(C=*)]-;!@[#6]', '-'), 
  '5': ('[N;!$(N=*);!$(N-[!#6;!#16;!#0;!#1]);!$([N;R]@[C;R]=O)]', '-'),
  '6': ('[C;D2;!R](=O)-;!@[#0,#6,#7,#8]', '-'), 
  '7': ('[C;D1,D2]-[#6]', '='),
  '8': ('[C;!R;!$(C!-*)]', '-'), 
  '9': ('[n;+0;$(n(:[c,n,o,s]):[c,n,o,s])]', '-'),
  '10': ('[N;R;$(N(@C(=O))@[C,N,O,S])]', '-'), 
  '11': ('[S;D1](-;!@[#0,#6])', '-'), 
  '12': ('[S;D3]([#6,#0])(=O)(=O)', '-'), 
  '13': ('[C;$(C(-;@[C,N,O,S])-;@[N,O,S])]', '-'), 
  '14': ('[c;$(c(:[c,n,o,s]):[n,o,s])]', '-'),
  '15': ('[C;$(C(-;@C)-;@C)]', '-'), 
  '16': ('[c;$(c(:c):c)]', '-'),
}

BRICS_substructure = {k:Chem.MolFromSmarts(v[0]) for k, v in BRICS_SMARTS_MOL.items()}


def compose1(
    frag1: Union[str, Mol],
    frag2: Union[str, Mol],
    idx1: int,
    idx2: int,
    returnMols: bool = False,
    returnBricsType: bool = False
    ) -> Union[str, Mol, Tuple, None] :
    
    if isinstance(frag1, str) :
        frag1 = Chem.MolFromSmiles(frag1)
    if isinstance(frag2, str) :
        frag2 = Chem.MolFromSmiles(frag2)
    # Validity Check
    atom1 = frag1.GetAtomWithIdx(idx1)
    atom2 = frag2.GetAtomWithIdx(idx2)
    if (atom1.GetAtomicNum() != 0) or (atom2.GetAtomicNum() != 0) :
        print(f"ERROR: frag1's {idx1}th atom '{atom1.GetSymbol()}' and frag2's {idx2}th atom '{atom2.GetSymbol()}' should be [*].")
        return None
    bricsidx1 = str(atom1.GetIsotope())
    bricsidx2 = str(atom2.GetIsotope())
    if (bricsidx2 not in BRICS_TYPE[bricsidx1]) :
        print(f"ERROR: connection between '{bricsidx1}'(frag1) and '{bricsidx2}'(frag2) is not allowed.")
        return None
    
    # Combine Molecules
    num_atoms1 = frag1.GetNumAtoms()
    neigh_atom_idx1 = atom1.GetNeighbors()[0].GetIdx()
    neigh_atom_idx2 = atom2.GetNeighbors()[0].GetIdx()
    bt = (Chem.rdchem.BondType.SINGLE if bricsidx1 != '7' else Chem.rdchem.BondType.DOUBLE)
    
    starting_mol = Chem.CombineMols(frag1, frag2)
    edit_mol = Chem.EditableMol(starting_mol)
    edit_mol.AddBond(neigh_atom_idx1,
                     num_atoms1 + neigh_atom_idx2,
                     order = bt)
    edit_mol.RemoveAtom(idx1)
    edit_mol.RemoveAtom(num_atoms1 + idx2 - 1)    #idx1'th atom is removed
    
    combined_mol = edit_mol.GetMol()
    retval = Chem.MolToSmiles(combined_mol)
    if returnMols :
        retval = Chem.MolFromSmiles(combined_smiles)
    if returnBricsType :
        return retval, (bricsidx1, bricsidx2)
    else :
        return retval

def compose2(
    frag1: Union[str, Mol],
    frag2: Union[str, Mol],
    idx1: int,
    idx2: int,
    returnMols: bool = False,
    returnBricsType: bool = False,
    warning: bool = False,
    ) -> Union[str, Mol, Tuple, None] :
    
    if isinstance(frag1, str) :
        frag1 = Chem.MolFromSmiles(frag1)
    if isinstance(frag2, str) :
        frag2 = Chem.MolFromSmiles(frag2)

    #print(f'compose {Chem.MolToSmiles(frag1)} + {Chem.MolToSmiles(frag2)}')
    # Validity Check
    atom1 = frag1.GetAtomWithIdx(idx1)
    atom2 = frag2.GetAtomWithIdx(idx2)
    if (atom2.GetAtomicNum() != 0):
        if warning: print(f"ERROR: frag2's {idx2}th atom '{atom2.GetSymbol()}' should be [*].")
        return None
    bricsidx2 = str(atom2.GetIsotope())

    validity = False
    for bricsidx1 in BRICS_TYPE[bricsidx2] :
        substructure = BRICS_substructure[bricsidx1]
        for idxs_list in frag1.GetSubstructMatches(substructure) :
            if idx1 == idxs_list[0] :
                validity = True
                break
    if not validity :
        if warning: print(f"ERROR: frag1's {idx1}th atom '{atom1.GetSymbol()}' couldn't be connected with frag2.")
        return None

    atom1.SetNoImplicit(False)
    atom1.SetNumExplicitHs(0)

    # Combine Molecules
    num_atoms1 = frag1.GetNumAtoms()
    neigh_atom_idx2 = atom2.GetNeighbors()[0].GetIdx()
    bt = (Chem.rdchem.BondType.SINGLE if bricsidx2 != '7' else Chem.rdchem.BondType.DOUBLE)
    
    starting_mol = Chem.CombineMols(frag1, frag2)
    edit_mol = Chem.EditableMol(starting_mol)
    edit_mol.AddBond(idx1,
                     num_atoms1 + neigh_atom_idx2,
                     order = bt)
    edit_mol.RemoveAtom(num_atoms1 + idx2)
    edit_mol.ReplaceAtom(idx1, atom1)
    combined_mol = edit_mol.GetMol()

    retval = Chem.MolToSmiles(combined_mol)
    if returnMols :
        retval = Chem.MolFromSmiles(combined_smiles)
    if returnBricsType :
        return retval, (bricsidx1, bricsidx2)
    else :
        return retval

def remove_brics(smi) :
    new_smi = re.sub('\[\d+\*\]', '[H]', smi)
    new_mol = Chem.MolFromSmiles(new_smi)
    new_smi = Chem.MolToSmiles(new_mol)
    new_mol = Chem.MolFromSmiles(new_smi)
    return Chem.MolToSmiles(new_mol)

now_data = pd.read_csv('data/train_uniq_.csv')
with open('data/library.csv') as f :
    lines = f.readlines()[1:]
    now_lib = [Chem.MolFromSmiles(l.split(',')[1]) for l in lines]
prev_data = pd.read_csv('data/train_jaechang1.csv')
with open('data/old_library.csv') as f :
    lines = f.readlines()
    prev_lib = [Chem.MolFromSmiles(l.split(',')[0]) for l in lines]

comp_smi1_ = []
comp_smi1 = []
comp_smi2 = []

t = 0
for tup in prev_data.itertuples() :
    if t == 1000 : break
    t += 1
    smi = compose1(tup.SMILES, prev_lib[tup.FID], tup.Index1, tup.Index2)
    comp_smi1_.append(smi)
    comp_smi1.append(remove_brics(smi))

t = 0
for tup in now_data.itertuples() :
    if t == 1000 : break
    t += 1
    try :
        comp_smi2.append(compose2(tup.SMILES, now_lib[tup.FID], tup.Idx, 0))
    except:
        print(tup)
        exit(1)

for i, (smi_, smi1, smi2) in enumerate(zip(comp_smi1_, comp_smi1, comp_smi2)) :
    if (smi1 != smi2) :
        print('---------------')
        #print(prev_data.iloc[i])
        print(smi_)
        print(smi1)
        print(smi2)
