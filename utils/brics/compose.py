import random
import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol
from rdkit.Chem import rdChemReactions as Reactions
from typing import Union, List, Tuple, Optional, Dict
import re
import pandas as pd
import gc
p = re.compile('\[\d+\*\]')

"""
composing style
Molecule + Fragment(With Star) -> Molecule
We do not use linker for fragment
"""

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

def compose(
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

    # check explicit Hydrogen
    explicitH = -1
    if atom1.GetNumExplicitHs() > 0 :
        for a in atom1.GetNeighbors() :
            if a.GetAtomicNum() == 1 :
                explicitH = a.GetIdx()
    
    # Combine Molecules
    num_atoms1 = frag1.GetNumAtoms()
    neigh_atom_idx2 = atom2.GetNeighbors()[0].GetIdx()
    bt = (Chem.rdchem.BondType.SINGLE if bricsidx1 != '7' else Chem.rdchem.BondType.DOUBLE)
    
    starting_mol = Chem.CombineMols(frag1, frag2)
    edit_mol = Chem.EditableMol(starting_mol)
    edit_mol.AddBond(idx1,
                     num_atoms1 + neigh_atom_idx2,
                     order = bt)
    edit_mol.RemoveAtom(num_atoms1 + idx2)
    if explicitH >= 0 :
        edit_mol.RemoveAtom(explicitH)
    
    combined_mol = edit_mol.GetMol()
    retval = Chem.MolToSmiles(combined_mol)
    if returnMols :
        retval = Chem.MolFromSmiles(combined_smiles)
    if returnBricsType :
        return retval, (bricsidx1, bricsidx2)
    else :
        return retval

buildTemplate= []
buildReaction = []
for typ1, typ2list in BRICS_TYPE.items() :
    for typ2 in typ2list :
        r1 = BRICS_SMARTS_MOL[typ1][0]
        r2, bond = BRICS_SMARTS_FRAG[typ2]
        react = '[$(%s):1].[$(%s):2]%s;!@[%s*]' % (r1, r2, bond, typ2)
        prod = '[*:1]%s;!@[*:2]' % (bond)
        tmpl = '%s>>%s' % (react, prod)
        buildTemplate.append(tmpl)
buildReaction = [Reactions.ReactionFromSmarts(template) for template in buildTemplate]

def all_possible_compose(frag1: Union[str, Mol],
                         frag2: Union[str, Mol]) -> Union[str, None] :
    
    if isinstance(frag1, str) :
        frag1 = Chem.MolFromSmiles(frag1)
    if isinstance(frag2, str) :
        frag2 = Chem.MolFromSmiles(frag2)
    
    possible_list = []
    for rxn in buildReaction :
        products = rxn.RunReactants((frag1, frag2))
        for p in products :
            possible_list.append(Chem.MolToSmiles(p[0]))
        
    return possible_list

def get_broken(frag: Union[str, Mol]) -> List[Tuple[int, str]] :
    if isinstance(frag, str) :
        frag = Chem.MolFromSmiles(frag)
    broken_idx_list = []
    for atom in frag.GetAtoms() :
        if atom.GetAtomicNum() == 0 :
            broken_idx_list.append((atom.GetIdx(), str(atom.GetIsotope())))
    return broken_idx_list

def get_possible_indexs(frag1: Union[str, Mol],
                        frag2: Union[str, Mol, None] = None,
                        bidx2: Optional[str] = None) -> List[Tuple[Tuple[int, str]]] :
    """
    Get Indexs which can be connected to target brics type for target fragment
    Return List[Tuple(AtomIndex:int, BRICSIndex:str)]
    Example
    >>> s1 = 'Nc1c(C)cccc1C'
    >>> get_possible_indexs(s1, bidx2 = '12')
    [(0, '5')]
    >>> s2 = 'C(=O)CCNC=O'
    >>> s2_ = '[10*]N1C(=O)COC1=O'
    >>> get_possible_indexs(s2, frag2 = s2_)
    [(0, '1'), (5, '1'), (2, '8'), (3, '8')]
    """
    assert (frag2 is not None) ^ (bidx2 is not None)
    if isinstance(frag1, str) :
        frag1 = Chem.MolFromSmiles(frag1)
    if isinstance(frag2, str) :
        frag2 = Chem.MolFromSmiles(frag2)

    if frag2 is None :
        bidx2_list = [bidx2]
    else :
        bidx2_list = [bidx for aidx, bidx in get_broken(frag2)]
    
    idxs = []
    for bidx2 in bidx2_list :
        for bidx1 in BRICS_TYPE[bidx2] :
            substructure = BRICS_substructure[bidx1]
            for idxs_list in frag1.GetSubstructMatches(substructure) :
                aidx1 = idxs_list[0]
                idxs.append((aidx1, bidx1))
    return idxs

def get_possible_connections(frag1: Union[str, Mol],
                             frag2: Union[str, Mol]) -> List[Tuple[Tuple[int, int], Tuple[str, str]]] :
    """
    Get all possible connections between frag1(w/o *) and frag2(w/ *)
    Return List[Tuple(Tuple(AtomIndex1:int, AtomIndex2:int), Tuple(BRICSIndex1:str, BRICSIndex2:str))]
    Example
    >>> s1 = 'Nc1c(C)cccc1C'
    >>> s2 = '[12*]S(=O)(=O)c1cc2c3c(c1)C(C)C(=O)N3CCC2'
    >>> get_possible_connections(s1, s2)
    [((0, 0), ('5', '12'))]
    """
    if isinstance(frag1, str) :
        frag1 = Chem.MolFromSmiles(frag1)
    if isinstance(frag2, str) :
        frag2 = Chem.MolFromSmiles(frag2)
    idx_set2 = get_broken(frag2)
    connections = []
    for aidx2, bidx2 in idx_set2 :
        for aidx1, bidx1 in get_possible_indexs(frag1, bidx2 = bidx2) :
            connections.append(((aidx1, aidx2), (bidx1, bidx2)))
    return connections

