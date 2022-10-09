import random
import numpy as np
from rdkit import Chem
from rdkit.Chem import Mol, Atom
from rdkit.Chem import rdChemReactions as Reactions
from typing import Union, List, Tuple, Optional, Dict
import re
import pandas as pd
import gc

from .constant import BRICS_ENV, BRICS_SMARTS_MOL, BRICS_SMARTS_FRAG

p = re.compile('\[\d+\*\]')

"""
composing style
Molecule + Fragment(With Star) -> Molecule
We do not use linker for fragment
"""

BRICS_substructure = {k:Chem.MolFromSmarts(v[0]) for k, v in BRICS_SMARTS_MOL.items()}

def compose(
    mol: Union[str, Mol],
    frag: Union[str, Mol],
    atom_idx_mol: int,
    atom_idx_frag: int,
    returnMol: bool = False,
    returnBricsType: bool = False,
    force: bool = False,
    warning: bool = False,
    ) -> Union[str, Mol, Tuple, None] :
    
    if isinstance(mol, str) :
        mol = Chem.MolFromSmiles(mol)
    else :
        mol = Chem.Mol(mol)

    if isinstance(frag, str) :
        frag = Chem.MolFromSmiles(frag)
    else :
        frag = Chem.Mol(frag)
    # Validity Check
    atom_mol = mol.GetAtomWithIdx(atom_idx_mol)
    atom_frag = frag.GetAtomWithIdx(atom_idx_frag)
    if (atom_frag.GetAtomicNum() != 0):
        if warning: print(f"ERROR: frag's {atom_idx_frag}th atom '{atom_frag.GetSymbol()}' should be [*].")
        if not force: return None
    brics_label_frag = str(atom_frag.GetIsotope())

    validity = False
    for brics_label in BRICS_ENV[brics_label_frag] :
        substructure = BRICS_substructure[brics_label]
        for idxs_list in mol.GetSubstructMatches(substructure) :
            if atom_idx_mol == idxs_list[0] :
                validity = True
                break
    if not validity :
        if warning: print(f"ERROR: mol's {atom_idx_mol}th atom '{atom_mol.GetSymbol()}' couldn't be connected with frag.")
        if not force: return None

    # Combine Molecules
    num_atoms_mol = mol.GetNumAtoms()
    neigh_atom_idx_frag = atom_frag.GetNeighbors()[0].GetIdx()
    bt = (Chem.rdchem.BondType.SINGLE if brics_label_frag != '7' else Chem.rdchem.BondType.DOUBLE)

    edit_mol = Chem.RWMol(Chem.CombineMols(mol, frag))
    atom_mol = edit_mol.GetAtomWithIdx(atom_idx_mol)
    if atom_mol.GetTotalNumHs() == atom_mol.GetNumExplicitHs():
        atom_mol.SetNumExplicitHs(atom_mol.GetNumExplicitHs()-1)
    edit_mol.AddBond(atom_idx_mol,
                     num_atoms_mol + neigh_atom_idx_frag,
                     order = bt)
    edit_mol.ReplaceAtom(atom_idx_mol, atom_mol)
    edit_mol.RemoveAtom(num_atoms_mol + atom_idx_frag)   # Remove Dummy Atom
    combined_mol = edit_mol.GetMol()

    if returnMol :
        Chem.SanitizeMol(combined_mol)
    else :
        combined_mol = Chem.MolToSmiles(combined_mol)

    if returnBricsType :
        return combined_mol, (brics_label_mol, brics_label_frag)
    else :
        return combined_mol

buildTemplate= []
buildReaction = []
for typ1, typ2list in BRICS_ENV.items() :
    for typ2 in typ2list :
        r1 = BRICS_SMARTS_MOL[typ1][0]
        r2, bond = BRICS_SMARTS_FRAG[typ2]
        react = '[$(%s):1].[$(%s):2]%s;!@[%s*]' % (r1, r2, bond, typ2)
        prod = '[*:1]%s;!@[*:2]' % (bond)
        tmpl = '%s>>%s' % (react, prod)
        buildTemplate.append(tmpl)
buildReaction = [Reactions.ReactionFromSmarts(template) for template in buildTemplate]

def all_possible_compose(mol: Union[str, Mol],
                         frag: Union[str, Mol]) -> Union[str, None] :
    
    if isinstance(mol, str) :
        mol = Chem.MolFromSmiles(mol)
    if isinstance(frag, str) :
        frag = Chem.MolFromSmiles(frag)
    
    possible_list = []
    for rxn in buildReaction :
        products = rxn.RunReactants((mol, frag))
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

def get_possible_indexs(mol: Union[str, Mol],
                        frag: Union[str, Mol, None] = None,
                        brics_label_frag: Optional[str] = None) -> List[Tuple[Tuple[int, str]]] :
    """
    Get Indexs which can be connected to target brics type for target fragment
    Return List[Tuple(AtomIndex:int, BRICSIndex:str)]
    Example
    >>> s1 = 'Nc1c(C)cccc1C'
    >>> get_possible_indexs(s1, brics_label_frag = '12')
    [(0, '5')]
    >>> s2 = 'C(=O)CCNC=O'
    >>> s2_ = '[10*]N1C(=O)COC1=O'
    >>> get_possible_indexs(s2, frag = s2_)
    [(0, '1'), (5, '1'), (2, '8'), (3, '8')]
    """
    assert (frag is None) or (brics_label_frag is None)
    if isinstance(mol, str) :
        mol = Chem.MolFromSmiles(mol)
    if isinstance(frag, str) :
        frag = Chem.MolFromSmiles(frag)

    if frag is not None :
        brics_label_frag = str(frag.GetAtomWithIdx(0).GetIsotope())
    
    idxs = []
    if brics_label_frag is not None :
        brics_list = BRICS_ENV[brics_label_frag]
    else :
        brics_list = list(BRICS_ENV.keys())

    for brics_label_mol in brics_list :
        substructure = BRICS_substructure[brics_label_mol]
        for idxs_list in mol.GetSubstructMatches(substructure) :
            atom_idx_mol = idxs_list[0]
            idxs.append((atom_idx_mol, brics_label_mol))
    return idxs

def get_possible_brics_labels(mol: Union[str, Mol],
                       atom_idx: Optional[str] = None) -> List[Tuple[Tuple[int, str]]] :
    def fast_brics_search(atom: Atom) :
        atomicnum = atom.GetAtomicNum()
        aromatic = atom.GetIsAromatic()
        if atomicnum == 6 :
            if aromatic :
                return ['14', '16']
            else :
                return ['1', '4', '6', '7', '8', '13', '15']
        elif atomicnum == 7 :
            if aromatic :
                return ['9']
            else :
                return ['5', '10']
        elif atomicnum == 8 : 
            return ['3']
        elif atomicnum == 16 :
            return ['11', '12']
        else :
            return []
    if isinstance(mol, str) :
        mol = Chem.MolFromSmiles(mol)
    
    labels = []
    if atom_idx is not None :
        brics_list = fast_brics_search(mol.GetAtomWithIdx(atom_idx))
    else :
        brics_list = list(BRICS_ENV.keys())

    for brics_label in brics_list :
        substructure = BRICS_substructure[brics_label]
        for idxs_list in mol.GetSubstructMatches(substructure) :
            __atom_idx = idxs_list[0]
            if atom_idx is None or atom_idx == __atom_idx:
                labels.append(brics_label)
                break

    return labels

def get_possible_connections(mol: Union[str, Mol],
                             frag: Union[str, Mol]) -> List[Tuple[Tuple[int, int], Tuple[str, str]]] :
    """
    Get all possible connections between mol(w/o *) and frag(w/ *)
    Return List[Tuple(Tuple(AtomIndex1:int, AtomIndex2:int), Tuple(BRICSIndex1:str, BRICSIndex2:str))]
    Example
    >>> s1 = 'Nc1c(C)cccc1C'
    >>> s2 = '[12*]S(=O)(=O)c1cc2c3c(c1)C(C)C(=O)N3CCC2'
    >>> get_possible_connections(s1, s2)
    [((0, 0), ('5', '12'))]
    """
    if isinstance(mol, str) :
        mol = Chem.MolFromSmiles(mol)
    if isinstance(frag, str) :
        frag = Chem.MolFromSmiles(frag)
    idx_set2 = get_broken(frag)
    connections = []
    for atom_idx_frag, brics_label_frag in idx_set2 :
        for atom_idx_mol, brics_label_mol in get_possible_indexs(mol, brics_label_frag = brics_label_frag) :
            connections.append(((atom_idx_mol, atom_idx_frag), (brics_label_mol, brics_label_frag)))
    return connections

