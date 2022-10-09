from rdkit import Chem
from rdkit.Chem import Mol, BRICS
from typing import Union, Optional, List, Tuple
"""
BRICSFragment:
    Object to save fragment's state. (similar to rdkit.Chem.Mol Object)
    It has smiles, idx, connection.
    smiles, idx, connection is from BRICSSplitter.setup()

BRICSSplitter:
    Object that divides target molecule to fragments and expresses them in BRICSFragment.
    It is iterable and we can handle it similar to list.
    >>> bs = BRICSSplitter('CCOC(=O)N1CCN(C(=O)c2ccoc2)CC1')
    >>> bs
    <BRICSSplitter '[4*]CC.[3*]O[3*].[1*]C([1*])=O.[5*]N1CCN([5*])CC1.[1*]C([6*])=O.[16*]c1ccoc1'>
    >>> bs[0]
    <BRICSFragment 0()  '[4*]CC'    connection: [((0, 0, '4'), (1, 2, '3'))]>
    >>> # <BRICSFragment idx(fid=None)   smiles  connection> 
    >>> print([frag.smiles for frag in bs])
    ['[4*]CC', '[3*]O[3*]', '[1*]C([1*])=O', '[5*]N1CCN([5*])CC1', '[1*]C([6*])=O', '[16*]c1ccoc1']
"""

class BRICSFragment() :
    def __init__(self, frag: str, idx: int) :
        """
        representation of connection :
        [ ((fragmentidx1, atomidx1, BRICSidx1), (fragmentidx2, atomidx2, BRICSidx2))) ]
        connection bewteen idx1'th fragment's atomidx1'th atom and idx2'th fragment's atomidx2'th atom.
        """
        self.smiles: str = frag
        self.idx: int = idx
        self._mol: Optional[Mol] = None
        self.connection: List[Tuple] = []
            
    def __repr__(self) :
        return str(f"<BRICSFragment {self.idx}()\t'{self.smiles}'\tconnection: {self.connection}>")

    def __str__(self) :
        return self.__repr__()
   
    @property
    def mol(self) :
        if self._mol is None :
            self._mol = Chem.MolFromSmiles(self.smiles)
        return self._mol


class BRICSSplitter() :
    def __init__(self, mol: Union[str, Mol], setup: bool = True) :
        if isinstance(mol, Mol) :
            smiles = Chem.MolToSmiles(mol)
            mol = Chem.MolFromSmiles(smiles)
        if isinstance(mol, str) :
            mol = Chem.MolFromSmiles(mol)
            smiles = Chem.MolToSmiles(mol)

        self.smiles = smiles
        self.mol = mol
        self.frag_set = []
        self.brics_bonds = list(BRICS.FindBRICSBonds(self.mol))
        if setup:
            self.setup()
    
    def initialize(self) :
        self.frag_set = []

    def setup(self, brics_bonds: Optional[List] = None) :
        if brics_bonds is None :
            brics_bonds = self.brics_bonds
        
        # Put Label (Start from 1)
        brics_bonds = [(a[0], (str((label+1)*100 + int(a[1][0])), str((label+1)*100 + int(a[1][1])))) for label, a in enumerate(brics_bonds)]

        # Decompose
        breakmol = BRICS.BreakBRICSBonds(self.mol, brics_bonds)
        frag_set = list(Chem.GetMolFrags(breakmol, asMols=True))
        idx_pair = {}
        for f_idx, f in enumerate(frag_set) :
            smiles, idx_set = _remove_frag_label(f)
            if smiles is None :
                self.initialize()
                print(f"WARNING: '{self.smiles}' can't be decomposed. \
                  It is caused by a bug in the Chem.MolToSmiles() function of the RdKit module in some bridged molecules.")
                return None
            self.frag_set.append(BRICSFragment(smiles, f_idx))
            for label, atom_idx, brics_label in idx_set :
                if label in idx_pair :
                    idx_pair[label].append((f_idx, atom_idx, brics_label))
                else :
                    idx_pair[label] = [(f_idx, atom_idx, brics_label)]
        for _, values in idx_pair.items() :
            if len(values)<2:
                """
                If the input molecule was not completed,
                it couldn't track the broken part properly
                example input: '*COCc1ccccc1'
                """
                continue
            f_idx1, atom_idx1, brics_label1 = values[0]
            f_idx2, atom_idx2, brics_label2 = values[1]
            self.frag_set[f_idx1].connection.append(((f_idx1, atom_idx1, brics_label1), (f_idx2, atom_idx2, brics_label2)))
            self.frag_set[f_idx2].connection.append(((f_idx2, atom_idx2, brics_label2), (f_idx1, atom_idx1, brics_label1)))

    def __len__(self) :
        return len(self.frag_set)

    def __iter__(self) :
        return iter(self.frag_set)

    def __getitem__(self, idx) :
        return self.frag_set[idx]

    def __str__(self) :
        return "<BRICSSplitter '"+'.'.join([f.smiles for f in self.frag_set])+"'>"


    @staticmethod
    def decompose(mol: Union[str, Mol], returnMols: bool = False) -> List[str] :
        """Use it when you are only interested in set of substructures
           and are not interested in the connections between substructures."""
        if isinstance(mol, str) :
            mol = Chem.MolFromSmiles(mol)
        bricsbonds = BRICS.FindBRICSBonds(mol)
        breakmol = BRICS.BreakBRICSBonds(mol, bricsbonds)
        frag_set = list(Chem.GetMolFrags(breakmol, asMols=True))
        if returnMols :
            frag_set = [Chem.MolFromSmiles(Chem.MolToSmiles(f)) for f in frag_set]
        else :
            frag_set = [Chem.MolToSmiles(f) for f in frag_set]

        return frag_set

def _remove_frag_label(frag: Union[str, Mol]) -> Tuple[str, List[Tuple[int, int, str]]] :
    """
    output
    (smiles, [(label, atom idx, brics idx)])
    
    Note :
        Atom indexing can change during remove label.
        - before removing
          [713*]CCC(C)C[1005*]          (SMILES)
        - after converting to Mol
          [713*]CCC(C)C[1005*]          (Mol)
        - after removing label 
          [13*]CCC(C)C[5*]              (Mol)
        - after translating to SMILES
          [5*]CC(C)CC[13*]              (SMILES)

        To solve this problem, we use following process.
    """ 
    if isinstance(frag, str) :
        frag = Chem.MolFromSmiles(frag)
    idx_list = []
    for atom in frag.GetAtoms() :
        if atom.GetAtomicNum() == 0 :
            atom_idx = atom.GetIdx()
            isotope = atom.GetIsotope()
            brics_label = isotope % 100
            label = int(isotope // 100)
            atom.SetIsotope(brics_label)
            idx_list.append((label, atom_idx, str(brics_label)))

    convert_smiles = Chem.MolToSmiles(frag)
    convert_mol = Chem.MolFromSmiles(convert_smiles)
    if convert_mol is None:
        return None, None
    idxmap = {a1:a2 for a1, a2 in enumerate(convert_mol.GetSubstructMatch(frag))}
    if len(idxmap) == 0 :
        idxmap = {a1:a2 for a2, a1 in enumerate(frag.GetSubstructMatch(convert_mol))}
        if len(idxmap) == 0 :
            return None, None

    for i in range(len(idx_list)) :
        label, atom_idx, brics_label = idx_list[i]
        idx_list[i] = (label, idxmap[atom_idx], brics_label)

    return convert_smiles, idx_list      
