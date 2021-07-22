from rdkit import Chem
from rdkit.Chem import Bond
from typing import List

NUM_BOND_FEATURES = 6

def bond_features(bond: Bond) -> List[int] :
    bondtype = bond.GetBondType()
    conjugated = bond.GetIsConjugated()
    ring = bond.IsInRing()
    features = [
        bondtype == Chem.rdchem.BondType.SINGLE,
        bondtype == Chem.rdchem.BondType.DOUBLE,
        bondtype == Chem.rdchem.BondType.TRIPLE,
        bondtype == Chem.rdchem.BondType.AROMATIC,
        (1 if conjugated else 0),
        (1 if ring else 0)
    ]
    return features
