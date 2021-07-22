from rdkit import Chem
from rdkit.Chem import Atom
from typing import List, Union

# cited from https://github.com/chemprop/chemprop.git
# We add atom feature period, group, brics_idx, electronegativity
# We use only B, C, N, O, F, Si, P, S, Cl, Br, Sn, I

__all__ = ['NUM_ATOM_FEATURES', 'NUM_ATOM_FEATURES_BRICS', 'atom_features']

others=-100
ATOM_FEATURES = {
    'period': [0, 1, 2, 3, 4, 5],
    'group': [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'brics_idx': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
    'degree': [0, 1, 2, 3, 4, 5, others],
    'valence' : [0, 1, 2, 3, 4, 5, 6, 7, 8, others],
    'formal_charge': [-1, -2, 1, 2, 0, others],
    'num_Hs': [0, 1, 2, 3, 4, others],
    'hybridization': [
        Chem.rdchem.HybridizationType.UNSPECIFIED,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        others
    ]
#   'aromatics': 0 or 1 (Bool)
#   'mass': atom mass * 0.01 (Float)
#   'electonegativity': EN * 0.2 (Float)
}

electronegativity = {
    0: 0.00,    # *
    5: 2.04,    # B
    6: 2.55,    # C
    7: 3.04,    # N
    8: 3.44,    # O
    9: 3.98,    # F
    14: 1.90,   # Si
    15: 2.19,   # P
    16: 2.59,   # O
    17: 3.16,   # Cl
    35: 2.96,   # Br
    50: 1.96,   # Sn
    53: 2.66,   # I
}
    

NUM_ATOM_FEATURES_BRICS = sum(len(choices) for choices in ATOM_FEATURES.values()) + 3
# 6 + 9 + 17 + 7 + 10 + 6 + 6 + 7 + 1 + 2 = 71
NUM_ATOM_FEATURES = NUM_ATOM_FEATURES_BRICS - 17
# 6 + 9 + 0  + 7 + 10 + 6 + 6 + 7 + 1 + 2 = 54

def atom_features(atom: Atom, brics: bool) -> List[Union[int, float]]:
    atomic_num = atom.GetAtomicNum()
    period, group = _get_periodic_feature(atomic_num)
    brics_idx = atom.GetIsotope()
    degree = atom.GetTotalDegree()
    valence = atom.GetTotalValence()
    formal_charge = atom.GetFormalCharge()
    num_Hs = atom.GetTotalNumHs()
    aromatics = atom.GetIsAromatic()
    hybridization = atom.GetHybridization()
    mass = atom.GetMass()
    en = electronegativity[atomic_num]

    features = _onek_encoding_unk(period, ATOM_FEATURES['period']) + \
               _onek_encoding_unk(group, ATOM_FEATURES['group']) + \
               _onek_encoding_unk(degree, ATOM_FEATURES['degree']) + \
               _onek_encoding_unk(valence, ATOM_FEATURES['valence']) + \
               _onek_encoding_unk(formal_charge, ATOM_FEATURES['formal_charge']) + \
               _onek_encoding_unk(num_Hs, ATOM_FEATURES['num_Hs']) + \
               _onek_encoding_unk(hybridization, ATOM_FEATURES['hybridization']) + \
               [1 if aromatics else 0] + \
               [mass * 0.01, en * 0.2] #scaled to ablut the same range as other features

    if brics :
        features = features + _onek_encoding_unk(brics_idx, ATOM_FEATURES['brics_idx'])

    return features
               
_periodic_table = Chem.GetPeriodicTable()
def _get_periodic_feature(atomic_num: int) :
    periodic_list = [0, 2, 10, 18, 36, 54]
    for i in range(len(periodic_list)) :
        if periodic_list[i] >= atomic_num :
            period = i
            group = _periodic_table.GetNOuterElecs(atomic_num)
    return period, group

def _onek_encoding_unk(value: int, choices: List[int]) -> List[int] :
    encoding = [0] * len(choices)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1
    return encoding
