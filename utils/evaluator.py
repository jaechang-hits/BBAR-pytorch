from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem

def canonicalize_smiles(smiles) :
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None :
        return Chem.MolToSmiles(mol)
    else :
        return None

def canonicalize_smiles_list(smiles_list) :
    smiles_list = list(map(canonicalize_smiles, smiles_list))
    smiles_list = list(filter(lambda x:x is not None, smiles_list))
    return smiles_list

def get_uniq_smiles_set(smiles_list) :
    return set(smiles_list)

def validity(smiles_list, num_gen) :
    return len(smiles_list) / num_gen

def uniqueness(smiles_list, smiles_set = None) :
    if smiles_set is None :
        smiles_set = get_uniq_smiles_set(smiles_list)
    return len(smiles_set) / len(smiles_list)

def novelty(smiles_set, train_smiles_set) :
    if isinstance(smiles_set, list) :
        smiles_set = get_uniq_smiles_set(smiles_set)
    if isinstance(train_smiles_set, list) :
        train_smiles_set = get_uniq_smiles_set(train_smiles_set)
    return len(smiles_set.difference(train_smiles_set)) / len(smiles_set)

def diversity(smiles_set) :
    if isinstance(smiles_set, list) :
        smiles_set = get_uniq_smiles_set(smiles_set)
    mol_set = [Chem.MolFromSmiles(smiles) for smiles in smiles_set]
    fps = [AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048, useChirality=False) for mol in mol_set]

    distance_sum = 0.
    for i in range(len(fps)) :
        fp1 = fps[i]
        for j in range(i+1, len(fps)) :
            fp2 = fps[j]
            distance = 1 - DataStructs.TanimotoSimilarity(fp1, fp2)
            distance_sum += distance
    diversity = distance_sum / (len(fps) * (len(fps)-1) / 2)
    return diversity

def property(smiles_list, fn, input_type: str = 'smiles', progress_bar: bool = True) :
    if input_type == 'mol': 
        _fn = lambda x:fn(Chem.MolFromSmiles(x))
    else :
        _fn = fn
    properties = list(map(_fn, smiles_list))
    return properties
