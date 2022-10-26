from rdkit import Chem
from rdkit.Chem import Descriptors
from tqdm import tqdm
from tdc import Oracle

import networkx as nx

SA_Oracle = Oracle(name = 'SA')

# modifed version of PyTDC's implementation
def penalized_logp(smi, normalize = False):
    """Evaluate LogP score of a SMILES string
        Args:
        smiles: str
        Returns:
        logp_score: float, between - infinity and + infinity 
    """  
    if smi is None: 
        return -100.0
    mol = Chem.MolFromSmiles(smi)
    if mol is None: 
        return -100.0

    log_p = Descriptors.MolLogP(mol)
    SA = -SA_Oracle(smi)

    # cycle score
    cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
    if len(cycle_list) == 0:
        cycle_length = 0
    else:
        cycle_length = max([len(j) for j in cycle_list])
    if cycle_length <= 6:
        cycle_length = 0
    else:
        cycle_length = cycle_length - 6
    cycle_score = -cycle_length

    if normalize :
        logP_mean = 2.4570953396190123
        logP_std = 1.434324401111988
        SA_mean = -3.0525811293166134
        SA_std = 0.8335207024513095
        cycle_mean = -0.0485696876403053
        cycle_std = 0.2860212110245455
        normalized_log_p = (log_p - logP_mean) / logP_std
        normalized_SA = (SA - SA_mean) / SA_std
        normalized_cycle = (cycle_score - cycle_mean) / cycle_std
        return normalized_log_p + normalized_SA + normalized_cycle
    else :
        return log_p + SA + cycle_score

smiles_desc_list = {
    'sa': Oracle(name = 'SA'),
    'qed': Oracle(name = 'qed'),
    'plogp': penalized_logp
}

# rdkit Descriptors
mol_desc_list = {
    'mw': Descriptors.ExactMolWt,
    'logp': Descriptors.MolLogP,
    'tpsa': Descriptors.TPSA,
}

property_list = ['mw', 'tpsa', 'logp', 'sa', 'plogp', 'qed']
floating_point = {
    'mw': 2,
    'tpsa': 3,
    'sa': 3,
    'logp': 5,
    'qed': 5,
    'plogp': 5,
}

with open('./all.txt') as f :
    lines = f.readlines()

with open('./property.db', 'w') as w :
    w.write(lines[0].strip() + ',' + ','.join(property_list) + '\n')
    for l in tqdm(lines[1:]) :
        smiles = l.split(',')[1]
        properties = []
        for key in property_list :
            if key in smiles_desc_list :
                value = smiles_desc_list[key](smiles)
                properties.append(f'{value:.{floating_point[key]}f}')
            else :
                mol = Chem.MolFromSmiles(smiles)
                value = mol_desc_list[key](mol)
                properties.append(f'{value:.{floating_point[key]}f}')
        w.write(f'{l.strip()},{",".join(properties)}\n')

