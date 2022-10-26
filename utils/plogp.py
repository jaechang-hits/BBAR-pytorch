from rdkit import Chem
from rdkit.Chem import Descriptors
from .SA_Score import sascorer
import networkx as nx

# modifed version of PyTDC's implementation
def plogp(mol, normalize = False):
    """Evaluate LogP score of a SMILES string
        Args:
        #smiles: str
        Returns:
        logp_score: float, between - infinity and + infinity 
    """  
    #mol = Chem.MolFromSmiles(smi)

    log_p = Descriptors.MolLogP(mol)
    SA = - sascorer.calculateScore(mol)

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

#def num_long_cycles(mol):
#  """Calculate the number of long cycles.
#  Args:
#    mol: Molecule. A molecule.
#  Returns:
#    negative cycle length.
#  """
#  cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
#  if not cycle_list:
#    cycle_length = 0
#  else:
#    cycle_length = max([len(j) for j in cycle_list])
#  if cycle_length <= 6:
#    cycle_length = 0
#  else:
#    cycle_length = cycle_length - 6
#  return -cycle_length
#
#def plogp(molecule):
#  log_p = Descriptors.MolLogP(molecule)
#  sas_score = sascorer.calculateScore(molecule)
#  cycle_score = num_long_cycles(molecule)
#  return log_p - sas_score + cycle_score

