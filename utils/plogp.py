from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from .sascorer.sascorer import sa_scorer
import networkx as nx
from tqdm import tqdm

def num_long_cycles(mol):
  """Calculate the number of long cycles.
  Args:
    mol: Molecule. A molecule.
  Returns:
    negative cycle length.
  """
  cycle_list = nx.cycle_basis(nx.Graph(Chem.rdmolops.GetAdjacencyMatrix(mol)))
  if not cycle_list:
    cycle_length = 0
  else:
    cycle_length = max([len(j) for j in cycle_list])
  if cycle_length <= 6:
    cycle_length = 0
  else:
    cycle_length = cycle_length - 6
  return -cycle_length

def plogp(molecule):
  log_p = Descriptors.MolLogP(molecule)
  sas_score = sascorer.calculateScore(molecule)
  cycle_score = num_long_cycles(molecule)
  return log_p - sas_score + cycle_score
