import compose
from rdkit import Chem

a1_ =    'Nc1c(C)cccc1C'
a2 = '[12*]S(=O)(=O)c1cc2c3c(c1)C(C)C(=O)N3CCC2'
print(compose.all_possible_compose(a1_, a2))
print(compose.compose(a1_, a2, 0, 0))
print(compose.get_possible_indexs(a1_, bidx2 = '12'))
print(compose.get_possible_indexs(a1_, frag2 = a2))
print(compose.get_possible_connections(a1_, a2))

a1_ = 'C(=O)CCNC=O'
a2 = '[10*]N1C(=O)COC1=O'
print(compose.get_possible_indexs(a1_, a2))
