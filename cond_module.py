import pandas as pd
import numpy as np
from collections import namedtuple, OrderedDict
from typing import List, Tuple
import gc

Scale = namedtuple('Scale', ('mean', 'std'))

"""
Save condition with standardized value. (Same function as Dictionary)
Cond_Module[MolID] -> Tuple[Float]

Cond_Module.scale: OrderedDict for standardize scale(mean, std) of each descriptors
Cond_Module.db: Dict[int, Tuple[Float]]
"""

class Cond_Module () :
    def __init__(self, db_file: str, target: List[str]) :
        usecols = ['MolID'] + target
        db = pd.read_csv(db_file, index_col = 0, usecols = usecols)
        self.scale = OrderedDict()
        for desc in target :
            mean = np.mean(db[desc])
            std = np.std(db[desc])
            self.scale[desc] = Scale(mean, std)
            db.update((db[desc]-mean)/std)
        self.db = {row.Index : tuple(getattr(row, desc) for desc in target) for row in db.itertuples()}
        del(db)
        gc.collect()

    def __getitem__(self, molID: int) -> Tuple[float] :
        return self.db[molID]
