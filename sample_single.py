import numpy as np
import logging
import time

from utils import common
from utils.hydra_runner import hydra_runner
from utils.exp_manager import sample_manager

from src.generator import MoleculeBuilder

@hydra_runner(config_path='conf', config_name='sample')
def main(cfg) : 
    common.set_seed(0)
    mb_cfg = cfg.generator
    device = common.set_device(cfg.gpus)
    mb = MoleculeBuilder(mb_cfg, device, None)
    sample_smiles = mb(cfg.start_mol)
    if sample_smiles is not None :
        print(f"Sampled Molecule: {sample_smiles}")

if __name__ == '__main__' :
    main()
