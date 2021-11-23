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
    cfg, logger = sample_manager(cfg, cfg.exp_dir)
    mb_cfg = cfg.generator
    device = common.set_device(cfg.gpus)

    assert cfg.start_mol is None or cfg.start_mol_path is None
    if cfg.start_mol is not None :
        start_list = [cfg.start_mol]
    elif cfg.start_mol_path is not None :
        start_list = common.load_txt(cfg.start_mol_path)
    else :
        start_list = [None]
    total_n_sample = cfg.n_sample * len(start_list)
    total_sample_list = []
    total_step = 0

    mb = MoleculeBuilder(mb_cfg, device, None)

    st = time.time()
    validity = []
    uniqueness = []
    for i, start_mol in enumerate(start_list) :
        sample_list, n_step = mb.generate(start_mol, n_sample = cfg.n_sample)
        validity.append(len(sample_list) / cfg.n_sample)
        if len(sample_list) > 0 :
            uniqueness.append(len(set(sample_list)) / len(sample_list))
        total_step += n_step
        logger.log(sample_list)
    end = time.time()

    avg_validity = np.mean(np.array(validity))*100

    try :
        avg_uniqueness = np.mean(np.array(uniqueness))*100
    except :
        avg_uniqueness = np.NAN

    avg_step = total_step / total_n_sample
    logging.info(f'validity: {avg_validity:.1f}\t'
                 f'uniqueness: {avg_uniqueness:.1f}\t'
                 f'num step: {avg_step:.2f}\t'
                 f'time: {(end-st)/total_n_sample:.5f}\t'
                 f'total time: {(end-st):.5f}')

if __name__ == '__main__' :
    main()
