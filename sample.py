import os
import numpy as np
import torch
import random
import logging
import time
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True

from generator import MoleculeBuilder
from utils import common
from utils.hydra_runner import hydra_runner
from utils.exp_manager import sample_manager

@hydra_runner(config_path='conf', config_name='sample')
def main(cfg) : 
    cfg, logger = sample_manager(cfg, cfg.save_dir)
    mb_cfg = cfg.generator

    mb = MoleculeBuilder(mb_cfg, None)
    if cfg.start_frag_path is not None :
        start_frag_list = common.load_txt(cfg.start_frag_path)
    else :
        start_frag_list = [None]
    total_n_sample = cfg.n_sample * len(start_frag_list)
    total_sample_list = []
    total_step = 0

    st = time.time()
    validity = []
    uniqueness = []
    for i, start_frag in enumerate(start_frag_list) :
        sample_list, n_step = mb.generate(start_frag, n_sample = cfg.n_sample)
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
