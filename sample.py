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
    start_frag_list = common.load_txt(cfg.start_frag_path)
    total_n_sample = cfg.n_sample * len(start_frag_list)
    sample_dict = {}
    total_step = 0

    st = time.time()
    for i, start_frag in enumerate(start_frag_list) :
        sample_list, n_step = mb.generate(start_frag, n_sample = cfg.n_sample)
        sample_dict[start_frag] = sample_list
        total_step += n_step
    end = time.time()

    validity = [len((sample_dict[start_frag])) / cfg.n_sample for start_frag in start_frag_list]
    uniqueness = [len(set(sample_dict[start_frag])) / len(sample_dict[start_frag]) for start_frag in start_frag_list \
                                                                                        if len(sample_dict[start_frag]) > 0]
    avg_validity = np.mean(np.array(validity))*100
    avg_uniqueness = np.mean(np.array(uniqueness))*100
    avg_step = total_step / total_n_sample
    """
    logging.info(f'validity: {avg_validity:.1f}\t'
                 f'uniqueness: {avg_uniqueness:.1f}\t'
                 f'num step: {avg_step:.2f}\t'
                 f'time: {(end-st):.5f}\t'
                 f'time: {(end-st)/total_n_sample:.5f}')
    """
    logging.info(f'{avg_validity:.1f}\t'
                 f'{avg_uniqueness:.1f}\t'
                 f'{avg_step:.2f}\t'
                 f'{(end-st):.5f}\t'
                 f'{(end-st)/total_n_sample:.5f}')

    for start_frag in start_frag_list :
        logger.log(sample_dict[start_frag])

if __name__ == '__main__' :
    main()
