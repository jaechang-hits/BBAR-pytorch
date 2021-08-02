import os
import logging
from pathlib import Path
from omegaconf import OmegaConf
from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
desc_dic = Descriptors.__dict__
desc_key = desc_dic.keys()

def train_manager(cfg, exp_dir='result') :
    save_dir = os.path.join(exp_dir, cfg.name)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(save_dir, 'output.log')
    conf_file = os.path.join(save_dir, 'config.yaml')
    
    filehandler = logging.FileHandler(log_file, 'w')
    logger = logging.getLogger()
    logger.addHandler(filehandler)

    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)

    logging.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    with open(conf_file, 'w') as w :
        OmegaConf.save(config=cfg, f=w, resolve=True)

    return cfg, save_dir

exp_manager = train_manager

def sample_manager(cfg, exp_dir='sample') :
    save_dir = os.path.join(exp_dir, cfg.name+'.csv')
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg = OmegaConf.create(cfg)

    if cfg.save_property:
        descs = list(cfg.condition.keys())
        descs.sort()
    else :
        descs = []

    logger = SampleLogger(save_dir, descs, cfg.formatter)

    return cfg, logger

class SampleLogger() :
    def __init__(self, save_dir, desc, formatter) :
        self.save_dir = save_dir
        self.desc_fn = {}
        self.desc = []
        self.formatter = {}
        for d in desc:
            if d not in desc_key :
                logging.warning(f"WARNING: {d} doesn't exist in rdkit.Chem.Descriptors")
            else :
                self.desc.append(d)
                self.desc_fn[d] = desc_dic[d]
                self.formatter[d] = formatter.get(d, '.3f')
        with open(save_dir, 'w') as w :
            w.write('SMILES,'+','.join(self.desc) + '\n')

    def log(self, smiles_list) :
        mol_list = [Chem.MolFromSmiles(s) for s in smiles_list]
        if len(smiles_list) != len(mol_list) :
            smiles_list = [Chem.MolToSmiles(m) for m in mol_list]
        prob_result = {}
        for d in self.desc :
            prob_result[d] = [self.desc_fn[d](m) for m in mol_list]
        
        with open(self.save_dir, 'a') as w :
            for i, s in enumerate(smiles_list) :
                str_prob = [format(prob_result[d][i], self.formatter[d]) for d in self.desc]
                w.write(f"{s},{','.join(str_prob)}\n")

        
