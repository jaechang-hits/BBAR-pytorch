import os
import logging
from pathlib import Path
from omegaconf import OmegaConf

def exp_manager(cfg, exp_dir='result') :
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
