import logging
from omegaconf import OmegaConf
import pathlib

from src.trainer import Trainer

from utils import common
from utils.argparser import Train_ArgParser
from utils.experiment import setup_logger

def setup_trainer() :
    parser = Train_ArgParser()
    args = parser.parse_args()
    
    # Setup Logger
    run_dir: pathlib.Path = setup_logger(args.exp_dir, args.name)
    config_path = run_dir / 'config.yaml'
    checkpoint_dir = run_dir / 'checkpoint'

    # Setup Trainer
    trainer_cfg = OmegaConf.load(args.trainer_config)
    model_cfg = OmegaConf.load(args.model_config)
    data_cfg = OmegaConf.load(args.data_config)
    OmegaConf.resolve(trainer_cfg)
    OmegaConf.resolve(model_cfg)
    OmegaConf.resolve(data_cfg)
    properties: list = args.property

    # Save Config
    OmegaConf.save({
        'property': properties,
        'model_config': model_cfg,
        'trainer_config': trainer_cfg,
        'data_config': data_cfg,
    }, config_path)

    # Print Config
    logging.info(
            'Training Information\n' +
            'Argument\n' + '\n'.join([f'{arg}:\t{getattr(args,arg)}' for arg in vars(args)]) + '\n\n' +
            'Trainer Config\n' + OmegaConf.to_yaml(trainer_cfg) + '\n' +
            'Data Config\n' + OmegaConf.to_yaml(data_cfg)
    )

    trainer = Trainer(trainer_cfg, model_cfg, data_cfg, properties, checkpoint_dir)
    return trainer, args

def main() : 
    common.set_seed(0)
    trainer, args = setup_trainer()
    trainer.fit()

if __name__ == '__main__' :
    main()
