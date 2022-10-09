import argparse
import configargparse
from typing import List, Tuple

class Train_ArgParser(configargparse.ArgParser) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)

        # Required Parameters
        required_args = self.add_argument_group('train information')
        required_args.add_argument('--name', type=str, help='job name', required=True)
        required_args.add_argument('--exp_dir', type=str, help='path of experiment directory', default='./result/')
        required_args.add_argument('-p', '--property', type=str, nargs='+', help='property list')

        # Modules
        module_args = self.add_argument_group('module config(required)')
        module_args.add_argument('--trainer_config', type=str, default='./config/trainer.yaml')
        module_args.add_argument('--model_config', type=str, default='./config/model.yaml')
        module_args.add_argument('--data_config', type=str, default='./config/data.yaml')

        # Configuration Files (Optional)
        cfg_args = self.add_argument_group('config (optional)')
        cfg_args.add_argument('-c', '--config', is_config_file=True, type=str)

class Generation_ArgParser(configargparse.ArgParser) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)
        self.formatter_class=argparse.ArgumentDefaultsHelpFormatter

        # Required Parameters
        required_args = self.add_argument_group('required')
        required_args.add_argument('-g', '--generator_config', type=str, default='./config/generator.yaml',
                                                        help='generator config file')

        # Scaffold-based Generation
        scaf_args = self.add_argument_group('scaffold-based generation')
        scaf_args.add_argument('-s', '--scaffold', type=str, default=None, help='scaffold SMILES')
        scaf_args.add_argument('-S', '--scaffold_file', type=str, default=None, help='scaffold SMILES path')

        # Optional Parameters
        opt_args = self.add_argument_group('optional')
        opt_args.add_argument('-o', '--output_path', type=str, help='output file name')
        opt_args.add_argument('--seed', type=int, help='explicit random seed')
        opt_args.add_argument('--num_samples', type=int, help='number of generation', default=1)
        opt_args.add_argument('--verbose', action='store_true', help='print generating message')
        opt_args.add_argument('-q', action='store_true', help='no print sampling script message')

        # Configuration Files (Optional)
        cfg_args = self.add_argument_group('config (optional)')
        cfg_args.add_argument('-c', '--config', is_config_file=True, type=str)
        
