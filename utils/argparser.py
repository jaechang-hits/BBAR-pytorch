import configargparse
from typing import List, Tuple

class Generation_ArgParser(configargparse.ArgParser) :
    def __init__(self, **kwargs) :
        super().__init__(**kwargs)

        # Required Parameters
        required_args = self.add_argument_group('required')
        required_args.add_argument('--generator_config', type=str, required=True)

        # Scaffold-based Generation
        scaf_args = self.add_argument_group('scaffold-based generation')
        scaf_args.add_argument('-s', '--scaffold', type=str, default=None, help='scaffold SMILES')

        # Output
        opt_args = self.add_argument_group('optional')
        opt_args.add_argument('-o', '--output_path', type=str, help='output file name')
        opt_args.add_argument('--seed', type=int, help='explicit random seed')
        opt_args.add_argument('--num_samples', type=int, help='number of generation')
        opt_args.add_argument('--verbose', action='store_true', help='print generating message')

        # Configuration Files (Optional)
        cfg_args = self.add_argument_group('config (optional)')
        cfg_args.add_argument('-c', '--config', is_config_file=True, type=str)
        
