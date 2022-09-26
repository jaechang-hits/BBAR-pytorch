from rdkit import Chem

import numpy as np
import logging
import time
from omegaconf import OmegaConf

from utils import common
from utils.argparser import Generation_ArgParser

from src.generator import MoleculeBuilder

def setup_generator() :
    # Parsing
    parser = Generation_ArgParser()
    args, remain_args = parser.parse_known_args()

    generator_cfg = OmegaConf.load(args.generator_config)
    generator = MoleculeBuilder(generator_cfg, None)

    # Second Parsing To Read Condition
    if len(generator.target_properties) > 0 :
        for property_name in generator.target_properties :
            parser.add_argument(f'--{property_name}', type = float, required=True)
    args = parser.parse_args()
    condition = {property_name: args.__dict__[property_name] for property_name in generator.target_properties}
    generator.setup(condition)

    return generator, args

def main() : 
    generator, args = setup_generator()

    if args.scaffold is not None :
        start_mol = Chem.MolFromSmiles(args.scaffold)
    else :
        start_mol = None

    if args.output_path not in [None, 'null'] :
        output_path = args.output_path
    else :
        output_path = '/dev/null'
    w = open(output_path, 'w')

    st = time.time()
    common.set_seed(args.seed)
    for i in range(args.num_samples) :
        print(f"{i}th Generation...")
        generated_mol = generator.generate(start_mol, args.verbose)
        if generated_mol is None :
            continue
        smiles = Chem.MolToSmiles(generated_mol)
        if smiles is not None :
            print(f"Finish\t{smiles}\n")
            w.write(smiles+'\n')
        else :
            print("FAIL\n")

    w.close()
    end = time.time()
    print(end-st, (end-st)/args.num_samples)

if __name__ == '__main__' :
    main()
