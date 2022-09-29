from rdkit import Chem

import random
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
    # Set Generator
    generator, args = setup_generator()

    # Set Output File
    if args.output_path not in [None, 'null'] :
        output_path = args.output_path
    else :
        output_path = '/dev/null'
    out_writer = open(output_path, 'w')

    # Load Scaffold
    if args.scaffold_file is not None :
        with open(args.scaffold_file) as f :
            scaffold_list = [l.strip() for l in f.readlines()]
    else :
        if args.scaffold is not None :
            scaffold_list = [args.scaffold]
        else :
            scaffold_list = [None]

    # Set Seed
    if args.seed is None :
        args.seed = random.randint(0, 1e6)
    print(f"Seed: {args.seed}")

    # Start
    global_st = time.time()
    global_success = 0
    for scaf_idx, scaffold_smi in enumerate(scaffold_list) :
        # Encoding Scaffold Molecule
        common.set_seed(args.seed)
        scaffold_mol = Chem.MolFromSmiles(scaffold_smi)
        print(f"[{scaf_idx+1}/{len(scaffold_list)}]")
        print(f"Scaffold: {scaffold_smi}")

        local_st = time.time()
        success = 0
        for i in range(1, args.num_samples + 1) :
            if not args.q :
                print(f"{i}th Generation...")
            try :
                generated_mol = generator.generate(scaffold_mol, args.verbose)
            except KeyboardInterrupt :
                raise KeyboardInterrupt
            except :
                generated_mol = None

            if generated_mol is None :
                if not args.q :
                    print("FAIL\n")
                continue
            else :
                smiles = Chem.MolToSmiles(generated_mol)
                if smiles is not None :
                    success += 1
                    out_writer.write(smiles+'\n')
                    if not args.q :
                        print(f"Finish\t{smiles}\n")
                else :
                    if not args.q :
                        print("FAIL\n")

        local_end = time.time() 
        time_cost = local_end - local_st 
        global_success += success
        print(f"Num Generated Mol: {success}") 
        print(f"Time Cost: {time_cost:.3f}, {time_cost/args.num_samples:.3f}\n")

    out_writer.close()
    global_end = time.time()
    time_cost = global_end - global_st
    print(f"Num Generated Mol: {global_success}") 
    print(f"Total Time Cost: {time_cost:.3f}, {time_cost/args.num_samples/len(scaffold_list):.3f}")

if __name__ == '__main__' :
    main()
