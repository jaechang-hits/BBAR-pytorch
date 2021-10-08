import pandas as pd
import numpy as np
from collections import Counter
import argparse
import os

def main(args) :
    molecules_file = os.path.join(args.run_directory, 'train.csv')
    smis = pd.read_csv(molecules_file, usecols = ['SMILES'])['SMILES'].tolist()
    counter = Counter(smis)

    freq = np.zeros(len(smis))
    for i, smi in enumerate(smis) :
        freq[i] = counter[smi]
    weight = 1. / freq

    weight_file = os.path.join(args.run_directory, args.output)
    np.save(weight_file, weight)

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('run_directory', type=str)
    parser.add_argument('--dataset', type=str, default='train.csv')
    parser.add_argument('--output', type=str, default='train_weight.npy')
    args = parser.parse_args()
    main(args)
