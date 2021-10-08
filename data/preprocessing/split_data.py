import random
import csv
import os
import argparse

random.seed(42)

def main(args) :
    molecule_file = os.path.join(args.run_directory, args.mol)
    with open(molecule_file) as f :
        lines = list(csv.reader(f))[1:]

    a = int(len(lines)*args.train_ratio)
    b = int(len(lines)*(args.train_ratio + args.val_ratio))

    output_dir = os.path.join(args.run_directory, 'smiles')
    os.system('mkdir -p ' + output_dir)
    if args.shuffle :
        random.shuffle(lines)

    with open(output_dir + '/train_smiles.csv', 'w') as w :
        w.write('MolID,SMILES\n')
        for l in lines[:a] :
            w.write(f'{l[0]},{l[1]}\n')
    with open(output_dir + '/val_smiles.csv', 'w') as w :
        w.write('MolID,SMILES\n')
        for l in lines[a:b] :
            w.write(f'{l[0]},{l[1]}\n')
    with open(output_dir + '/test_smiles.csv', 'w') as w :
        w.write('MolID,SMILES\n')
        for l in lines[b:] :
            w.write(f'{l[0]},{l[1]}\n')

if __name__ == '__main__' :
    parser = argparse.ArgumentParser()
    parser.add_argument('run_directory', type=str)
    parser.add_argument('--mol', type=str, default='property.db')
    parser.add_argument('--train_ratio', type=float, default=0.75)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--shuffle', dest='shuffle', action='store_true')
    parser.add_argument('--no-shuffle', dest='shuffle', action='store_false')
    parser.set_defaults(shuffle=True)
    args = parser.parse_args()
    main(args)
