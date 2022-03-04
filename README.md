# Fragment-based Molecular Generative Model

Official github of *Improvement of generating molecules with rare targetproperty values using a molecular fragment-based deepgenerative model without restriction in chemical space* by Seonghwan Seo, Jaechang Lim, Woo Youn Kim.

#### Pytorch-lightning version
In this repository, there will be no updates except documents. Instead, we provide pytorch-lightning version, repository for development. If you would like to request or suggest additional functions, please use following link.
- https://github.com/SeonghwanSeo/FMGM.git

## Table of Contents

- [Environment](#environment)
- [Data](#data)
  - [Data Structure](#data-structure)
  - [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Generating](#generate)
- Research

## Environment

- python>=3.7
- [Pandas](https://pandas.pydata.org/)=1.2.2
- [Hydra](https://hydra.cc/)=1.0.6
- [PyTorch]((https://pytorch.org/))=1.7.1
- [RDKit](https://www.rdkit.org/docs/Install.html)>=2020.09.3

## Data

### Data Structure

#### Data Directory Structure

Move to `data/` directory. Initially, the structure of directory `data/` is as follows.

```bash
├── data/
    ├── data_preprocess.sh
    ├── preprocessing/
    ├── start_list/
    ├── molport/
    │   └── property.db
    └── docking/
        └── property.db
```

- `data_preprocess.sh`, `preprocessing/`: Scripts for preprocessing of data. Unless you must change default setting, you can use the given shell script, `data_preprocess.sh`.
- `start_list/` is for our molecular sampling.
- `molport/`, `docking/`: Dataset which is used in our research.

#### Prepare your own dataset

To use a given preprocessing script, you need to format your own dataset as follows.

```
MolID,SMILES,Property1,Property2,...
id1,c1ccccc1,10.25,32.21,...
id2,C1CCCC1,35.1,251.2,...
...
```

- SMILES must be RDKit-readable.
- If you want to train a single molecule set with different properties, you don't have to configure datasets separately for each property. You need to configure just one dataset file which contains all of property information. For example, `molport/property.db` contains information about `ExactMolWt`, `MolLogP`, `TPSA` and we can train the model with a property pair `[ExactMolWt, MolLogP]`.

After constructing your own dataset to the given format, make directory in `data/` and put your dataset with the name `property.db`.

### Preprocessing

First default setting is splitting SMILES set by train:validation:test = 0.75:0.15:0.10. If you do not wish this setting, please follow the steps below. If you want to use your own dataset, put your data directory name instead of `molport/`.

```shell
cd data/
python preprocessing/get_library.py molport/ --cpus <cpus>
python preprocessing/split_data.py molport/ --train_ratio <train-ratio> --val_ratio <val_ratio>
python preprocessing/get_datapoint.py molport/ --cpus <cpus> --mol train_smiles.csv --output train.csv
python preprocessing/get_frag1_freq.py molport/
python preprocessing/get_datapoint.py molport/ --cpus <cpus> --mol val_smiles.csv --output val.csv
python preprocessing/get_datapoint.py molport/ --cpus <cpus> --mol test_smiles.csv --output test.csv
```

Else, you just run the script `data_preprocess.sh`.

```shell
./data_preprocess.sh molport/ <cpus>
```

After preprocessing step, the structure of directory `data/` is as follows. Our training/sampling code is based on following directory structure, so renaming the file is not recommended.

```bash
├── data/
    ├── molport/
    │   ├── library.csv
    │   ├── library_map.csv
    │   ├── property.db
    │   ├── smiles
    │   │   ├── train_smiles.csv
    │   │   ├── val_smiles.csv
    │   │   └── test_smiles.csv
    │   ├── train.csv
    │   ├── train_weight.npy
    │   ├── val.csv
    │   └── test.csv
    ├── ...
```



## Model Training

Move to the root directory. Our script handles arguments and hyperparameters with Hydra module. The following commands are the minimum command for implementing our paper, and you can handle more arguments and hyperparameters with [hydra override](https://hydra.cc/docs/intro#basic-example).

You can handle model hyperparameter with hydra module. The default setting is the hyperparameter used in our research.

Training Script Format Example (See `conf/train.yaml`)

```shell
python train.py \
    name=<exp-name> \
    exp_dir=<exp-dir> \
    timezone=<timezone> \
    data_dir=<data-dir-name> \
    condition.descriptors='[PropertyList]' \
    train.num_workers=<num-workers> \
    train.max_epoch=<max_epochs> \
    ns_trainer.n_sample=<num-negative-sample> \
    ns_trainer.alpha=<alpha> \
    data.train.batch_size=<batch_size> \
    data.train.sampler.n_sample=<num_train_data> \
    data.train.max_atoms=<max_atoms> \
    data.val.batch_size=<batch_size> \
    data.val.max_atoms=<max_atoms>
```

- `name`, `exp_dir`: The model output is saved at `<exp-dir>/<name>/`. Default setting of `exp_dir` is `result/`
- `timezine`: Argument just for logging. Enter your timezone. Default value is None.(System Time)
- `data_dir`: Since the directory structure is fixed, you just need to use the data directory. If you change the data directory structure, you should add argument for each file according to the config file `conf/train.yaml`. Default value is `data/molport/`
- `condition.descriptors`: For control the target properties.
  - condition.descriptors='[]'          # Default, Unconditional
  - condition.descriptors='[MolLogP]'
  - condition.descriptors='[MolLogP,TPSA]'
- `ns_trainer.n_sample`: Number of negative samples. Default value is 10.
- `ns_trainer.alpha`: Power value of fragment frequency distribution. Default value is 0.75, which is commonly used in Word2Vec. *Mikolov, T. et al, (2013)*
- `data.train.sampler.n_sample`: The training step uses a balanced sampler. This parameter is according to `num_samples` of  `torch.utils.data.WeightedRandomSampler`. Default value is 4,000,000.
- `data.train.max_atoms`, `data.val.max_atoms`: For simple implementation, our model requires the maximum number of atoms. Enter the maximum number of atoms of molecules in the dataset. Default value is 50.

Example running script.

```shell
python train.py \
    name='logp-tpsa' \
    exp_dir='result/molport' \
    data_dir='data/molport' \
    condition.descriptors='[MolLogP,TPSA]' \
    train.num_workers=4 \
    train.max_epoch=10 \
    data.train.batch_size=128 \
    data.train.sampler.n_sample=4000000 \
    data.train.max_atoms=50 \
    data.val.batch_size=256 \
    data.val.max_atoms=50
```


## Generating

```shell
python sample.py --help
```

Example running script.

```shell
python sample.py \
    gpus=1 \
    name='logp\=4-tpsa\=60' \
    exp_dir='result/sample/molport/logp-tpsa' \
    data_dir='data/molport' \
    model_path='result/model/molport/logp-tpsa/save9.tar' \
    start_mol_path='data/start_list/start100.smi' \
    n_sample=100 \
    save_property=true \
    +condition.MolLogP=4 \
    +condition.TPSA=60
```


