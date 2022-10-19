# BBAR: Building Block based molecular AugoRegressive model

Official github of *Molecular generative model via retrosynthetically prepared chemical building block assembly* by Seonghwan Seo, Jaechang Lim, Woo Youn Kim.

You will be able to access updated version in https://github.com/SeonghwanSeo/BBAR.git.

## Table of Contents

- [Environment](#environment)
- [Data](#data)
  - [Data Structure](#data-structure)
  - [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Generation](#generation)

## Environment

- python=3.9
- [Pandas](https://pandas.pydata.org/)=1.5.0
- [PyTorch]((https://pytorch.org/))=1.12
- [RDKit](https://www.rdkit.org/docs/Install.html)=2022.3.5
- [PyTDC](https://tdcommons.ai) (Optional)
- Parmap=1.6.0

## Data

### Data Structure

#### Data Directory Structure

Move to `data/` directory. Initially, the structure of directory `data/` is as follows.

```bash
├── data/
    ├── data_preprocess.sh
    ├── preprocessing/
    ├── start_scaffold/
    ├── ZINC/
    │   ├── smiles/
    │   ├── all.txt 		(source data)
    │   ├── get_metadata.py
    │   ├── library.csv
    │   ├── library_map.csv
    │   ├── train.txt 	(https://github.com/wengong-jin/icml18-jtnn/tree/master/data/zinc/train.txt)
    │   ├── valid.txt 	(https://github.com/wengong-jin/icml18-jtnn/tree/master/data/zinc/valid.txt)
    │   └── test.txt 		(https://github.com/wengong-jin/icml18-jtnn/tree/master/data/zinc/test.txt)
    ├── generalization/
    └── 7l13_docking/ 	(Smina calculation result. (ligands: ZINC, receptor: 7L13))
        ├── smiles/
        ├── library.csv	(Same to ZINC/library.csv)
        ├── library_map.csv
        └── property.db	(metadata)
```

- `data_preprocess.sh`, `preprocessing/`: Scripts for preprocessing of data. Unless you must change default setting, you can use the given shell script, `data_preprocess.sh`.
- `start_scaffold/` is for our molecular sampling.
- `ZINC/`, `7l13_docking/`, `generalization/`: Dataset which is used in our research.

#### Prepare your own dataset

To use a given preprocessing script, you need to format your own dataset as follows.

```
MolID,SMILES,Property1,Property2,...
id1,c1ccccc1,10.25,32.21,...
id2,C1CCCC1,35.1,251.2,...
...
```

- SMILES must be RDKit-readable.
- If you want to train a single molecule set with different properties, you don't have to configure datasets separately for each property. You need to configure just one dataset file which contains all of property information. For example, `molport/property.db` contains information about `MolWt`, `LogP`, `TPSA` and you can train the model with a property pair `[MolWt, MolLogP]`.

After constructing your own dataset to the given format, make directory in `data/` and put your dataset with the name `property.db`.

### Preprocessing
#### Preprocessing (ZINC)

First, you need to create metadata. Go to `data/ZINC` and run `python get_metadata.py`. For 7L13 docking dataset, I already uploaded a metadata in github.

```shell
cd data/ZINC
python get_metadata.py
# property.db (metadata) is created from all.txt
```

And then, just run the script `data_preprocess.sh`

```shell
cd ../
./data_preprocess.sh ./ZINC/ <cpus>
# Create train.csv, val.csv, test.csv from ./ZINC/smiles/ and ./ZINC/library.csv

# For 7l13_docking
./data_preprocess.sh ./7l13_docking/ <cpus>
# Create train.csv, val.csv, test.csv from ./7l13_docking/smiles/ and ./7l13_docking/library.csv
```

After preprocessing step, the structure of directory `data/` is as follows. Our training/sampling code is based on following directory structure, so renaming the file is not recommended.

```bash
├── data/
    ├── ZINC/
    │   ├── ...
    │   ├── property.db
    │   ├── train.csv
    │   ├── train_weight.npy
    │   ├── val.csv
    │   └── test.csv
    ├── ...
```

#### Preprocessing (Own Data)

If you want to use your own data, follow below procedure. You need to put `property.db` in `data/<NEW-DIR>`.

There are two script: for partitioning dataset and creating library.

 There is a script for splitting dataset. (source file is `property.db`).

```shell
python preprocessing/split_data.py <NEW-DIR> --train_ratio <train-ratio> --val_ratio <val_ratio>
python preprocessing/get_library.py <NEW-DIR> --cpus <cpus>
./data_preprocess.sh <NEW-DIR> <cpus>
```

```
├── data/
    ├── <NEW-DIR>/
    │   ├── property.db (Source File)
    │   ├── smiles
    │   │   ├── train_smiles.csv
    │   │   ├── val_smiles.csv
    │   │   └── test_smiles.csv
    │   ├── library.csv
    │   ├── library_map.csv
    │   ├── train.csv
    │   ├── train_weight.npy
    │   ├── val.csv
    │   └── test.csv
    ├── ...
```



## Model Training

```shell
python train.py -h
```

Move to the root directory. Our training script reads config files in `./config/`, you can handle them by modifying or creating config files.

Training Script Format Example

```shell
python train.py \
    name <exp-name> \
    exp_dir <exp-dir> \
    property <property1> <property2> ... \
    trainer_config <train-config-path> \
    model_config <model-config-path> \
    data_config <data-config-path>
```

- `name`, `exp_dir`: The model output is saved at `<exp-dir>/<name>/`. Default setting of `exp_dir` is `result/`
- `property`: List of interesting properties
- `trainer_config`: Config file for training. Default setting is `./config/trainer.yaml`
- `model_config`: Config file for model. Default setting is `./config/model.yaml`
- `data_config`: Config file for data. There are some settings we used in `./config/data/`

Example running script.

```shell
python train.py \
    name 'logp-tpsa' \
    exp_dir 'result/ZINC' \
    property logp tpsa \
    data_config './config/data/zinc.yaml'
```

Yaml File Example

- data_config (`./config/data/zinc.yaml`)

```yaml
data_dir: ./data/ZINC
property_path: ${data_dir}/property.db
library_path: ${data_dir}/library.csv
train_data_path: ${data_dir}/train.csv
train_weight_path: ${data_dir}/train_weight.npy
val_data_path: ${data_dir}/val.csv
train_max_atoms: 40
val_max_atoms: 40
```

- trainer_config

```yaml
# Training Environment
gpus: 1
num_workers: 4

# Hyperparameter for Model Training
lr: 0.001
train_batch_size: 128
val_batch_size: 256

# Hyperparameter for Negative Sampling
num_negative_samples: 10
alpha: 0.75

# unit: step (batch)
max_step: 500000
log_interval: 5000
val_interval: 10000
save_interval: 10000
```



## Generation

```shell
python sample.py -h
```

Example running script.

```shell
# Non scaffold-based generation.
python sample.py \
    --generator_config './config/generator/logp_tpsa.yaml' \
    --o './result_sample/logp\=4-tpsa\=60.smi' \
    --num_samples 100 \
    --logp 4 --tpsa 60 	# generator config specific parameters. (No help message (python sample.py -h))

# Scaffold-based generation. (Single Scaffold)
python sample.py \
    --generator_config './config/generator/no_condition.yaml' \
    --scaffold 'Cc1ccccc1'
    --o './result_sample/no_condition.smi' \
    --num_samples 100
    
# Scaffold-based generation. (Multi Scaffold)
python sample.py \
    --generator_config './config/generator/logp.yaml' \
    --scaffold './data/start_scaffold/start100.smi' \
    --o './result_sample/logp\=6.smi' \
    --num_samples 100 \
    --logp 6
```

Yaml File Example

- generator config (`./config/generator/logp_tpsa.yaml`)
```yaml
model_path: './result/ZINC/logp-tpsa/checkpoint/best.tar'
library_path: './data/ZINC/library.csv'

# Below is library built-in model file path. (Model Parameter + SMILES and Latent Vectors for fragments in library.)
# During generation, model vectorizes the fragments in library.
# You can skip this process by saving all of them: model parameter and library informations.
# I called it `library built-in model`
# If below is not `null`, generator save or load library built-in model.
# If the built-in model file exists, upper two parameters (`model_path`, `library_path`) are not needed.
library_builtin_model_path: './builtin_model/zinc_logp-tpsa' # (optional)

# Required
n_library_sample: 2000
alpha: 0.75
max_iteration: 10
idx_masking: True
compose_force: False
```


