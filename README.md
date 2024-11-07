# dpvt-experiments-1

This repo allows testing models from [dpvt](https://github.com/matsengrp/dpvt) on datasets generated in various ways.
This repo also contains the code for generating this data.

## Installation
To run the code in this workflow, install the conda environment from `environment.yaml` and pip install the package dpvtex:

```
conda env create -f environment.yaml
pip install -e .
```

We also need to install the `dpvt` package.
Clone the repo [dpvt](https://github.com/matsengrp/dpvt), move into the repo and pip install `dpvt`:

```
pip install -e .
```

## Setup
Create the conda environment from file: `conda env create --file environment.yml`.
Install `dpvtex` as a python package with pip: `pip install -e .`. 

## Training Workflow

We have a workflow implemented in Snakemake (`Snakefile`), which takes as input in `config.yaml` names of models (see *Neural Network Model*), datasets (see *Training Data*), and the device on which we want to train (e.g. cpu or gpu, see *Device*), and trains and evaluates the given models on all given datasets.

The input data is expected to be located in a `data` folder in the root directory of this repo.
Two lists `train_data` and `test_data` containing names of datasets need to be specified, so that the *i*th dataset in `train_data` is the training data for a model that is then tested on the *i*th dataset of `test_data`.
Note that we use nicknames for our datasets in `config.yaml`.
We need to define the paths to the actual datasets for each nickname in `dpvtex/dpvt_data.py`.
The data shall be made of dictionaries where trees are keys and their values are lists that assign `0` or `1` to edges in the tree, ordered by pre-order traversal, where `0` means this edge is in a Maximum Parsimony tree and `1` indicates that it is not.

To execute the workflow, run `snakemake -c[num_cores]` in the directory `train`, where `[num_cores]` should be replaced with the number of cores you want to use.
Alternatively, run `snakemake --snakefile train/Snakefile -c[num_cores]` in the root directory, or from any directory with the `--snakefile` path argument replaced as appropriate.


### Neural Network models

We have four different models:
- `TraverseNN`
- `TraverseAvgPooling`
- `TraverseMaxPooling`
- `TransformerEncoderTraversal`

Details about these models can be foung in [dpvt](https://github.com/matsengrp/dpvt)

### Training data

Training data can be generated either from empirical or simulated alignments using `larch` to construct Maximum Parsimony trees (see `dpvtex/larch/README.md`) or by generating perfect phylogenies (see `dpvtex/perfect_phylogenies/README.md`).

Nicknames for the datasets and paths to those datasets must be provided in a `dpvt_zoo.py`.
We assume that each dataset is given by one file that contains a pickled dictionary.
The keys of this dictionary shall be trees and their values lists of `0`s and `1`s indicating if an edge (indexed in pre-order) is present in a MP tree or not.
Trees are allowed to have varying lengths.
The current implementation reads such a dictionary and splits it into training, validation, and test set.
Training set is used to train our models, validation set is used for hyperparameter optimization and to assess overfitting, and the test set is used for evaluating the trained models.

### Device

By default, we train on CPUs.
If the device is changed to `gpu` or `cuda` in the config file, we use the TraversalDataset structure and convert trees to tensors representing traversals on the trees.
A detailed explanation of this can be found in [dpvt](https://github.com/matsengrp/dpvt).


## Logging training

To view training logs, run `tensorboard --logdir .` and direct your browser to `http://localhost:6006/`.
The tensorboard additionally shows ROC curves for the performance of classification on the test set.


## File structure of this repo

- `train`: contains `Snakefile` and `config.yaml`, in which models and datasets for training are specified.
- `dpvtex`: contains `dpvt_data.py`, which implements functions to get datasets for a given nickname and `dpvt_zoo.py`, which creates models for a given nickname. These nicknames are provided to the `Snakefile` in `config.yaml`.

