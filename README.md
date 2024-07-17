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

## Training Workflow

We have a workflow implemented in Snakemake (`Snakefile`), which takes as input in `config.yaml` names of models (see *Neural Network Model*) and datasets (see *Training Data*) and trains and evaluates the given models on all given datasets.

The input data is expected to be located in a `data` folder in the root directory of this repo.
Two lists `train_data` and `test_data` containing names of datasets need to be specified, so that the *i*th dataset in `train_data` is the training data for a model that is then tested on the *i*th dataset of `test_data`.
Note that we use nicknames for our datasets in `config.yaml`.
We need to define the paths to the actual datasets for each nickname in `dpvtex/dpvt_data.py`.
[TODO: explain how to generate this data, when those files are merged in.]
The data shall be made of dictionaries where trees are keys and their values are lists that assign `0` or `1` to edges in the tree, ordered by pre-order traversal, where `0` means this edge is in a Maximum Parsimony tree and `1` indicates that it is not.

To execute the workflow, run `snakemake -c[num_cores]` in the directory `train`, where `[num_cores]` should be replaced with the number of cores you want to use.
Alternatively, run `snakemake --snakefile train/Snakefile -c[num_cores]` in the root directory, or from any directory with the `--snakefile` path argument replaced as appropriate.


## Logging training

To view training logs, run `tensorboard --logdir .` and direct your browser to `http://localhost:6006/`.
The tensorboard additionally shows ROC curves for the performance of classification on the test set.


## File structure of this repo

- `train`: contains `Snakefile` and `config.yaml`, in which models and datasets for training are specified.
- `dpvtex`: contains `dpvt_data.py`, which implements functions to get datasets for a given nickname and `dpvt_zoo.py`, which creates models for a given nickname. These nicknames are provided to the `Snakefile` in `config.yaml`.