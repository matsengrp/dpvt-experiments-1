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


## Training Data

### Generating Perfect Phylogenies
Python classes for creating perfect phylogenies, for a given tree topology, are in 
`generate_data/perfect_phylogeny.py` and `generate_data/perfect_phylogeny.py`. 

Call the `make_phylogenies` method of a `PerfectPhylogeny` instance to generate all 
perfect phylogenies (with a certain minimality condition) for a topology. This class
requires a large number of computations at initialization, which can be very slow for a
topology with a moderate number of leaves (over 100). The minimality condition is 
explained in the documentation for the `MinimalCovers` class. Furthermore, the list of
perfect phylogenies contains a single representative of each set of perfect phylogenies
that are equivalent by permuting the order of sites (e.g., just one of
`((0[&&NHX:sequence=GA],1[&&NHX:sequence=GA])[&&NHX:sequence=GA],(2[&&NHX:sequence=AG],3[&&NHX:sequence=AG])[&&NHX:sequence=AG]);`
and
`((0[&&NHX:sequence=AG],1[&&NHX:sequence=AG])[&&NHX:sequence=AG],(2[&&NHX:sequence=GA],3[&&NHX:sequence=GA])[&&NHX:sequence=GA]);`).
The list of perfect phyologenies can be trimmed down further by specifying
`skip_perms=True`, so that the list contains a single representative of each set of
perfect phylogenies that are equivalent by permuting the bases assigned in each site
(e.g., just one of
`((0:1[&&NHX:sequence=G],1:1[&&NHX:sequence=G])1:1[&&NHX:sequence=G],2:1[&&NHX:sequence=A]);`
and
`((0:1[&&NHX:sequence=C],1:1[&&NHX:sequence=C])1:1[&&NHX:sequence=C],2:1[&&NHX:sequence=G]);`).

Call the `make_random_perfect_phylogeny` method of a `RandomPerfectPhylogeny` instance
to generate a random perfect phylogeny. This class avoids the computations at 
initialization of `PerfectPhylogeny`, but only samples with replacement. Runtime
is non-linear in the number of leaves of the topology, but it is fast enough to
generate a large number of perfect phylogenies on topologies with a few hundred leaves. 
For example, generating a random perfect phylogeny on 100 leaves takes about 0.02 
seconds, 500 leaves takes about 2.2 seconds, while 1000 leaves takes 17.2 seconds.
Unlike `PerfectPhylogeny`, this class draws from all perfect phylogenies meeting the 
minimality condition regardless of of site order, but specifying 
`no_permutations=True` works the same as `skip_perms=True`.

In short, use `PerfectPhylogeny` to get all pefect phylogenies and `RandomPefectPhylogeny` to get a
single perfect phylogeny at random.




### Perturbing the phylogenies
Perturbing trees is handled by `perfect_phylogenies/perturb_phylogeny.py`. See
`perfect_phylogenies/examples/perturb_random_perfect_phylogenies.py` for an
example of generating random perfect phylogenies and perturbing them to obtain a similar 
phylogeny, but with worse parsimony score.

To generate data for training the neural network, see `perfect_phylogenies/examples/make_datasets.py`.
This repo allows testing models from [dpvt](https://github.com/matsengrp/dpvt) on datasets generated in various ways.
The code for generating this data is also contained in this repo.

## Training Workflow

We have a workflow implemented in Snakemake (`Snakefile`), which takes as input in `config.yaml` names of models (see *Neural Network Model*) and datasets (see *Training Data*) and trains and evaluates the given models on all given datasets.
The input data is expected to be located in a `data` folder in the root directory of this repo.
The data shall be made of dictionaries where trees are keys and their values are lists that assign `0` or `1` to edges in the tree, ordered by pre-order traversal, where `0` means this edge is in a Maximum Parsimony tree and `1` indicates that it is not.

To execute the workflow, run `snakemake -c[num_cores]` in the directory `train`, where `[num_cores]` should be replaced with the number of cores you want to use.
Alternatively, run `snakemake --snakefile train/Snakefile -c[num_cores]` in the root directory, or from any directory with the `--snakefile` path argument replaced as appropriate.


## File structure of this repo

- `train`: contains `Snakefile` and `config.yaml`, in which models and datasets for training are specified.
- `dpvtex`: contains `dpvt_data.py`, which implements functions to get datasets for a given nickname and `dpvt_zoo.py`, which creates models for a given nickname. These nicknames are provided to the `Snakefile` in `config.yaml`.
