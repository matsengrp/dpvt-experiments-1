# dpvt-experiments-1

This repo provides a workflow for training and testing
[dpvt](https://github.com/matsengrp/dpvt) models (see *Standard Training
Workflow*). We also provide different ways of generating training and
testing data from simulated or empirical alignments (*Training/Testing Data
Generation*).


## Installation

To run the code of this repo, you will need to creat a pixi environment. To
install [pixi](https://pixi.sh/latest/installation/) with curl (on Linux or
MacOS), run:

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

Next, clone the [dpvt](https://github.com/matsengrp/dpvt) repo into the same
parent directory as this repo. If you clone it to a different location, you need
to update the path to the repo in `pixi.toml`.

You are now ready to install the pixi environment *dpvt-experiments*, which
includes the dpvt package from the dpvt repo and the dpvtex package from this
repo. To install and then activate the environment, run:

```bash
pixi install
pixi shell
```


## Standard Training Workflow

The standard workflow to train and test the dpvt models is implemented as a
Snakemake workflow (`Snakefile`), which requires the `config.yaml` file to
specify models, data, and setting to use for training and testing. These are:
-  `models`: Names of models to use (see *Neural Network Model*). One or more
    of: `[ "TraverseNN", "TraverseMaxPooling", "TraverseAvgPooling",
    "BaselineReversion"]`
- `train_data`: List of nicknames of training data sets (see _Training Data_). A
  dictionary containing those nicknames as keys and paths to the files
  containing the data must be provided (see `data_nicknames_path`) as json file
- `test_data`: List of nicknames of testing data sets, just like training data
- `device`: Device you want to use for training/testing (see _Device_). E.g.
  `"gpu"`, "`cuda"`, `"cpu"`. Special case is `cpu-tree-dataset`, which uses a
  different data structure for the trees inside the models, which is less
  efficient than the default
- `timestamp`: Used for saving output: `{output_dir}/run.{timestamp}/` (see
  `output_dir`)
- `use_cross_datasets`: If True, every trained model is tested on every test
  data set. If False, model trained on train_data set *i* is only tested on
  test_data set *i* (*i* being the index in the list of train_data and test_data
  provided in this config)
- `output_dir`: Name of output directory. Contains all output computed during
  training and testing, including checkpoints
- `data_nicknames_path`: Relative path to json file containing a dictionary with
  data nicknames as keys and paths to files as values. This file needs to
  provide paths to all train_data and test_data nicknames provided in this
  config
- `use_hyperparameter_optimize`: If True, runs hyperparameter optimization with
  Optuna, unless this has been done already and hyperparameters can be loaded
  from log file (in
  `{output_dir}/run.{timestamp}/checkpoint_logs/optimize_hyperparameters/{model}-{train_data}-Param{i}.json`).
  If False, uses hyperparameters defined at the bottom of this config file
- `n_hyperparameter_trials`: Number of hyperparameter trials to be run by Optuna
- `hyperparameters`: Default hyperparameters to be used

More details to these inputs can be found in the following subsections.
To execute the workflow, move into the directory `train` and run:

```bash
snakemake -c[num_cores]
```

`[num_cores]` should be replaced with the number of cores you want to use.
Alternatively, run `snakemake --snakefile path/to/Snakefile -c[num_cores]` by
providing the path to the Snakefile.


### Neural Network models

We have four different models:

-   `TraverseNN`
-   `TraverseAvgPooling`
-   `TraverseMaxPooling`
-   `BaselineReversion`

Details about these models can be found in
[dpvt](https://github.com/matsengrp/dpvt).


### Training/testing data

Training and testing data can be generated either from empirical or simulated
alignments using `larch` to construct Maximum Parsimony trees (see
`dpvtex/larch/README.md`) or by generating perfect phylogenies (see
`dpvtex/perfect_phylogenies/README.md`).

The input data must be saved in dictionaries where trees in ete3 format are keys
and their values are lists that assign `0` or `1` to all edges in the tree
(including pendant edges), ordered by pre-order traversal, where `0` means this
edge is in a Maximum Parsimony tree and `1` indicates that it is not.

> Note: Avoid using `-` in nicknames for models or datasets, as this might
> result in issues with Snakemake

Nicknames for datasets and paths to those datasets must be provided in the json
file that is provided as `data_nicknames_path` in the config.

#### Data format
We assume that each dataset is provided by one file that contains a pickled
dictionary. The keys of this dictionary shall be trees and their values lists of
`0`s and `1`s indicating if an edge (indexed in pre-order) is present in a MP
tree or not, respectively. The trees contained in one dataset do not need to
have the same leaf set and can vary in the number of leaves. Training and
testing data are loaded separately and need to be in separate files. The
training set is split into training and validation set. The training set is used
to train our models, validation set is used for hyperparameter optimization and
to assess overfitting, and the test set is used for evaluating the trained
models.

The default data structure for out training and testing data is
`TraversalDataset`, which creates tensors representing tree traversals when
loading the data. To use the `TreeDataset` data structure (see more details in
the `dpvt` repo), set the `device` in `config.yaml` to `cpu-tree-dataset`. The
`TreeDataset` cannot be used on a GPU. The only exception to the default usage
of `TraversalDataset` is the Baseline model `BaselineReversion`, which can only
run on the CPU as it only works with the `TreeDataset` data structure. Even if
one requests running on a GPU, this model will run on the CPU.

### Device

By default, we train on CPUs. If the device is changed to `gpu` or `cuda` in the
config file, we train on the GPU. A detailed explanation of this can be found in
[dpvt](https://github.com/matsengrp/dpvt). Note that the `BaselineReversion`
model is required to run on a cpu, so if you want to use it, make sure to
provide `cpu` as device in the config.

## Logging training

To view training logs, run `tensorboard --logdir .` and direct your browser to
`http://localhost:6006/`. The tensorboard additionally shows ROC curves for the
performance of classification on the test set.

## File structure of this repo

-   `train`: contains `Snakefile` and `config.yaml`, in which models and
    datasets for training are specified.
-   `dpvtex`: contains `dpvt_data.py`, which implements functions to get
    datasets for a given nickname and `dpvt_zoo.py`, which creates models for a
    given nickname. The mapping from nicknames to file paths is provided in
    `dataset_dict.json` and nicknames for datasets are given to the `Snakefile`
    in `config.yaml`.

    Also contains directories `perfect_phylogenies` and `larch`, which provide
    code for creating datasets for training and testing dpvt models (See
    _Training Data_).

## Training/Testing Data Generation

Training and testing data can be generated from empirical or simulated
alignments by running the inference software `larch`, which generates maximum
parsimony trees, and then perturbing them to introduced non-MP edges.
This is our standard approach to generating data sets for `dpvt`.

A pipeline for generating training and testing dataset with this methods is
provided in `dpvtex/larch`, which also contains a
[README](dpvtex/larch/README.md) with details on how to use it.


### Alternative Data Generation

#### Generating Perfect Phylogenies

Python classes for creating perfect phylogenies, for a given tree topology, are
in `generate_data/perfect_phylogeny.py` and
`generate_data/perfect_phylogeny.py`.

Call the `make_phylogenies` method of a `PerfectPhylogeny` instance to generate
all perfect phylogenies (with a certain minimality condition) for a topology.
This class requires a large number of computations at initialization, which can
be very slow for a topology with a moderate number of leaves (over 100). The
minimality condition is explained in the documentation for the `MinimalCovers`
class. Furthermore, the list of perfect phylogenies contains a single
representative of each set of perfect phylogenies that are equivalent by
permuting the order of sites (e.g., just one of
`((0[&&NHX:sequence=GA],1[&&NHX:sequence=GA])[&&NHX:sequence=GA],(2[&&NHX:sequence=AG],3[&&NHX:sequence=AG])[&&NHX:sequence=AG]);`
and
`((0[&&NHX:sequence=AG],1[&&NHX:sequence=AG])[&&NHX:sequence=AG],(2[&&NHX:sequence=GA],3[&&NHX:sequence=GA])[&&NHX:sequence=GA]);`).
The list of perfect phyologenies can be trimmed down further by specifying
`skip_perms=True`, so that the list contains a single representative of each set
of perfect phylogenies that are equivalent by permuting the bases assigned in
each site (e.g., just one of
`((0:1[&&NHX:sequence=G],1:1[&&NHX:sequence=G])1:1[&&NHX:sequence=G],2:1[&&NHX:sequence=A]);`
and
`((0:1[&&NHX:sequence=C],1:1[&&NHX:sequence=C])1:1[&&NHX:sequence=C],2:1[&&NHX:sequence=G]);`).

Call the `make_random_perfect_phylogeny` method of a `RandomPerfectPhylogeny`
instance to generate a random perfect phylogeny. This class avoids the
computations at initialization of `PerfectPhylogeny`, but only samples with
replacement. Runtime is non-linear in the number of leaves of the topology, but
it is fast enough to generate a large number of perfect phylogenies on
topologies with a few hundred leaves. For example, generating a random perfect
phylogeny on 100 leaves takes about 0.02 seconds, 500 leaves takes about 2.2
seconds, while 1000 leaves takes 17.2 seconds. Unlike `PerfectPhylogeny`, this
class draws from all perfect phylogenies meeting the minimality condition
regardless of of site order, but specifying `no_permutations=True` works the
same as `skip_perms=True`.

In short, use `PerfectPhylogeny` to get all pefect phylogenies and
`RandomPefectPhylogeny` to get a single perfect phylogeny at random.

#### Perturbing the phylogenies

Perturbing trees is handled by `perfect_phylogenies/perturb_phylogeny.py`. See
`perfect_phylogenies/examples/perturb_random_perfect_phylogenies.py` for an
example of generating random perfect phylogenies and perturbing them to obtain a
similar phylogeny, but with worse parsimony score.

To generate data for training the neural network, see
`perfect_phylogenies/examples/make_datasets.py`.
