# dpvt-experiments-1

This repo allows testing models from [dpvt](https://github.com/matsengrp/dpvt)
on datasets generated in various ways. This repo also contains the code for
generating this data.


## Installation
To run the code in this workflow, install the conda environment from
`environment.yaml` and pip install the package dpvtex:

```
conda env create -f environment.yaml
pip install -e .
```

We also need to install the `dpvt` package. Clone the repo
[dpvt](https://github.com/matsengrp/dpvt), move into the repo and pip install
`dpvt`:

```
pip install -e .
```


## Setup
Create the conda environment from file: `conda env create --file
environment.yml`. Install `dpvtex` as a python package with pip: `pip install -e
.`. 


## Training Workflow

We have a workflow implemented in Snakemake (`Snakefile`), which takes as input
in `config.yaml` names of models (see *Neural Network Model*), datasets (see
*Training Data*), and the device on which we want to train (e.g. cpu or gpu, see
*Device*), and trains and evaluates the given models on all given datasets.

The input data is expected to be located in a `data` folder in the root
directory of this repo. Two lists `train_data` and `test_data` containing names
of datasets need to be specified, so that the *i*th dataset in `train_data` is
the training data for a model that is then tested on the *i*th dataset of
`test_data`. Note that we use nicknames for our datasets in `config.yaml`. We
need to define the paths to the actual datasets for each nickname in
`dpvtex/dpvt_data.py`. The data shall be made of dictionaries where trees are
keys and their values are lists that assign `0` or `1` to edges in the tree,
ordered by pre-order traversal, where `0` means this edge is in a Maximum
Parsimony tree and `1` indicates that it is not.

To execute the workflow, run `snakemake -c[num_cores]` in the directory `train`,
where `[num_cores]` should be replaced with the number of cores you want to use.
Alternatively, run `snakemake --snakefile train/Snakefile -c[num_cores]` in the
root directory, or from any directory with the `--snakefile` path argument
replaced as appropriate.

We have a workflow implemented in Snakemake (`Snakefile`), which takes as input
in `config.yaml` names of models (see _Neural Network Model_) and datasets (see
_Training Data_) and trains and evaluates the given models on all given
datasets. The input data is expected to be located in a `data` folder in the
root directory of this repo. The data shall be made of dictionaries where trees
are keys and their values are lists that assign `0` or `1` to edges in the tree,
ordered by pre-order traversal, where `0` means this edge is in a Maximum
Parsimony tree and `1` indicates that it is not.

To execute the workflow, run `snakemake -c[num_cores]` in the directory `train`,
where `[num_cores]` should be replaced with the number of cores you want to use.
Alternatively, run `snakemake --snakefile train/Snakefile -c[num_cores]` in the
root directory, or from any directory with the `--snakefile` path argument
replaced as appropriate.

### Neural Network models

We have four different models:
- `TraverseNN`
- `TraverseAvgPooling`
- `TraverseMaxPooling`

Details about these models can be found in
[dpvt](https://github.com/matsengrp/dpvt)


### Training data

Training data can be generated either from empirical or simulated alignments
using `larch` to construct Maximum Parsimony trees (see
`dpvtex/larch/README.md`) or by generating perfect phylogenies (see
`dpvtex/perfect_phylogenies/README.md`).

Nicknames for the datasets and paths to those datasets must be provided in a
`dpvt_zoo.py`. We assume that each dataset is given by one file that contains a
pickled dictionary. The keys of this dictionary shall be trees and their values
lists of `0`s and `1`s indicating if an edge (indexed in pre-order) is present
in a MP tree or not. Trees are allowed to have varying lengths. The current
implementation reads such a dictionary and splits it into training, validation
when loading training data. The testing data is loaded separately. Training set
is used to train our models, validation set is used for hyperparameter
optimization and to assess overfitting, and the test set is used for evaluating
the trained models.

The default data structure for out training and testing data is
`TraversalDataset`, which creates tensors representing tree traversals when
loading the data. To use the `TreeDataset` data structure (see more details in
the `dpvt` repo), set the `device` in `config.yaml` to `cpu-tree-dataset`.


### Device

By default, we train on CPUs. If the device is changed to `gpu` or `cuda` in the
config file, we train on the GPU. A detailed explanation of this can be found in
[dpvt](https://github.com/matsengrp/dpvt).


## Logging training

To view training logs, run `tensorboard --logdir .` and direct your browser to
`http://localhost:6006/`. The tensorboard additionally shows ROC curves for the
performance of classification on the test set.


## File structure of this repo

- `train`: contains `Snakefile` and `config.yaml`, in which models and datasets
  for training are specified.
- `dpvtex`: contains `dpvt_data.py`, which implements functions to get datasets
  for a given nickname and `dpvt_zoo.py`, which creates models for a given
  nickname. The mapping from nicknames to file paths is provided in
  `dataset_dict.json` and nicknames for datasets are given to the `Snakefile` in
  `config.yaml`.

   Also contains directories `perfect_phylogenies` and `larch`, which provide
  code for creating datasets for training and testing dpvt models (See _Training
  Data_).


## Training Data

### Generating Perfect Phylogenies

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

### Perturbing the phylogenies

Perturbing trees is handled by `perfect_phylogenies/perturb_phylogeny.py`. See
`perfect_phylogenies/examples/perturb_random_perfect_phylogenies.py` for an
example of generating random perfect phylogenies and perturbing them to obtain a
similar phylogeny, but with worse parsimony score.

To generate data for training the neural network, see
`perfect_phylogenies/examples/make_datasets.py`.

### Larch

Larch is a program that can be used to infer a collection of Maximum Parsimony
trees for a given alignment. We set up a pipeline that uses larch to create such
trees and then perturbs them to create training and testing sets for dpvt. We
describe how to do this in more detail in this
[README.md](dpvtex/larch/README.md)