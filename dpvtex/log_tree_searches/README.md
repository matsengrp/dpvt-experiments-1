# README

This directory contains code to run a Maximum Parsimony tree search for a set of input alignments, saves intermediate trees during this tree search, runs larch-usher to get a collection of maximum parsimony trees, and extracts the intermediate trees and MP edge labelling for them in the correct format to be used as dpvt testing input.

The pipeline can easily be executed by running the bash script `run_tree_searches.sh`, providing as command line arguments: (i) a path to the directory containing all input alignments, (ii) a path to a build of `larch-usher`, (iii) a path to the directory in which to store the output, and (iv) a path to a json file containing nicknames for training/testing data of dpvt:

```bash
./run_tree_searches.sh <path_to_alignment_dir> <path_to_larch_usher> <path_to_output_dir> <path_to_nickname_json?
```

## Prerequisites

To run the entire pipeline, the `dpvtex` package from the base directory of this repo needs to be installed.
Also needed is the software `R` and its packages `optparse` and `remotes`.
Within the R script, a slightly modified version of the `phangorn` package (version 2.12.1) is installed from [this](https://github.com/lenacoll/phangorn/tree/log-mp-search-trees) github repo, which allows logging trees along a maximum parsimony tree search.  

Additionally, the installation of the `larch-usher` software is required.
An installation guide can be found in their [github repo](https://github.com/matsengrp/larch).


## Details

The scripts that are executed in this bash script are (in this order):

### 1. Clean_data

The `clean_data` script from `dpvtex.larch.scripts` is called to remove all sites containing gaps, ambiguous charaters, or uninformative sites.
Additionally, it ensures that the alignment is saved as FASTA, even if a NEXUS file is given as input.

### 2. MP tree search

The R script `log_mp_tree_search.R` is using the modified version of the *phangorn* library that allows logging intermediate trees on the tree search to an optimal tree.
This modified version can be found on [github](https://github.com/lenacoll/phangorn/tree/log-mp-search-trees).
These trees are saved in the same directory as the input alignment, replacing the suffix `.fasta` by `_log.trees`.

### 3. Larch-usher

We run `larch-usher` to generate a collection of all maximum parsimony trees.
This requires one pre-processing step of the alignment to get the correct input files for larch-usher and then running larch-usher.
With the resulting collection of MP trees we can ensure that we label the edges of trees found in the MP tree search correctly as MP or non-MP edges in the next step.

### 4. Label edges in trees found in MP tree search (step 2.)

We use the collection of MP trees found in step 3 to label all edges in the trees in step 2 as MP or non-MP edges.
The resulting trees and edge labels are saved in the format required by dpvt and nicknames are added to the provided nickname file.