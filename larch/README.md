# README

This is a summary of how to disambiguate an input alignment and then run larch-usher on it to produce an hDAG.


## Conda environment

To run this code, we need to install a conda environment:

```bash
conda env create -f environment.yml
conda activate larch-data-gen
```


## Removing ambiguities

Before building a historydag with `larch-usher`, we clean input alignments by deleting all sites that either contain gaps or ambiguous characters.
We use [biopython](https://biopython.org/) to delete ambiguous characters and gaps.
To do this, run

```bash
python remove_ambiguities.py /path/to/input/fasta /path/to/disambiguated/output/fasta
```

We later require the fasta file on which we run `larch-usher` to be called `input.fasta`, so it might be a good idea to call the output file for `input.fasta`.


## Constructing an hDAG

You first need to install [larch](https://github.com/matsengrp/larch), building instructions can be found at this link.

Next, we need to set up the larch inputs.
Make sure you have activated the `larach-data-gen` environment for this.
First, we need to run the snakefile in `setup_larch_inputs` on our dataset.
We assume here that our alignment is named `input.fasta` and is located in the directory `/path/to/disambiguated/fasta`.
Then we can set up larch inputs by running:

```bash
cd setup_larch_inputs
snakemake --snakefile convert_fasta_to_larch_input.snakefile -d /path/to/disambiguated/fasta --cores <num_cores>
```

Note that we need to specify the number of cores `<num_cores>` here.

We are now ready to run larch-usher.
Therefore, navigate to the folder in which you built larch-usher and run larch-usher:

```bash
cd your-local-larch-dir/larch/build
./larch-usher -i  /path/to/disambiguated/fasta/output.pb -r /path/to/disambiguated/fasta/output.txt -v /path/to/disambiguated/fasta/output.vcf -o /path/to/output/protobuf -c <number of larch-usher iterations you want to run> -l /path/to/log/directory/
```


## Extracting hDAG trees as training data

We extract trees from the given hDAG to get ete3 trees for training our dpvt models.
This can be done by executing 

```bash
python extract_data_from_hdag.py /path/to/protobuf /path/to/dpvt/training/data
```

This script will read the hDAG in the protobuf in `/path/to/protobuf` and pickles trees and edge labels as training/testing/validation data as required by `dpvt` in `/path/to/dpvt/training/data`.
The following steps are executed by this script:
- read the (compact genome) hDAG
- trim the hDAG to only contain MP trees
- convert the hDAG to a sequence dag
- sample from the hDAG without replacement -- currently samples as many trees are in the hDAG
- perturbs the sampled trees by replacing a subtree with depth `tree_depth/3` by a random tree
- labels edges by whether they are MP edges are not
- pickle trees and edge labels and safe to `/path/to/dpvt/training/data`