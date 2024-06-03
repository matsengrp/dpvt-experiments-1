# README

This is a summary of how to take an input alignment and then run larch-usher on it to produce an hDAG.
We currently simply delete sites with ambiguous characters or gaps from the alignment.


## Install

To run this code, you will need to install the `dpvt-experiments` environment and the `dpvtex` package (`pip install -e .` in base folder of this repo).

Additionally, we use [larch](https://github.com/matsengrp/larch), which you will first need to build.
Follow the link to get instructions on how to build larch.
Note that this will require creating a new conda environment, so make sure that once you have installed `larch`, you activate `dpvt-experiments` again.


## Construct historydag with larch-usher and extract trees

This code allows to input an alignment in fasta format and returns a pickled file containing a dictionary with trees as keys and a list of 0s and 1s as values, which assign each edge (in preorder) value 0 if the edge is present in one of the MP trees found by larch-usher and otherwise returns 1.
To run this code, you can either follow the all-in-one description, which uses Snakemake to run all the code, or you can follow separate steps.


### All-in-one

First, we need to specify where we store input data, where we want output data to be stored, and some parameters for the run in `config.yaml`:
- `input_data`: directory containing for each alignment a directory with the name of the alignment and a fasta file with the same name as its directory, e.g.: `alignment_1/alignment_1.fasta`
- `larch_build`: path to the `build` directory created when building `larch` (see instructions in `larch` repo)
- `ouput_data`: path to directory in which output, which is a pickled directory containing trees and corresponding edge vectors containing MP edge labels, should be saved
- `num_larch_iterations`: number of iterations we want to run larch, defaults to `20`
- `num_cores`: number of cores used for running larch-usher and tree extraction, ideally matches `num_cores` that is provided for snakemake run (see below)

To execute the pipeline, run:
```bash
snakemake --cores <number of cores>
```


### Running steps individually

It sometimes might be useful to be able to run each step of the pipeline for creating the `dpvt` training data separately.
For example, we cannot call `larch-usher` from a snakefile on the Fred Hutch clusters.


#### Removing ambiguities

Before building a historydag with `larch-usher`, we clean input alignments by deleting all sites that contain gaps or ambiguous characters.
We use [biopython](https://biopython.org/) to delete ambiguous characters and gaps.
To do this, run

```bash
python scripts/remove_ambiguities.py /path/to/input/fasta /path/to/disambiguated/output/fasta
```

We later require the fasta file on which we run `larch-usher` to be called `input.fasta`, so it might be a good idea to call the output file for `input.fasta`.


#### Constructing an hDAG

To set up input for `larch-usher`, we first need to run the snakefile in `setup_larch_inputs` on our dataset.
We assume here that our alignment is named `input.fasta` and is located in the directory `/path/to/disambiguated/`.
We can then set up larch inputs by running:

```bash
cd setup_larch_inputs
snakemake --snakefile convert_fasta_to_larch_input.snakefile -d /path/to/disambiguated/input.fasta --cores <num_cores>
```

Note that we need to specify the number of cores `<num_cores>` here.

We are now ready to run larch-usher.
Therefore, navigate to the folder in which you built larch-usher and run larch-usher:

```bash
cd your-local-larch-dir/larch/build
./larch-usher -i  /path/to/disambiguated/fasta/output.pb -r /path/to/disambiguated/fasta/output.txt -v /path/to/disambiguated/fasta/output.vcf -o /path/to/output/protobuf -c <number of iterations> -l /path/to/log/directory/
```

This generates a protobuf `/path/to/output/protobuf` containing an hDAG that can then be used in the following to extract trees as training data for dpvt models.
The log will be written to the directory `/path/to/log/directory/`.

If you want to extend the larch-usher run, maybe because the Parsimony score still improved in one of the last iterations of the previous runs (which you can check in the log file), you can run:

```bash
./larch-usher -i /path/to/output/protobuf -o /path/to/extended/output/protobuf -c <number of iterations> -l /path/to/log/directory/ --trim
```


#### Extracting hDAG trees as training data

We extract trees from the given hDAG to get ete3 trees for training our dpvt models.
Make sure to navigate back into the `larch` folder of this repo before executing:

```bash
python extract_data_from_hdag.py /path/to/protobuf /path/to/dpvt/training/data
```

This script will read the hDAG in the protobuf in `/path/to/protobuf` that was output by larch-usher and pickles trees and edge labels as training/testing/validation data as required by `dpvt` in `/path/to/dpvt/training/data`.
The following steps are executed by this script:
- read the (compact genome) hDAG
- trim the hDAG to only contain MP trees
- convert the hDAG to a sequence dag
- sample from the hDAG without replacement -- currently samples as many trees are in the hDAG
- perturbs the sampled trees by replacing a subtree with depth `tree_depth/3` by a random tree
- labels edges by whether they are MP edges are not
- pickle trees and edge labels and safe to `/path/to/dpvt/training/data`