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

This produces an output protobuf, which we can then read with the python [historydag](https://matsengrp.github.io/historydag/) package!