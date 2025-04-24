# README

In this README we describe how to use larch-usher to infer Maximum Parsimony
trees from input alignments and then perturb them to create training sets for
dpvt. We currently delete sites with ambiguous characters or gaps from the
alignment.

## Install

To run this code, you will need to install the `larch-data` environment. This
can be done by running the following in the base folder of this repo:

```bash
    conda create env --file environment.yml
    pip install -e .
```

Additionally, we use [larch](https://github.com/matsengrp/larch), which first
needs to be built. Follow the link to get instructions on how to build larch.
Note that this will require creating a new conda environment, so make sure that
once you are done installing `larch`, you activate `larch-data` again to run the
code in this repo.

## Construct historydag with larch-usher and extract trees

This code allows to input an alignment in fasta format and returns a pickled
file containing a dictionary with trees as keys and lists of 0s and 1s as
values, which assign each edge in the corresponding tree (in preorder) value 0
if the edge is present in one of the MP trees found by larch-usher and otherwise
returns 1. To run this code, you can either follow the all-in-one description,
which uses Snakemake, or you can follow separate steps.

### Simulating alignments

If you want to simulate alignmnents for generating the training data, you can
use the script `scripts/create_alisim_alignments.sh`, which uses IQ-TREE's
alisim to simulate alignments
([http://www.iqtree.org/doc/AliSim](http://www.iqtree.org/doc/AliSim)). At the
beginning of this script you can specify the number of alignments you want to
simulate and a list of number of sequences and sequence lengths that you want to
generate alignments for. For each combination of number of sequences and
sequence length, a directory will be created in
`dpvtex/data/simulated_alignments/` that contains a directory for each alignment
simulated, which itself contains that alignment. The path to the directory with
all these alignments can then be added to `config.yaml` in the next step to
create training and testing data for `dpvt` (see below in _All-in-one_).

### All-in-one

First, we need to specify where we store input data, where we want output data
to be stored, and some parameters for the run in `config.yaml`:

-   `input_data`: directory containing for each alignment a directory with the
    name of the alignment and a fasta file with the same name as its directory,
    e.g.: `alignment_1/alignment_1.fasta`
     _Note that we assume DNA sequences. Columns containing gaps or ambiguous characters
    in the input alignment get deleted in our pipeline._
-   `larch_build`: path to the `build` directory created when building `larch`
    (see instructions in `larch` repo)
-   `output_data`: path to directory in which output, which is a pickled file
    containing trees and corresponding edge vectors containing MP edge labels,
    should be saved
-   `dataset_name`: name for the dataset that will be used for the output files
    containing training and testing data. The output files will be named
    `{dataset_name}_YYYY-MM-DD_train.p` and `{dataset_name}_YYYY-MM-DD_test.p`
-   `num_larch_iterations`: number of iterations we want to run larch, defaults
    to `20`. The number of iteration automatically increases by 5 if in the last
    5 iteration of larch there is a decrease in parsimony score.
-   `num_cores`: number of cores used for running larch-usher and tree
    extraction, should match `num_cores` that is provided for snakemake run (see
    below)
-   `make_worse_spr`: True or False depending on whether non-MP edges are
    supposed to be introduced by SPR moves (True) or by replacing MP subtrees by
    random subtrees (False)

To execute the pipeline, run the following in the `larch/` directory of this
repo:

```bash
snakemake --cores <number of cores>
```

If you have a special config file that you want to use, you can specify it with
`--configfile <configfile_path>`. If you used alisim to create a number of
alignments and config files, you can run the larch pipeline on all of those with
something like

```bash
for file in configs/*; do
    echo $file
    snakemake --cores <number of cores> --configfile $file
done
```

The output of this snakemake run will be a pickled file containing dictionaries
with ete3 trees as keys and lists as values, which indicate whether edges in the
tree are in a MP tree (0) or not (1), according to a preorder traversal.

### Running steps individually

In some cases we want to run each step of the pipeline for creating the `dpvt`
training data separately. For example, we cannot call `larch-usher` from a
snakefile on the Fred Hutch clusters.

In the following we describe the individual steps of the pipeline and how to
executed them independently of each other.

#### Removing ambiguities

Before building a historydag with `larch-usher`, we clean input alignments by
deleting all sites that contain gaps or ambiguous characters. We use
[biopython](https://biopython.org/) to delete ambiguous characters and gaps. To
do this, run

```bash
python scripts/remove_ambiguities.py /path/to/input/fasta /path/to/disambiguated/output/fasta
```

We later require the fasta file on which we run `larch-usher` to be called
`input.fasta`, so it might be a good idea to name the output file `input.fasta`.

#### Constructing an hDAG

To set up input for `larch-usher`, we first need to run the snakefile in
`setup_larch_inputs` on our dataset. We assume here that our alignment is named
`input.fasta` and is located in the directory `/path/to/disambiguated/`. We can
then set up larch inputs by running:

```bash
cd setup_larch_inputs
snakemake --snakefile convert_fasta_to_larch_input.snakefile -d /path/to/disambiguated/input.fasta -c1
```

Note that we specify the number of cores to be `1` here. Since all rules in this
particular snakefile need to run sequentially, there is no benefit in increasing
the number of cores.

We are now ready to run larch-usher. Therefore, navigate to the folder in which
you built larch-usher and run larch-usher:

```bash
cd your-local-larch-dir/larch/build
./larch-usher -i  /path/to/disambiguated/fasta/output.pb -r /path/to/disambiguated/fasta/output.txt -v /path/to/disambiguated/fasta/output.vcf -o /path/to/output/protobuf -c <number of iterations> -l /path/to/log/directory/
```

This generates a protobuf `/path/to/output/protobuf` containing an hDAG that can
then be used in the following to extract trees as training data for dpvt models.
The log will be written to the directory `/path/to/log/directory/`.

If you want to extend the larch-usher run, e.g. because the Parsimony Score
still improved in one of the last iterations of the previous runs (which you can
check in the log file), you can run:

```bash
./larch-usher -i /path/to/output/protobuf -o /path/to/extended/output/protobuf -c <number of iterations> -l /path/to/log/directory/ --trim
```

#### Extracting hDAG trees as training data

We extract trees from the given hDAG to get `ete3` trees for training our dpvt
models. Make sure to navigate back into the `larch` folder of this repo before
executing:

```bash
python extract_data_from_hdag.py /path/to/protobuf /path/to/dpvt/training/data use_spr
```

This script reads the hDAG in the protobuf in `/path/to/protobuf` produced by
larch-usher and pickles trees and edge labels in a dictionary as required by
`dpvt` in `/path/to/dpvt/training/data`. If `use_spr` is `True`, SPR moves are
used to create non-MP edges in the trees, otherwise MP subtrees are replaced by
random subtrees to create those edges. The following steps are executed by this
script:

-   read the (compact genome) hDAG
-   trim the hDAG to only contain MP trees
-   convert the hDAG to a sequence dag
-   unlabel hDAG and sample from it without replacement -- currently samples as
    many trees as there are in the hDAG
-   perturbs the sampled trees by (i) performing `len(tree)/1` SPR moves on the
    tree (if `use_spr` is True) or (ii) replacing a subtree with depth
    `tree_depth/3` by a random tree
-   labels edges by whether they are MP edges are not
-   pickle trees and edge labels and safe to `/path/to/dpvt/training/data`

## More Details and file structure

Here we describe the purpose of each file in the `larch/` directory. Some of
this is already mentioned above, here we just present a more thorough summary
that this is mostly interesting for code development and should not be needed
for running the code.

In `dpvtex/larch`:

-   `README.md`: Explains in detail how to run the code to create training data
    from alignments using larch
-   `preprocess_alignments.snakefile`: Delete sites with gaps and ambiguous
    (i.e. non- ACGTacgt characters) from alignments. Additionally flags all
    directories with alignments have more than 5 characters to be included in
    further analysis. Alignments with less than 5 characters will be ignored.
-   `generate_dpvt_input.snakefile`: inferring MP hDAG with larch, extracting MP
    trees, and perturbing them, creating pickled files of training and testing
    data that can be used as for `dpvt`.
-   `Snakefile`: brings together `preprocess_alignments.snakefile` and
    `generate_dpvt_input.snakefile` for data as specified in `config.yaml`. This
    snakefile is required as some filtering of data happens in
    `preprocess_alignments.snakefile` results in wildcards changing: alignments
    with less than 5 sites after removing ambiguous characters and gaps will not
    be used.
-   `config.yaml`: specifying input and output files and additional parameters
    needed in Snakefile. Description can be found in README
-   `environment.yml`: larch-data environment required for running code

In `dpvtex/larch/scripts`: scripts called from snakefiles:

-   Called from `preprocess_alignments.snakefile`:

    -   `clean_data.py`: Removes all sites from alignments that contain
        characters that are not in {A,C,G,T,a,c,g,t}
    -   `check_size_fasta.py`: Checks whether the alignment after removing gaps
        and ambiguous sites contains more than 5 sites. If it does, it creates a
        flag file in the directory containing the alignment, so it doesn't get
        used in future steps.

-   Called from `generate_dpvt_input.snakefile`:
    -   `check_max_parsimony.py`: Checks if the smallest parsimony score of the
        hDAG produced by larch decreased in the last 5 iterations. If it did, we
        want to run larch for more iterations
    -   `extract_data_from_hdag.py`: Reads hDAGs produced by larch and extracts
        MP trees:
        -   read hDAG from larch output and trim to only contain MP trees +
            unlabel hDAG (i.e. remove internal sequences)
        -   MP trees are sampled uniformly from hDAG without replacement -- we
            sample from trees, not histories, to not sample the same tree
            multiple times with different ancestral sequences
        -   chooses random leaf of MP tree as root leaf
        -   randomly resolve polytomies (using `resolve_polytomy()` function
            from ete3) **Note:** this function resolves multifurcations to a
            ladder tree
        -   runs Sankoff to get sequences for internal nodes
        -   use either `make_worse_spr()` ors `make_worse_tree()` function from
            `dpvtex/perfect_phylogenies/perturb_phylogeny.py` to create non-MP
            edges by performing SPR moves or replacing a subtree by a random tree
        -   assigns `0`/`1` labels to edges of tree depending on whether the
            split represented by that edge is in the larch hDAG or not
        -   if more than $\frac{1}{6}$ of the edges are non-MP or we tried more
            than 100 times to create non-MP edges, stop and add tree and edge
            labels to dictionary
    -   `aggregate_training_data.py`: Aggregate all training data saved
        separately for each input alignment in one dictionary, split training
        and testing set to have the same ratio of MP to non-MP edges and save
        training and testing data as pickled dictionaries. This will be the data
        for `dpvt`. Additionally creates csv file with properties of data:
        number of trees, number of leaves, number of MP edges, number of non-MP
        edges and alignment length (for training and testing set together).

In `dpvtex/larch/setup_larch_inputs`:

-   `convert_fasta_to_larch_input.snakefile`: Read fasta file and create input
    files in format required to run larch -- called from
    `generate_dpvt_input.snakefile`

In `dpvtex/larch/tests`:

-   testing functions from `clean_data.py` and `extract_data_from_hdag.py`. Can
    be run with `pytest`.
