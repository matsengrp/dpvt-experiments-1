# README

In this README we describe how to use larch-usher to infer Maximum Parsimony
trees from input alignments and then perturb them to create training sets for
dpvt. We currently delete sites with ambiguous characters or gaps from the
alignment.

## Install

To run this code, you will need to install and activate the pixi
`dpvt-experiments` environment as explained [here](../../README.md).

Additionally, we use [larch](https://github.com/matsengrp/larch), which first
needs to be built. Follow the link to get instructions on how to build larch.
Note that this will require creating a conda environment, so make sure that once
you are done installing `larch`, you activate `dpvt-experiments` again to run
the code in this repo.

## Construct historydag with larch-usher and extract trees

This code allows to input an alignment in fasta format and returns a pickled
file containing a dictionary with trees as keys and lists of *0*s and *1*s as
values, which assign each edge in the corresponding tree (in preorder) value 0
if the edge is present in one of the MP trees found by larch-usher and otherwise
returns 1. To run this code, you can either follow the all-in-one description,
which uses Snakemake, or you can follow separate steps.

### Simulating alignments

If you want to simulate alignmnents for generating the training data, you can
use the script `scripts/create_alisim_alignments.sh`, which uses IQ-TREE's
alisim to simulate alignments
([http://www.iqtree.org/doc/AliSim](http://www.iqtree.org/doc/AliSim)). At the
beginning of this script you can specify a list of the number of alignments you
want to simulate, a list of number of sequences, and a list of sequence lengths
that you want to generate alignments for as well as a list of keywords
corresponding to different methods of introducing non-MP edges to the maximum
parsimony trees computed by larch (see Details in *Edge Distributions*). For
each combination of number of alignments, sequences, and sequence length, a
directory will be created in `dpvtex/data/simulated_alignments/` that contains a
directory for each alignment simulated, which itself contains that alignment.
The script additionally saves configs, one for each combination of simulated
dataset and edge distribution, which are needed for creating dpvt training data
(see in _All-in-one_) in the directory `configs/`. Note that the python script
`scripts/generate_sim_configs.py` is used to generate configs.

### Data generation pipeline

We provide a snakemake workflow that takes as input alignments and outputs data
in the format required by `dpvt` representing trees and labels on their edges,
identifying edges as MP and non-MP edges.

To provide the input and setting as to how to generate this data, the following
information needs to be provided in the snakemake config `config.yaml`:

- `input_data`: directory containing for each alignment a directory with the
    name of the alignment and a fasta file with the same name as its directory,
    e.g.: `alignment_1/alignment_1.fasta` _Note that we assume DNA sequences.
    Columns containing gaps or ambiguous characters in the input alignment get
    deleted in our pipeline._ The scripts `scripts/create_alisim_alignments.sh`
    creates data in exactly this format.
- `larch_build`: path to the `build` directory created when building `larch`
    (see instructions in `larch` repo)
- `output_data`: path to directory in which output, which is a pickled file
    containing trees and corresponding edge vectors containing MP edge labels,
    should be saved
- `dataset_name`: name for the dataset that will be used for the output files
    containing the data. The output files are named
    `{dataset_name}_{edge_distribution}.p`, where `{edge_distribution}`
    identifies the method that is used to introduce non-MP edges in the maximum
    parsimony trees returned by `larch` (see Details in `Edge Distributions`
    below)
- `edge_distribution`: Method used to introduce non-MP edges, see details in the
  `Edge Distributions` section below
- `num_cores`: number of cores used for running larch-usher and tree extraction,
    should match `num_cores` that is provided for snakemake run (see below)
- `remove_duplicate_site_patterns`: If set to True, duplicate site patterns are
  removed when computing dpvt datasets

To execute the pipeline, run the following in the `larch/` directory of this
repo:

```bash
snakemake --cores <number of cores>
```

If you have a special config file that you want to use, you can specify it with
`--configfile <configfile_path>`. If you simulated alignments as described
above, configs of the appropriate format are saved in the `configs/` directory
and can be used straight away to execute the pipeline for generating dpvt data.
You can then run the pipeline on all configs with:

```bash
./run_on_simulated_alignments.sh
```

The output of this snakemake run is a pickled file containing dictionaries with
ete3 trees as keys and lists as values, which indicate whether edges in the tree
are in a MP tree (*0*) or not (*1*), according to a preorder traversal.

### Edge Distributions

Initially in our pipeline, we run larch-usher on the provided alignments to
generate Maximum Parsimony trees. We therefore need to perturb these trees to
introduce non-MP edges. Since larch-usher provides us with a collection of
maximum parsimony trees, we can compare trees after perturbation with the
collection of larch-usher trees to ensure that the newly introduced edges are
actually not present in a maximum parsimony tree.

We present four different methods to introduce non-MP edges, which can be
specified with the corresponding keyword in the config file
(`edge_distribution`):

- `constant`: $min(\frac{n}{2}, 100)$ random SPR moves to create non-MP edges
- `uniform`: number of random SPRs drawn from uniform distribution between 0 and
  $min(n, 100)$
- `treesearch`: generate as many random trees as there are MP trees sampled. Of
  the MP trees, introduce $min(\frac{n}{2}, 100)$ SPR moves on half of these
  trees, and for the rest, draw the number of SPR moves from a uniform
  distribution between 0 and $min(n, 100)$
- `random_subtree`: replace random subtree of depth $\frac{d}{2}$, where d is depth of
  entire tree, by random tree and repeat until at least $\frac{1}{6}$ th of all edges
  (including pendant) are non-MP or random subtree replacement has been tried
  100 times



### Running all steps individually

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
            edges by performing SPR moves or replacing a subtree by a random
            tree
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
