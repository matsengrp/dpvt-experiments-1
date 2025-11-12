# README

In this README we describe how to use larch-usher to infer Maximum Parsimony
trees from input alignments and then perturb them to create training sets for
dpvt.

The pipeline consists of three phases:

1. **Phase 1: Preprocessing** - Clean alignments by removing low-quality
   sequences and generate quality statistics
2. **Phase 2: Dataset Preparation** - Filter alignments by quality metrics and
   split into train/test sets
3. **Phase 3: Training Data Generation** - Run larch-usher and extract training
   data

For detailed documentation, see [WORKFLOW_GUIDE.md](WORKFLOW_GUIDE.md).

## Install

To run this code, you will need to install and activate the pixi
`dpvt-experiments` environment as explained [here](../../README.md).

Additionally, we use [larch](https://github.com/matsengrp/larch) for inferring
maximum parsimony trees. There are two ways to install larch:

### Option 1: Install via conda (recommended)

The easiest way to install larch is through conda:

```bash
conda install -c matsengrp larch-phylo
```

This installs the `larch-phylo` executable. To use it with this pipeline (which
defaults to calling `larch`), add an alias to your `~/.bashrc`:

```bash
alias larch='larch-phylo'
```

Then reload your bashrc:

```bash
source ~/.bashrc
```

Alternatively, you can skip the alias and instead set
`larch_command: "larch-phylo"` in your config files.

### Option 2: Build from source

You can also build larch from source by following the instructions at the
[larch repository](https://github.com/matsengrp/larch). Note that this will
require creating a conda environment, so make sure that once you are done
installing `larch`, you activate `dpvt-experiments` again to run the code in
this repo.

After building larch from source, set up an alias to make the `larch-usher`
executable accessible as `larch`. Add the following line to your `~/.bashrc`
file:

```bash
alias larch='/path/to/larch/build/bin/larch-usher'
```

Replace `/path/to/larch/build/bin/larch-usher` with the actual path to your
larch-usher binary. After adding this line, reload your bashrc:

```bash
source ~/.bashrc
```

### Configuration note

The pipeline defaults to using the command `larch` to run larch (specified by
the `larch_command` parameter in config files). This works out-of-the-box if you
set up the alias as described above. If you prefer not to use an alias, you can
override this default by setting `larch_command: "larch-phylo"` or
`larch_command: "/path/to/larch/build/bin/larch-usher"` in your config files.

## Data generation pipeline

The pipeline takes empirical or simulated alignments as input and generates
pickled files containing dictionaries with trees as keys and lists of *0*s and
*1*s as values, which assign each edge in the corresponding tree (in preorder)
value 0 if the edge is present in one of the MP trees found by larch-usher and
otherwise returns 1.

The workflow is split into three phases with manual checkpoints between them to
review quality and filtering results. With these manual checkpoints we can
easily assess the suitability of empirical datasets for `dpvt` training and
testing. In the first step, sequences with too many gaps or ambiguous characters
are removed, and in the second phase all sites with gaps and ambiguous
characters are removed. The manual checkpoints after these phases are designed
for a manual check to decide whether enough data is left after these steps to
generate `dpvt` training data. The third and final phase consists of actually
generating dpvt training data by generating maximum parsimony trees with `larch`
and perturbing them to introduce non-MP edges.

> If alignments are simulated without gaps or ambiguous characters, the first
> two steps of the pipeline can be skipped.

### Simulating alignments

If you want to simulate alignmnents for generating the training data, you can
use the script `scripts/create_alisim_alignments.sh`, which uses IQ-TREE's
alisim to simulate alignments
([http://www.iqtree.org/doc/AliSim](http://www.iqtree.org/doc/AliSim)). At the
beginning of this script you can specify a list of the number of alignments you
want to simulate, a list of number of sequences, and a list of sequence lengths
that you want to generate alignments for as well as a list of keywords
corresponding to different methods of introducing non-MP edges to the maximum
parsimony trees computed by larch (see Details in _Edge Distributions_). For
each combination of number of alignments, sequences, and sequence length, a
directory will be created in `dpvtex/data/simulated_alignments/` that contains a
directory for each alignment simulated, which itself contains that alignment.
The script additionally saves configs, one for each combination of simulated
dataset and edge distribution, which are needed for creating dpvt training data
(see in _All-in-one_) in the directory `configs/`. Note that the python script
`scripts/generate_sim_configs.py` is used to generate configs.

### Running the three-phase workflow

The workflow uses separate snakefiles for each phase, with manual checkpoints in
between.

#### Configuration

Create a config file for Phases 1 & 2 (e.g.,
`configs/{my_dataset}_prepare.yaml`). See
`configs/simulated_15seq_20sites_100algnmnts_prepare.yaml` for an example using
simulated alignments. The config file needs to contain the following parameters:

**Phase 1 & 2 config parameters:**

- `dataset_name`: name for the dataset
- `input_data`: directory containing alignment subdirectories. Each alignment
  should be in its own directory with a `.fasta` or `.nex` file (e.g.,
  `alignment_1/alignment_1.fasta`)
- `max_ambiguous_site_frac_per_seq`: Maximum fraction of gaps/ambiguous
  characters allowed per sequence (default: 0.2). Sequences exceeding this
  threshold are removed.
- `remove_duplicate_site_patterns`: Whether to remove duplicate alignment
  columns (default: false)
- `output_datasets`: Base directory to save filtered datasets (will create new
  subdirectory in specified location)
- `min_frac_sites_retained`: Minimum fraction of sites an alignment must retain
  after preprocessing to be included in remaining analysis (default: 0.8)
- `create_train_test_split`: Whether to split filtered alignments into
  train/test sets (default: true)
- `test_fraction`: Fraction of alignments for test set (default: 0.2)

**Phase 3 config parameters** (separate configs for train/test):

- `input_data`: path to prepared dataset from Phase 2 (e.g.,
  `"../../data/{my_dataset}_train_0.8"`)
- `output_data` or `larch_output`: output directory for training data
- `dataset_name`: name used for output files (e.g., `"{my_dataset}_train_0.8"`)
- `larch_command`: command to run larch (default: `"larch"`). Can also be set to
  `"larch-phylo"` or a full path like `"/path/to/larch/build/bin/larch-usher"`
- `edge_distribution`: Method for introducing non-MP edges (see Edge
  Distributions section)
- `larch_timeout`: Maximum time in seconds for larch to run on each alignment
  (default: 1800 seconds / 30 minutes). Alignments that timeout are
  automatically excluded from downstream processing
- `balance_by_median_num_MP_trees`: Whether to balance the dataset by
  subsampling alignments with more trees than the median (default: true). This
  helps prevent alignments with many trees from dominating the training data
- `num_cores`: number of cores for larch-usher and tree extraction

You can generate a config with default values as specified above by running the
following:

```bash
cd dpvtex/larch
python scripts/generate_configs.py PATH/TO/INPUT_DATA DATASET_NAME
```

#### Phase 1: Preprocessing

```bash
cd dpvtex/larch
snakemake --snakefile preprocess_alignments.snakefile \
  --configfile configs/{my_dataset}.yaml \
  --cores 4
```

**Output**: Cleaned alignments and `alignment_size_stats.csv` with quality
metrics

**Manual Checkpoint**: Review statistics to decide `min_frac_sites_retained`
threshold

#### Phase 2: Dataset Preparation

```bash
snakemake --snakefile prepare_datasets.snakefile \
  --configfile configs/{my_dataset}.yaml \
  --cores 1
```

**Output**: Filtered datasets with train/test splits (symlinks to preprocessed
data)

**Manual Checkpoint**: Review manifest files to verify filtering results

#### Phase 3: Training Data Generation

Create simple configs for train and test sets (see
`configs/simulated_15seq_20sites_100algnmnts_train.yaml` and
`configs/simulated_15seq_20sites_100algnmnts_test.yaml` for examples).

Run for training set:

```bash
snakemake --snakefile generate_dpvt_input.snakefile \
  --configfile configs/{my_dataset}_train.yaml \
  --cores 8
```

Run for test set:

```bash
snakemake --snakefile generate_dpvt_input.snakefile \
  --configfile configs/{my_dataset}_test.yaml \
  --cores 8
```

**Output**:

- Pickled files with ete3 trees as keys and edge label lists as values (0 = MP
  edge, 1 = non-MP edge)
- `larch_timeout_alignments.txt`: List of alignments that timed out during larch
  processing (automatically excluded)
- Pipeline log file: `pipeline_log_{dataset_name}.jsonl` containing structured
  logs of all operations

### Automatic Filtering and Error Handling

The pipeline automatically handles and filters out problematic alignments:

1. **Unequal sequence lengths**: Alignments with sequences of different lengths
   are detected during preprocessing and listed in
   `unequal_length_alignments.txt`. These are automatically excluded from
   Phase 3.

2. **Larch timeouts**: If larch exceeds the configured timeout (default 30
   minutes) for an alignment, it is automatically recorded in
   `larch_timeout_alignments.txt` and excluded from downstream processing. The
   pipeline continues with remaining alignments.

3. **Larch failures**: If larch fails for any other reason, the error is logged
   to the pipeline log and the alignment is excluded from downstream processing.

All filtering is logged in the `pipeline_log_{dataset_name}.jsonl` file for
transparency and debugging.

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
- `random_subtree`: replace random subtree of depth $\frac{d}{2}$, where d is
  depth of entire tree, by random tree and repeat until at least $\frac{1}{6}$
  th of all edges (including pendant) are non-MP or random subtree replacement
  has been tried 100 times

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
python extract_data_from_hdag.py /path/to/protobuf /path/to/dpvt/training/data edge_distribution
```

This script reads the hDAG in the protobuf in `/path/to/protobuf` produced by
larch-usher and pickles trees and edge labels in a dictionary as required by
`dpvt` in `/path/to/dpvt/training/data`. The `edge_distribution` parameter
specifies how non-MP edges are introduced (see Edge Distributions section for
options: "constant", "uniform", "treesearch_mimic", or "random_subtree"). The
following steps are executed by this script:

- read the (compact genome) hDAG
- trim the hDAG to only contain MP trees
- convert the hDAG to a sequence dag
- unlabel hDAG and sample from it without replacement -- currently samples as
  many trees as there are in the hDAG
- perturbs the sampled trees by (i) performing `len(tree)/1` SPR moves on the
  tree (if `use_spr` is True) or (ii) replacing a subtree with depth
  `tree_depth/3` by a random tree
- labels edges by whether they are MP edges are not
- pickle trees and edge labels and safe to `/path/to/dpvt/training/data`

## More Details and file structure

Here we describe the purpose of each file in the `larch/` directory. Some of
this is already mentioned above, here we just present a more thorough summary
that this is mostly interesting for code development and should not be needed
for running the code.

In `dpvtex/larch`:

- `README.md`: Explains in detail how to run the code to create training data
  from alignments using larch
- `preprocess_alignments.snakefile`: Delete sites with gaps and ambiguous (i.e.
  non- ACGTacgt characters) from alignments. Additionally flags all directories
  with alignments have more than 5 characters to be included in further
  analysis. Alignments with less than 5 characters will be ignored.
- `generate_dpvt_input.snakefile`: inferring MP hDAG with larch, extracting MP
  trees, and perturbing them, creating pickled files of training and testing
  data that can be used as for `dpvt`.
- `Snakefile`: brings together `preprocess_alignments.snakefile` and
  `generate_dpvt_input.snakefile` for data as specified in `config.yaml`. This
  snakefile is required as some filtering of data happens in
  `preprocess_alignments.snakefile` results in wildcards changing: alignments
  with less than 5 sites after removing ambiguous characters and gaps will not
  be used.
- `config.yaml`: specifying input and output files and additional parameters
  needed in Snakefile. Description can be found in README
- `environment.yml`: larch-data environment required for running code

In `dpvtex/larch/scripts`: scripts called from snakefiles:

- Called from `preprocess_alignments.snakefile`:

  - `clean_data.py`: Removes all sites from alignments that contain characters
    that are not in {A,C,G,T,a,c,g,t}
  - `check_size_fasta.py`: Checks whether the alignment after removing gaps and
    ambiguous sites contains more than 5 sites. If it does, it creates a flag
    file in the directory containing the alignment, so it doesn't get used in
    future steps.

- Called from `generate_dpvt_input.snakefile`:
  - `pipeline_logger.py`: Provides structured logging functionality for tracking
    pipeline operations, errors, and data transformations. Creates JSONL log
    files for transparency and debugging.
  - `extract_data_from_hdag.py`: Reads hDAGs produced by larch and extracts MP
    trees:
    - read hDAG from larch output and trim to only contain MP trees + unlabel
      hDAG (i.e. remove internal sequences)
    - MP trees are sampled uniformly from hDAG without replacement -- we sample
      from trees, not histories, to not sample the same tree multiple times with
      different ancestral sequences
    - chooses random leaf of MP tree as root leaf
    - randomly resolve polytomies (using `resolve_polytomy()` function from
      ete3) **Note:** this function resolves multifurcations to a ladder tree
    - runs Sankoff to get sequences for internal nodes
    - use either `make_worse_spr()` ors `make_worse_tree()` function from
      `dpvtex/perfect_phylogenies/perturb_phylogeny.py` to create non-MP edges
      by performing SPR moves or replacing a subtree by a random tree
    - assigns `0`/`1` labels to edges of tree depending on whether the split
      represented by that edge is in the larch hDAG or not
    - if more than $\frac{1}{6}$ of the edges are non-MP or we tried more than
      100 times to create non-MP edges, stop and add tree and edge labels to
      dictionary
  - `aggregate_training_data.py`: Aggregate all training data saved separately
    for each input alignment in one dictionary. If
    `balance_by_median_num_MP_trees` is enabled (default), subsamples alignments
    with more trees than the median to prevent domination by high-tree-count
    alignments. Splits training and testing set to have the same ratio of MP to
    non-MP edges and saves training and testing data as pickled dictionaries.
    This will be the data for `dpvt`. Additionally creates csv file with
    properties of data: number of trees, number of leaves, number of MP edges,
    number of non-MP edges and alignment length (for training and testing set
    together).

In `dpvtex/larch/setup_larch_inputs`:

- `convert_fasta_to_larch_input.snakefile`: Read fasta file and create input
  files in format required to run larch -- called from
  `generate_dpvt_input.snakefile`

In `dpvtex/larch/tests`:

- testing functions from `clean_data.py` and `extract_data_from_hdag.py`. Can be
  run with `pytest`.
